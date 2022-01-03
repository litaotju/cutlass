#include <algorithm>
#include <cstdio>
#include "cutlass_gemm.h"
#include "cutlass/util/debug.h"
#include <exception>
#include <vector>

struct GemmParams
{
    int M;
    int N;
    int K;
    void *ptrA;
    void *ptrB;
    void *ptrC;
    void *ptrD; 
    float alpha;
    float beta;
};

template<int M_, int N_, int K_>
struct CtaTile
{
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;
};

template<int M_, int N_, int K_>
struct ThreadTile
{
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;
};

__host__ __device__ int divUp(int a, int b)
{
    return (a+b-1)/b;
}

#define DEBUG 0

// D = alpha * (A @ B + C) + beta
// @ means matric multiply
template<typename CtaTileT, typename ThreadTileT>
__global__ void gemm(GemmParams params)
{
    if(blockIdx.x >= divUp (params.M, CtaTileT::M) || blockIdx.y >= divUp(params.N, CtaTileT::N))
    {
        return;
    }
    alignas(sizeof(float4)) float accmulator [ThreadTileT::M][ThreadTileT::N];

    // TODO: load accumlator init from C, assume C = 0 for now
    for(int m=0; m<ThreadTileT::M; ++m)
    {
        for(int n=0; n<ThreadTileT::N; ++n)
        {
            accmulator[m][n] = 0;
        }
    }

    int const ctaStartM = blockIdx.x * CtaTileT::M;
    int const ctaStartN = blockIdx.y * CtaTileT::N;

    // 8 wrap: 2 x 4;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    assert(blockDim.x == 256);
    constexpr int NUM_WARPS = 8; // blockDim.x/32;
    constexpr int WRAP_ROW = 2;
    constexpr int WRAP_COL = 4;
    constexpr int WARP_TILE_M = CtaTileT::M/WRAP_ROW;
    constexpr int WARP_TILE_N = CtaTileT::N/WRAP_COL; 

    const int wrapStartM = (warpId / 4) * WARP_TILE_M;
    const int wrapStartN = (warpId % 4) * WARP_TILE_N;

    // 1 wrap computes: 128 * 64 elements
    // 1 thread computes 16 * 16 elements
    // this makes the thread structure
    //  8 * 4
    const int rowInsideWarp = laneId / 4;
    const int colInsideWarp = laneId % 4; 
    const int threadStartM = wrapStartM + rowInsideWarp * ThreadTileT::M;
    const int threadStartN = wrapStartN + colInsideWarp * ThreadTileT::N;

    constexpr int MmaTileM = 4;
    constexpr int MmaTileN = 4;
    __shared__ float tileA [CtaTileT::K*CtaTileT::M];
    __shared__ float tileB [CtaTileT::K*CtaTileT::N];

    alignas(sizeof(float4)) float fragementA[MmaTileM];
    alignas(sizeof(float4)) float fragementB[MmaTileN];

    for(int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K)
    {
        //1. load A, B to shmem by all threads in this CTA

        // 256 threads load 256 x 16 tile A from global memory
        // Each thread load a stipe of 1x16 
        int const &localTid = threadIdx.x;

        int offsetM = blockIdx.x * CtaTileT::M + localTid;

        for(int k=0; k < CtaTileT::K/4; k+=1)
        {
            alignas(sizeof(float4)) float xa[4];
            int64_t addrA = offsetM * params.K + ctaK + k*4;
            *reinterpret_cast<float4*>(&xa[0]) = *reinterpret_cast<float4*>(&reinterpret_cast<float*>(params.ptrA)[addrA]);
            // Assuming A is row major
            for(int ik=0; ik < 4; ++ik)
            {
                tileA[(k*4+ik) *CtaTileT::M+localTid] = reinterpret_cast<float*>(&xa)[ik];
            }
        }

        // 256 threads load 16 x 256 tile B from global memory
        // Each thread load a stipe of 16x1
        int offsetN = blockIdx.y * CtaTileT::N + localTid;
        for(int k=0; k < CtaTileT::K; k+=4)
        {
           for(int ik=0; ik<4; ++ik)
           {
               int addrB = (ctaK + k+ik)  * params.N + offsetN;
               tileB[(k+ik)*CtaTileT::N+localTid] = reinterpret_cast<float*>(params.ptrB)[addrB];
           }
        }

        //2. sync cta
        __syncthreads();

        //3. do shmem gemm by all threds in this CTA

        for(int threadK = 0; threadK < CtaTileT::K; ++threadK)
        {
            static_assert(ThreadTileT::K ==  1, "Only support ThreadTile::K ==1 for now");
            for(int m = 0; m < ThreadTileT::M; m+=MmaTileM)
            {
                *reinterpret_cast<float4*>(&fragementA[0]) = *reinterpret_cast<float4*>(&tileA[threadK*CtaTileT::M + threadStartM + m]);
                for(int n = 0; n < ThreadTileT::N; n+=MmaTileN)
                {
                    *reinterpret_cast<float4*>(&fragementB[0]) = *reinterpret_cast<float4*>(&tileB[threadK*CtaTileT::N + threadStartN + n]);
                    #pragma unroll MmaTileM
                    for(int i=0; i < MmaTileM; ++i)
                    {
                        #pragma unroll MmaTileN
                        for(int j=0; j < MmaTileN; ++j)
                        {
                            accmulator[m+i][n+j] += fragementA[i]*fragementB[j] ;
                        }
                    }
                }
            }
        }
        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
    };


    // D = alpha * C + beta
    for(int m=0; m< ThreadTileT::M; m+= 1)
    {
        for(int n= 0; n< ThreadTileT::N; n+=MmaTileN)
        {
            float4 &x = *reinterpret_cast<float4*>(&accmulator[m][n]);
            x.x = params.alpha * x.x + params.beta;
            x.y = params.alpha * x.y + params.beta;
            x.z = params.alpha * x.z + params.beta;
            x.w = params.alpha * x.w + params.beta;
            int globalOffsetD = (ctaStartM + threadStartM +m) * params.N + ctaStartN + threadStartN + n;
            *reinterpret_cast<float4*>(&reinterpret_cast<float*>(params.ptrD)[globalOffsetD]) = x;
        }
    }
}


int main()
{
    int M = 4096;
    int N = 4096;
    int K = 256;
    float *A ;
    float *B ;
    float *C ;
    float *RefC ;

    std::vector<float> hC (M*N, 0);
    std::vector<float> hRefC (M*N, -1);

    CUDA_PERROR(AllocateMatrix(&A, M, K, 10));
    CUDA_PERROR(AllocateMatrix(&B, K, N, 101));
    CUDA_PERROR(AllocateMatrix(&C, M, N, 10000));
    CUDA_PERROR(AllocateMatrix(&RefC, M, N, 103));

    float alpha = 1;
    float beta = 0;
    GemmParams params {M, N, K, A, B, C, C, alpha, beta};

    // All fixed.
    using CtaTileShape = CtaTile<256, 256, 16>;
    using ThreadTileShape = ThreadTile<16, 16, 1>;
    dim3 block = {256};

    dim3 grid = {divUp(M, CtaTileShape::M), divUp(N, CtaTileShape::N), 1};

    gemm<CtaTileShape, ThreadTileShape> <<<grid, block, 0>>>(params);
    CUDA_PERROR(cudaDeviceSynchronize());
    CUDA_PERROR(cudaMemcpy(hC.data(), C, hC.size()*sizeof(float), cudaMemcpyDeviceToHost));

    CutlassSgemmTT(M, N, K, alpha, A, K, B, N, beta, RefC, N);
    CUDA_PERROR(cudaDeviceSynchronize());
    CUDA_PERROR(cudaMemcpy(hRefC.data(), RefC, hRefC.size()*sizeof(float), cudaMemcpyDeviceToHost));

    bool hasErr = false;
    for(int i=0; i < hC.size(); ++i)
    {
#if DEBUG
        printf("i %d: a: %f, b: %f\n", i, a, b);
#endif
        auto a = hC[i];
        auto b = hRefC[i];
        auto const err = std::abs(a-b);
        if(err >=1e-3 && err >= 0.05*std::max(std::abs(a), std::abs(b)))
        {
            // printf("i %d: Result: %f, Ref: %f\n", i, a, b);
            hasErr = true;
        }
    }
    printf(hasErr ? "Check Error\n" : "Check Pass\n");
    return hasErr;
}