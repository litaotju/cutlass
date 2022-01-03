#include <algorithm>
#include <cstdio>
#include "cutlass_gemm.h"
#include "cutlass/util/debug.h"
#include <exception>
#include <vector>


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
    constexpr int WARP_ROW = 2;
    constexpr int WARP_COL = 4;
    constexpr int WARP_TILE_M = CtaTileT::M/WARP_ROW;
    constexpr int WARP_TILE_N = CtaTileT::N/WARP_COL; 

    const int wrapStartM = (warpId / WARP_COL) * WARP_TILE_M;
    const int wrapStartN = (warpId % WARP_COL) * WARP_TILE_N;

    // 1 wrap computes: 128 * 64 elements

    // 1 thread computes 16 * 16 elements
    // this makes the thread structure
    //  8 * 4
    constexpr int THREADS_ROW = WARP_TILE_M/ ThreadTileT::M;
    constexpr int THREADS_COL = WARP_TILE_N/ ThreadTileT::N;
    static_assert(THREADS_ROW * THREADS_COL == 32, "A warp has 32 threads only");

    const int rowInsideWarp = laneId / THREADS_COL ;
    const int colInsideWarp = laneId % THREADS_COL; 
    constexpr int MmaTileM = 4;
    constexpr int MmaTileN = 4;

    const int threadStartM = wrapStartM + rowInsideWarp * MmaTileM;
    const int threadStartN = wrapStartN + colInsideWarp * MmaTileN;

    constexpr int threadStrideM = WARP_TILE_M/(ThreadTileT::M/MmaTileM);
    constexpr int threadStrideN = WARP_TILE_N/(ThreadTileT::N/MmaTileN);


    __shared__ float tileA [CtaTileT::K*CtaTileT::M];
    __shared__ float tileB [CtaTileT::K*CtaTileT::N];

    alignas(sizeof(float4)) float fragementA[MmaTileM];
    alignas(sizeof(float4)) float fragementB[MmaTileN];

    for(int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K)
    {
        //1. load A, B to shmem by all threads in this CTA

        // 256 threads load 128 x 16 tile A from global memory
        // Each thread load a stipe of 1x8
        int const &localTid = threadIdx.x;

        int offsetM = blockIdx.x * CtaTileT::M + localTid/2;

        for(int k=0; k < 8; k+=4)
        {
            int64_t addrA = offsetM * params.K + ctaK+(localTid%2)*8+k;
            *reinterpret_cast<float4*>(&fragementA[0]) = *reinterpret_cast<float4*>(&reinterpret_cast<float*>(params.ptrA)[addrA]);
            // Assuming A is row major
            for(int ik=0; ik < 4; ++ik)
            {
                tileA[((localTid%2)*8+k+ik) *CtaTileT::M+localTid/2] = fragementA[ik];
            }
        }

        // 256 threads load 16 x 128 tile B from global memory
        // Each thread load a stipe of 8x1
        int offsetN = blockIdx.y * CtaTileT::N + localTid%128;
        for(int k=0; k < 8; k+=4)
        {
           for(int ik=0; ik<4; ++ik)
           {
               int addrB = (ctaK+ (localTid/128)*8+k+ik)  * params.N + offsetN;
               fragementB[ik] =  reinterpret_cast<float*>(params.ptrB)[addrB];
               tileB[((localTid/128)*8+ k+ik)*CtaTileT::N+localTid%128] = fragementB[ik] ;
           }
        }

        //2. sync cta
        __syncthreads();

        //3. do shmem gemm by all threds in this CTA

        for(int threadK = 0; threadK < CtaTileT::K; ++threadK)
        {
            static_assert(ThreadTileT::K ==  1, "Only support ThreadTile::K ==1 for now");
            for(int m = 0; m < ThreadTileT::M/MmaTileM; m+=1)
            {
                *reinterpret_cast<float4*>(&fragementA[0]) = *reinterpret_cast<float4*>(&tileA[threadK*CtaTileT::M + threadStartM + m*threadStrideM]);
                for(int n = 0; n < ThreadTileT::N/MmaTileN; n+=1)
                {
                    *reinterpret_cast<float4*>(&fragementB[0]) = *reinterpret_cast<float4*>(&tileB[threadK*CtaTileT::N + threadStartN + n*threadStrideN]);
                    #pragma unroll MmaTileM
                    for(int i=0; i < MmaTileM; ++i)
                    {
                        #pragma unroll MmaTileN
                        for(int j=0; j < MmaTileN; ++j)
                        {
                            accmulator[m*MmaTileM+i][n*MmaTileN+j] += fragementA[i]*fragementB[j] ;
                        }
                    }
                }
            }
        }
        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
    };


    // D = alpha * C + beta
    for(int m=0; m< ThreadTileT::M/MmaTileM; m+= 1)
    {
        for(int n= 0; n< ThreadTileT::N/MmaTileN; n+=1)
        {
            for(int i=0; i < MmaTileM; ++i)
            {
                *reinterpret_cast<float4*>(&fragementB[0]) = *reinterpret_cast<float4*>(&accmulator[m*MmaTileM+i][n*MmaTileN]);
                for(int j=0; j < MmaTileN; ++j )
                {
                    fragementB[j] = params.alpha * fragementB[j] + params.beta;
                }
                int globalOffsetD = (ctaStartM + threadStartM + m*threadStrideM+i) * params.N + ctaStartN + threadStartN + n*threadStrideN;
                *reinterpret_cast<float4*>(&reinterpret_cast<float*>(params.ptrD)[globalOffsetD]) = *reinterpret_cast<float4*>(&fragementB[0]);
            }
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
    using CtaTileShape = CtaTile<128, 128, 16>;
    using ThreadTileShape = ThreadTile<8, 8, 1>;
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