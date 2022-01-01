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
    float accmulator [ThreadTileT::M][ThreadTileT::N];

    // TODO: load accumlator init from C, assume C = 0 for now
    for(int m=0; m<ThreadTileT::M; ++m)
    {
        for(int n=0; n<ThreadTileT::N; ++n)
        {
            accmulator[m][n] = 0;
        }
    }

    __shared__ float tileA [CtaTileT::M][CtaTileT::K];
    __shared__ float tileB [CtaTileT::K][CtaTileT::N];

    for(int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K)
    {
        //1. load A, B to shmem by all threads in this CTA

        // 256 threads load 256 x 16 tile A from global memory
        // Each thread load a stipe of 1x16 
        int localTid = threadIdx.x * blockDim.y + threadIdx.y;
        int offsetM = blockIdx.x * CtaTileT::M + localTid;
        for(int k=0; k < CtaTileT::K/4; k++)
        {
            // Assuming A is row major
            int64_t addrA = offsetM * params.K + (ctaK + k*4);
            *reinterpret_cast<float4*>(&tileA[localTid][k*4]) = *reinterpret_cast<float4*>(&reinterpret_cast<float*>(params.ptrA)[addrA]);
        }

        // 256 threads load 16 x 256 tile B from global memory
        // Each thread load a stipe of 1x16 by vector load
        //          16 ele     |  --16 ele  |--------------
        //  row 0 | ---- T0----  --- T1 --- .... ---T15----
        //  row 1 | ---- T6----  ----T17-----....----T31---
        //   ...
        //  row 15

        int row = localTid / 16;
        int col = localTid % 16;
        int offsetK = row;
        for(int k=0; k < CtaTileT::K/4; k++)
        {
           int offsetN = blockIdx.y * CtaTileT::N+ 16*col + k*4;
           int addrB = (ctaK + offsetK)  * params.N + offsetN;
           *reinterpret_cast<float4*>(&tileB[offsetK][16*col+k*4]) = *reinterpret_cast<float4*>(&reinterpret_cast<float*>(params.ptrB)[addrB]);
        }

        //2. sync cta
        __syncthreads();

        constexpr int InstructionM = 8;
        constexpr int InstructionN = 8;
        //3. do shmem gemm by all threds in this CTA

        // 32/8
        for(int instructM =0; instructM < ThreadTileT::M; instructM += InstructionM)
        {
            // 8/8
            for(int instructN = 0; instructN < ThreadTileT::N; instructN += InstructionN)
            {
                for(int threadK = 0; threadK < CtaTileT::K; ++threadK)
                {
                    float fragementA [InstructionM];
                    float fragementB [InstructionN];
                    static_assert(ThreadTileT::K ==  1, "Only support ThreadTile::K ==1 for now");

                    // fragment A = load A from shmem
                    // fragment B = load B from shmem
                    int offsetM = threadIdx.x * ThreadTileT::M + instructM;
                    for(int m = 0; m < InstructionM; m++)
                    {
                        fragementA[m] = tileA[offsetM+m][threadK];
                    }

                    int offsetN = threadIdx.y * ThreadTileT::N + instructN;
                    for(int n = 0; n < InstructionN; n++)
                    {
                       fragementB[n] = tileB[threadK][offsetN+n];
                    }

                    for(int m = 0; m < InstructionM; m++)
                    {
                        for(int n=0; n < InstructionN; n++)
                        {
                            accmulator[instructM+m][instructN+n] += fragementA[m]* fragementB[n];
#if DEBUG
                            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
                            {
                                CUDA_LOG("accumlator[%d][%d] %f\n", m, n, accmulator[m][n]);
                            }
#endif
                        }
                    }
                }
            }
        }
        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
    };

    int offsetM = blockIdx.x * CtaTileT::M + threadIdx.x * ThreadTileT::M;
    int offsetN = blockIdx.y * CtaTileT::N + threadIdx.y * ThreadTileT::N;

    // D = alpha * C + beta
    #pragma unroll ThreadTileT::M
    for(int m=0; m<ThreadTileT::M; ++m)
    {
        #pragma unroll ThreadTileT::M
        for(int n=0; n<ThreadTileT::N; ++n)
        {
            accmulator[m][n] = params.alpha * accmulator[m][n] + params.beta;
            //store accumulator into global memory
            int globalOffsetD = (offsetM+m) * params.N + offsetN +n;
            reinterpret_cast<float*>(params.ptrD)[globalOffsetD] = accmulator[m][n];
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
    using ThreadTileShape = ThreadTile<32, 8, 1>;
    dim3 block = {8, 32};

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
            printf("i %d: Result: %f, Ref: %f\n", i, a, b);
            hasErr = true;
        }
    }
    printf(hasErr ? "Check Error\n" : "Check Pass\n");
    return hasErr;
}