#include "cutlass_gemm.h"
#include "cutlass/util/debug.h"
#include "mma.h"

#include <vector>
#include <cstdio>

using namespace nvcuda;

// D = alpha * (A @ B + C) + beta
// @ means matric multiply
template<typename CtaTileT, typename ThreadTileT>
__global__ void gemm(GemmParams params)
{
    if(blockIdx.x >= divUp (params.M, CtaTileT::M) || blockIdx.y >= divUp(params.N, CtaTileT::N))
    {
        return;
    }

    int const ctaStartM = blockIdx.x * CtaTileT::M;
    int const ctaStartN = blockIdx.y * CtaTileT::N;

    // 8 wrap: 2 x 4;
    int warpId = threadIdx.x / 32;
    assert(blockDim.x == 256);
    constexpr int NUM_WARPS = 8; // blockDim.x/32;
    constexpr int WARP_ROW = 2;
    constexpr int WARP_COL = 4;
    constexpr int WARP_TILE_M = CtaTileT::M/WARP_ROW; // 128/2 = 64
    constexpr int WARP_TILE_N = CtaTileT::N/WARP_COL; // 128/4 = 32

    const int wrapStartM = (warpId / WARP_COL) * WARP_TILE_M;
    const int wrapStartN = (warpId % WARP_COL) * WARP_TILE_N;

    __shared__ half tileA [CtaTileT::M*CtaTileT::K];
    __shared__ half tileB [CtaTileT::N*CtaTileT::K];

    alignas(8*sizeof(half)) half fragementA[8];
    alignas(8*sizeof(half)) half fragementB[8];

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[WARP_TILE_M/WMMA_M][WARP_TILE_N/WMMA_N];

    static_assert(WMMA_K == CtaTileT::K, "Current implementation does not support WMMA_K != CtaTile::K yet");
    // Initialize the output to zero
    for(int m = 0; m < WARP_TILE_M/WMMA_M ; ++m)
    {
        for(int n = 0; n < WARP_TILE_N/WMMA_N; ++n)
        {
            wmma::fill_fragment(c_frag[m][n], 0);
        }
    }

    for(int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K)
    {
        //1. load A, B to shmem by all threads in this CTA

        int const &localTid = threadIdx.x;

        int offsetM = blockIdx.x * CtaTileT::M + localTid/2;

        // Assuming A is row major
        // 256 threads load 128 x 16 tile A from global memory
        // Each thread load a stipe of 1x8 (128 bits), this makes the threads inside a block structed as 128 x 2
        int64_t addrA = offsetM * params.K + ctaK+(localTid%2)*8;
        *reinterpret_cast<uint4*>(&fragementA[0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(params.ptrA)[addrA]);
        *reinterpret_cast<uint4*>(&tileA[(localTid/2)*CtaTileT::K + ((localTid%2)*8)]) = *reinterpret_cast<uint4*>(&fragementA[0]);

        // Assuming B is row major
        // 256 threads load 16 x 128 tile B from global memory
        // Each thread load a stipe of 1x8 (128 bits), this make the threads inside a block structured as: 16 x 16
        //!!!! Note: thread row = tid % 16 , col = tid / 16, this reduced the number of bank conflicts
        //!!!!       do not use row = tid / 16, col = tid % 16, since the tile B in shmem is NxK.
        int offsetN = blockIdx.y * CtaTileT::N + (localTid/16)*8;
        int addrB = (ctaK+ (localTid%16))  * params.N + offsetN;
        *reinterpret_cast<uint4*>(&fragementB[0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(params.ptrB)[addrB]);
        for(int i=0; i < sizeof(fragementB)/sizeof(half); ++i)
        {
            tileB[((localTid/16)*8 + i)*CtaTileT::K + (localTid%16)] = fragementB[i] ;
        }

        //2. sync cta
        __syncthreads();

        //3. do shmem gemm by all threds in this CTA
        static_assert(ThreadTileT::K ==  1, "Only support ThreadTile::K ==1 for now");

        wmma::load_matrix_sync(a_frag[0], &tileA[wrapStartM*CtaTileT::K], CtaTileT::K);
        wmma::load_matrix_sync(b_frag[0], &tileB[wrapStartN*CtaTileT::K], CtaTileT::K);
        for(int m = 0; m < WARP_TILE_M/WMMA_M; m+=1)
        {
            if(m < WARP_TILE_M/WMMA_M-1)
            {
                wmma::load_matrix_sync(a_frag[(m+1)%2], &tileA[(wrapStartM + WMMA_M*(m+1))*CtaTileT::K], CtaTileT::K);
            }
            for(int n = 0; n < WARP_TILE_N/WMMA_N; n+=1)
            {
                // Load the inputs
                if( n < WARP_TILE_N/ WMMA_N-1)
                {
                    wmma::load_matrix_sync(b_frag[(n+1)%2], &tileB[(wrapStartN + WMMA_N*(n+1))*CtaTileT::K], CtaTileT::K);
                }
                // Perform the matrix multiplication
                wmma::mma_sync(c_frag[m][n], a_frag[m%2], b_frag[n%2], c_frag[m][n]);
            }
        }
        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
    };

    // D = alpha * C + beta
    for(int m=0; m< WARP_TILE_M/WMMA_M; m+= 1)
    {
        for(int n= 0; n< WARP_TILE_N/WMMA_N; n+=1)
        {
            int globalOffsetM = ctaStartM + wrapStartM  + WMMA_M * m;
            int globalOffsetN = ctaStartN + wrapStartN  + WMMA_N * n;
            auto c = reinterpret_cast<half*>(params.ptrD) + globalOffsetM * params.N + globalOffsetN;
            wmma::store_matrix_sync(c, c_frag[m][n], params.N, wmma::mem_row_major);
        }
    }
}

// D = alpha * (A @ B + C) + beta
// @ means matric multiply
template<typename CtaTileT, typename ThreadTileT>
__global__ void gemm_double_buffer_shmem(GemmParams params)
{
    if(blockIdx.x >= divUp (params.M, CtaTileT::M) || blockIdx.y >= divUp(params.N, CtaTileT::N))
    {
        return;
    }

    int const ctaStartM = blockIdx.x * CtaTileT::M;
    int const ctaStartN = blockIdx.y * CtaTileT::N;

    // 8 wrap: 2 x 4;
    int warpId = threadIdx.x / 32;
    assert(blockDim.x == 256);
    constexpr int NUM_WARPS = 8; // blockDim.x/32;
    constexpr int WARP_ROW = 2;
    constexpr int WARP_COL = 4;
    constexpr int WARP_TILE_M = CtaTileT::M/WARP_ROW; // 128/2 = 64
    constexpr int WARP_TILE_N = CtaTileT::N/WARP_COL; // 128/4 = 32

    const int wrapStartM = (warpId / WARP_COL) * WARP_TILE_M;
    const int wrapStartN = (warpId % WARP_COL) * WARP_TILE_N;

    __shared__ half tileA [2][CtaTileT::M*CtaTileT::K];
    __shared__ half tileB [2][CtaTileT::N*CtaTileT::K];

    alignas(8*sizeof(half)) half fragementA[8];
    alignas(8*sizeof(half)) half fragementB[8];

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[WARP_TILE_M/WMMA_M][WARP_TILE_N/WMMA_N];

    static_assert(WMMA_K == CtaTileT::K, "Current implementation does not support WMMA_K != CtaTile::K yet");
    // Initialize the output to zero
    for(int m = 0; m < WARP_TILE_M/WMMA_M ; ++m)
    {
        for(int n = 0; n < WARP_TILE_N/WMMA_N; ++n)
        {
            wmma::fill_fragment(c_frag[m][n], 0);
        }
    }

    int shmemBuffer = 1;
    for(int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K)
    {
        //1. load A, B to shmem by all threads in this CTA

        int const &localTid = threadIdx.x;

        int offsetM = blockIdx.x * CtaTileT::M + localTid/2;
        int offsetN = blockIdx.y * CtaTileT::N + (localTid/16)*8;

        // Assuming A is row major
        // 256 threads load 128 x 16 tile A from global memory
        // Each thread load a stipe of 1x8 (128 bits), this makes the threads inside a block structed as 128 x 2
        if(ctaK == 0)
        {
            int64_t addrA = offsetM * params.K + ctaK+(localTid%2)*8;
            *reinterpret_cast<uint4*>(&fragementA[0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(params.ptrA)[addrA]);
            *reinterpret_cast<uint4*>(&tileA[0][(localTid/2)*CtaTileT::K + ((localTid%2)*8)]) = *reinterpret_cast<uint4*>(&fragementA[0]);

            // Assuming B is row major
            // 256 threads load 16 x 128 tile B from global memory
            // Each thread load a stipe of 1x8 (128 bits), this make the threads inside a block structured as: 16 x 16
            //!!!! Note: thread row = tid % 16 , col = tid / 16, this reduced the number of bank conflicts
            //!!!!       do not use row = tid / 16, col = tid % 16, since the tile B in shmem is NxK.
            int addrB = (ctaK+ (localTid%16))  * params.N + offsetN;
            *reinterpret_cast<uint4*>(&fragementB[0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(params.ptrB)[addrB]);
            for(int i=0; i < sizeof(fragementB)/sizeof(half); ++i)
            {
                tileB[0][((localTid/16)*8 + i)*CtaTileT::K + (localTid%16)] = fragementB[i] ;
            }
        }

        //2. sync cta
        __syncthreads();

        // Prefetch data to shmem tile for next main loop iteration
        if(ctaK < params.K - CtaTileT::K)
        {
            int64_t addrA = offsetM * params.K + (ctaK + CtaTileT::K) +(localTid%2)*8;
            *reinterpret_cast<uint4*>(&fragementA[0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(params.ptrA)[addrA]);
            *reinterpret_cast<uint4*>(&tileA[shmemBuffer][(localTid/2)*CtaTileT::K + ((localTid%2)*8)]) = *reinterpret_cast<uint4*>(&fragementA[0]);

            int addrB = ((ctaK + CtaTileT::K)+ (localTid%16))  * params.N + offsetN;
            *reinterpret_cast<uint4*>(&fragementB[0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(params.ptrB)[addrB]);
            for(int i=0; i < sizeof(fragementB)/sizeof(half); ++i)
            {
                tileB[shmemBuffer][((localTid/16)*8 + i)*CtaTileT::K + (localTid%16)] = fragementB[i] ;
            }
        }

        //3. do shmem gemm by all threds in this CTA
        static_assert(ThreadTileT::K ==  1, "Only support ThreadTile::K ==1 for now");

        wmma::load_matrix_sync(a_frag[0], &tileA[1-shmemBuffer][wrapStartM*CtaTileT::K], CtaTileT::K);
        wmma::load_matrix_sync(b_frag[0], &tileB[1-shmemBuffer][wrapStartN*CtaTileT::K], CtaTileT::K);
        for(int m = 0; m < WARP_TILE_M/WMMA_M; m+=1)
        {
            if(m < WARP_TILE_M/WMMA_M-1)
            {
                wmma::load_matrix_sync(a_frag[(m+1)%2], &tileA[1-shmemBuffer][(wrapStartM + WMMA_M*(m+1))*CtaTileT::K], CtaTileT::K);
            }
            for(int n = 0; n < WARP_TILE_N/WMMA_N; n+=1)
            {
                // Load the inputs
                if( n < WARP_TILE_N/ WMMA_N-1)
                {
                    wmma::load_matrix_sync(b_frag[(n+1)%2], &tileB[1-shmemBuffer][(wrapStartN + WMMA_N*(n+1))*CtaTileT::K], CtaTileT::K);
                }
                // Perform the matrix multiplication
                wmma::mma_sync(c_frag[m][n], a_frag[m%2], b_frag[n%2], c_frag[m][n]);
            }
        }
        shmemBuffer = !shmemBuffer;
        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
    };

    // D = alpha * C + beta
    for(int m=0; m< WARP_TILE_M/WMMA_M; m+= 1)
    {
        for(int n= 0; n< WARP_TILE_N/WMMA_N; n+=1)
        {
            int globalOffsetM = ctaStartM + wrapStartM  + WMMA_M * m;
            int globalOffsetN = ctaStartN + wrapStartN  + WMMA_N * n;
            auto c = reinterpret_cast<half*>(params.ptrD) + globalOffsetM * params.N + globalOffsetN;
            wmma::store_matrix_sync(c, c_frag[m][n], params.N, wmma::mem_row_major);
        }
    }
}

int main()
{
    int M = 4096;
    int N = 4096;
    int K = 256;
    half *A ;
    half *B ;
    half *C ;
    half *RefC ;

    std::vector<half> hC (M*N, 0);
    std::vector<half> hRefC (M*N, -1);

    CUDA_PERROR(AllocateMatrix(&A, M, K, 10));
    CUDA_PERROR(AllocateMatrix(&B, K, N, 101));
    CUDA_PERROR(AllocateMatrix(&C, M, N, 10000));
    CUDA_PERROR(AllocateMatrix(&RefC, M, N, 103));

    float alpha = 1;
    //TODO: non-zero value will fail the ref check? Why
    float beta = 0;
    GemmParams params {M, N, K, A, B, C, C, alpha, beta};

    // All fixed.
    using CtaTileShape = CtaTile<128, 128, 16>;
    using ThreadTileShape = ThreadTile<8, 8, 1>;
    dim3 block = {256};

    dim3 grid = {divUp(M, CtaTileShape::M), divUp(N, CtaTileShape::N), 1};

    gemm<CtaTileShape, ThreadTileShape> <<<grid, block, 0>>>(params);
    gemm_double_buffer_shmem<CtaTileShape, ThreadTileShape> <<<grid, block, 0>>>(params);

    CUDA_PERROR(cudaDeviceSynchronize());
    CUDA_PERROR(cudaMemcpy(hC.data(), C, hC.size()*sizeof(half), cudaMemcpyDeviceToHost));

#if 1
    // cutlass Tensor core gemm has some bug here???
    CUDA_PERROR(CutlassHgemmTT_TensorCore(M, N, K, alpha, reinterpret_cast<cutlass::half_t const*>(A), K, 
                 reinterpret_cast<cutlass::half_t const*>(B), N, beta,  reinterpret_cast<cutlass::half_t*>(RefC), N));
#else
    CUDA_PERROR(CutlassHgemmTT(M, N, K, alpha, A, K, B, N, beta, RefC, N));
#endif

    CUDA_PERROR(cudaDeviceSynchronize());
    CUDA_PERROR(cudaMemcpy(hRefC.data(), RefC, hRefC.size()*sizeof(half), cudaMemcpyDeviceToHost));

    bool hasErr = false;
    for(int i=0; i < hC.size(); ++i)
    {
#if DEBUG
        printf("i %d: a: %f, b: %f\n", i, a, b);
#endif
        auto a = toFloat(hC[i]);
        auto b = toFloat(hRefC[i]);
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