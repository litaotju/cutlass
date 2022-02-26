#include "cutlass_gemm.h"
#include "cutlass/util/debug.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "mma.h"

#include <vector>
#include <cstdio>

using namespace nvcuda;

#if 0
template<typename CtaTileT, typename ThreadTileT>
__global__ void gemm_use_cutlass_apis(GemmParams params)
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

    using ShapeA = cutlass::layout::PitchLinearShape<CtaTileT::K, CtaTileT::M>;
    using ShapeB = cutlass::layout::PitchLinearShape<CtaTileT::N, CtaTileT::K>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = cutlass::half_t;
    static int const kThreads = 32;
    using ThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<ShapeA, kThreads>;
    using ThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<ShapeB, kThreads>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        ShapeA, Element, Layout, 0/*A is Row Major, advance over continous K*/, ThreadMapA>;
    typename IteratorA::Params paramsA(params.K);
    typename IteratorA::Fragment shmem_fragementA;
    cutlass::MatrixCoord tbOffsetA{ctaStartM, 0};
    IteratorA iteratorA(paramsA, reinterpret_cast<Element*>(params.ptrA), cutlass::make_Coord(params.M, params.K), threadIdx.x, tbOffsetA);
    shmem_fragementA.clear();
    iteratorA.load(shmem_fragementA);

    using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        ShapeB, Element, Layout, 1/*B is Row major, advance over strided K*/, ThreadMapB>;
    typename IteratorB::Params paramsB(CtaTileT::K);
    typename IteratorB::Fragment shmem_fragementB;
    cutlass::MatrixCoord tbOffsetB{0, ctaStartN};
    IteratorB iteratorB(paramsB, reinterpret_cast<Element*>(params.ptrB), cutlass::make_Coord(params.K, params.N), threadIdx.x, tbOffsetB);
    shmem_fragementB.clear();
    iteratorB.load(shmem_fragementB);
    ++iteratorA;
    ++iteratorB;

    for(int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K)
    {
        //2. sync cta
        __syncthreads();

        //3. do shmem gemm by all threds in this CTA

        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
        iteratorA.load(shmem_fragementA);
        iteratorB.load(shmem_fragementB);
        ++iteratorA;
        ++iteratorB;
    };

}
#endif // 0

//!Assume it's row major
template<int Threads, int blockSizeR, int blockSizeC, typename T, int VectorSize, bool ThreadMapRowMajor = true>
class GloablMemoryIterator
{
public:
    struct Coords
    {
        int x;
        int y;
    };

    static constexpr int accessPerRow = blockSizeC/VectorSize; // 16/8 == 2, or 128/8 == 16
    static constexpr int iteration = blockSizeR / (Threads / accessPerRow); // 128 / (256/2) == 1, or 16/(256/16) == 1
    static_assert(iteration == 1 && "Only support 1 iteration now");
    static constexpr int ThreadsCols = accessPerRow; // 2 or 16
    static constexpr int ThreadsRows = Threads / ThreadsCols ; // 256 / 2 = 128, or 256/16 = 16

    using Fragment = T[VectorSize];
    __device__ GloablMemoryIterator(int threadId, Coords blockOffset, int lda, void *address)
    {
        int threadMajor = ThreadMapRowMajor ?  threadId/ThreadsCols : threadId % ThreadsRows;
        int threadMinor = ThreadMapRowMajor ?  threadId%ThreadsCols : threadId / ThreadsRows;
        int64_t blockStart = (blockOffset.x + threadMajor) * lda + blockOffset.y;
        mThreadStart = blockStart+(threadMinor)*VectorSize;
        mAddress = reinterpret_cast<T*>(address);
    }

    __device__ void load(Fragment& fragment)
    {
        *reinterpret_cast<uint4*>(&fragment[0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(mAddress)[mThreadStart]);
    }
    int64_t mThreadStart;
    T* mAddress;
};

// D = alpha * (A @ B + C) + beta
// @ means matric multiply
template<int BlockSize, typename CtaTileT, typename WarpTileT>
__global__ void gemm(GemmParams params) {
    if (blockIdx.x >= divUp(params.M, CtaTileT::M) || blockIdx.y >= divUp(params.N, CtaTileT::N)) {
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
    constexpr int WARP_TILE_M = CtaTileT::M / WARP_ROW; // 128/2 = 64
    constexpr int WARP_TILE_N = CtaTileT::N / WARP_COL; // 128/4 = 32

    const int wrapStartM = (warpId / WARP_COL) * WARP_TILE_M;
    const int wrapStartN = (warpId % WARP_COL) * WARP_TILE_N;

    __shared__ half tileA[CtaTileT::M * CtaTileT::K];
    __shared__ half tileB[CtaTileT::N * CtaTileT::K];

    alignas(8 * sizeof(half)) half fragementA[8];
    alignas(8 * sizeof(half)) half fragementB[8];

    constexpr int WMMA_M = WarpTileT::M;
    constexpr int WMMA_N = WarpTileT::N;
    constexpr int WMMA_K = WarpTileT::K;

    wmma::fragment <wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment <wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[2];
    wmma::fragment <wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[
            WARP_TILE_M / WMMA_M][WARP_TILE_N / WMMA_N];

    static_assert(WMMA_K == CtaTileT::K, "Current implementation does not support WMMA_K != CtaTile::K yet");
    // Initialize the output to zero
    for (int m = 0; m < WARP_TILE_M / WMMA_M; ++m) {
        for (int n = 0; n < WARP_TILE_N / WMMA_N; ++n) {
            wmma::fill_fragment(c_frag[m][n], 0);
        }
    }


    for (int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K) {
        //1. load A, B to shmem by all threads in this CTA
        int const &localTid = threadIdx.x;

        // Assuming A is row major
        // 256 threads load 128 x 16 tile A from global memory
        // Each thread load a stipe of 1x8 (128 bits), this makes the threads inside a block structed as 128 x 2
        constexpr int VectorSize = 8;
        using IteratorA = GloablMemoryIterator<BlockSize, CtaTileT::M, CtaTileT::K, half, VectorSize>;
        IteratorA iteratorA(threadIdx.x, {blockIdx.x * CtaTileT::M, ctaK}, params.lda, params.ptrA);
        iteratorA.load(fragementA);
        int shmemAddressA = (localTid / IteratorA::ThreadsCols) * CtaTileT::K +
                            ((localTid % IteratorA::ThreadsCols) * VectorSize);
        *reinterpret_cast<uint4 *>(&tileA[shmemAddressA]) = *reinterpret_cast<uint4 *>(&fragementA[0]);


        // Assuming B is row major
        // 256 threads load 16 x 128 tile B from global memory
        // Each thread load a stipe of 1x8 (128 bits), this make the threads inside a block structured as: 16 x 16
        using IteratorB = GloablMemoryIterator<BlockSize, CtaTileT::K, CtaTileT::N, half, VectorSize, false/*ThreadMapRowMajor*/>;
        IteratorB iteratorB(threadIdx.x, {ctaK, blockIdx.y * CtaTileT::N}, params.ldb, params.ptrB);
        iteratorB.load(fragementB);

        for (int i = 0; i < sizeof(fragementB) / sizeof(half); ++i) {
            int shmemAddressB = ((localTid / IteratorB::ThreadsRows) * VectorSize + i) * CtaTileT::K +
                                (localTid % IteratorB::ThreadsRows);
            tileB[shmemAddressB] = fragementB[i];
        }

        //2. sync cta
        __syncthreads();

        wmma::load_matrix_sync(a_frag[0], &tileA[wrapStartM * CtaTileT::K], CtaTileT::K);
        wmma::load_matrix_sync(b_frag[0], &tileB[wrapStartN * CtaTileT::K], CtaTileT::K);
        for (int m = 0; m < WARP_TILE_M / WMMA_M; m += 1) {
            if (m < WARP_TILE_M / WMMA_M - 1) {
                wmma::load_matrix_sync(a_frag[(m + 1) % 2], &tileA[(wrapStartM + WMMA_M * (m + 1)) * CtaTileT::K],
                                       CtaTileT::K);
            }
            for (int n = 0; n < WARP_TILE_N / WMMA_N; n += 1) {
                // Load the inputs
                if (n < WARP_TILE_N / WMMA_N - 1) {
                    wmma::load_matrix_sync(b_frag[(n + 1) % 2],
                                           &tileB[(wrapStartN + WMMA_N * (n + 1)) * CtaTileT::K], CtaTileT::K);
                }
                // Perform the matrix multiplication
                wmma::mma_sync(c_frag[m][n], a_frag[m % 2], b_frag[n % 2], c_frag[m][n]);
            }
        }
        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
    };

    // D = alpha * C + beta
    for (int m = 0; m < WARP_TILE_M / WMMA_M; m += 1) {
        for (int n = 0; n < WARP_TILE_N / WMMA_N; n += 1) {
            int globalOffsetM = ctaStartM + wrapStartM + WMMA_M * m;
            int globalOffsetN = ctaStartN + wrapStartN + WMMA_N * n;
            auto c = reinterpret_cast<half *>(params.ptrD) + globalOffsetM * params.N + globalOffsetN;
            wmma::store_matrix_sync(c, c_frag[m][n], params.N, wmma::mem_row_major);
        }
    }
}

template<int BlockSize, typename CtaTileT, typename WarpTileT>
struct GemmKernel
{
    void operator()(GemmParams params, cudaStream_t stream)
    {
        auto const M = params.M;
        auto const N = params.N;
        dim3 grid = {divUp(M, CtaTileT::M), divUp(N, CtaTileT::N), 1};
        gemm<BlockSize, CtaTileT, WarpTileT><<<grid, BlockSize, 0, stream>>>(params);
    }
};

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
    GemmParams params {M, N, K, A, B, C, C, alpha, beta, K, N};

    using hgemm_128x128_16x16x16 = GemmKernel<256, CtaTile<128, 128, 16>, WarpTile<16, 16, 16>>;
    hgemm_128x128_16x16x16()(params, 0);
    //TODO: instance other kernel variants correctly
//    using hgemm_256x128_16x16x16 = GemmKernel<128, CtaTile<256, 128, 16>, WarpTile<16, 16, 16>>;
//    hgemm_256x128_16x16x16()(params, 0);

    constexpr int BLOCK_SIZE = 256;
    using CtaTileShape = CtaTile<128, 128, 16>;
    using ThreadTileShape = ThreadTile<8, 8, 1>;
    dim3 grid = {divUp(M, CtaTileShape::M), divUp(N, CtaTileShape::N), 1};
//    gemm_double_buffer_shmem<CtaTileShape, ThreadTileShape> <<<grid, BLOCK_SIZE, 0>>>(params);

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