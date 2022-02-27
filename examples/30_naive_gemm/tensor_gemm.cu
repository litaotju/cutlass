#include "cutlass_gemm.h"
#include "cutlass/util/debug.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "mma.h"
#include <string>

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

enum class Layout: int32_t
{
    RowMajor,
    ColMajor
};

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

template<typename T, int R, int C>
class Matrix
{
public:

    HOST_DEVICE Matrix(T* memory, Layout layout):
        mMemory(memory), mLayout(layout)
    {
        if(layout == Layout::RowMajor)
        {
            mStrides[0] = C;
            mStrides[1] = 1;
        }
        else
        {
            mStrides[0] = 1;
            mStrides[1] = R;
        }
    }

    HOST_DEVICE T& at(int r, int c)
    {
        return mMemory[r*mStrides[0] + c*mStrides[1]];
    }
    HOST_DEVICE Layout layout() const
    {
        return mLayout;
    }

    HOST_DEVICE int ldm() const
    {
        return layout() == Layout::RowMajor ? mStrides[0] : mStrides[1];
    }

    HOST_DEVICE void dump()
    {
        for(int r=0; r<R; ++r)
        {
            printf("row :%d -- ", r);
            for(int c=0; c<C;++c)
            {
                printf("%f ", toFloat(at(r,c)));
            }
            printf("\n");
        }
    }

    //! Set the non default ldm value, by default, the ctor will assume it's a packed dense layout
    //!!! Use with caustion. Please be sure about what you are doing.
    HOST_DEVICE void setLdm(int ldm)
    {
       if(layout() == Layout::RowMajor)
       {
           mStrides[0] = ldm;
       } 
       else
       {
           mStrides[1] = ldm;
       }
    }
private:
    //!
    //! TODO: only support packed layout for now, publish this ctor and support non packed layout
    //!
    HOST_DEVICE Matrix(T* memory, int64_t strides[2]):
            mMemory(memory), mLayout(Layout::RowMajor)
    {
        mStrides[0]  = strides[0];
        mStrides[1]  = strides[1];
    }
    T* mMemory;
    Layout mLayout;
    int64_t mStrides[2] = {1, 1};
};

//!
//! GlobalMemoryIterator
//!     A thread block collectively load a tile from global memory to registers, using vector load
//!
//! \tparam Threads : the number of threads participate in this collectively load
//! \tparam blockSizeR : block size in row
//! \tparam blockSizeC : block size in column
//! \tparam T : element type
//! \tparam VectorSize : the number of scalar (T) per Vector, one memory access is loading one Vector
//! \tparam ThreadMapRowMajor : how to arrange the threads
//!           this is a perf tuning parameter, does not affect the correction of the load.
//!       For example, RowMajor 16x16 means the threads are arranged like follows:
//              0  1  2 ... 15
//              16 17 18 ... 31
//              .. ...      ...
//              240 ..       255
//        ColMajor( ThreadMapRowMajor = false), 16x16 means threads are arrange like the follows:
//               0  16 32 ..  240
//               1  17 ...    241
//               2  .. ... .. 242
//               .. .. ...... ..
//               15 .. ...... 255
template<int Threads, int blockSizeR, int blockSizeC, typename T, int VectorSize, bool ThreadMapRowMajor = true>
class GloablMemoryIterator
{
public:
    struct Coords
    {
        int x;
        int y;
    };

    static_assert(blockSizeC % VectorSize == 0 && "Global memory is raw major, elements in a row must be divided by vector size");
    static_assert((blockSizeR*blockSizeC/VectorSize) % Threads == 0 && "The matrix must be able to equally parted by threads");

    static constexpr int kIteration = blockSizeR * (blockSizeC / VectorSize) / Threads; // 128 / (256/2) == 1, or 16/(256/16) == 1
    static constexpr int ThreadStrideR = blockSizeR / kIteration; // 16/2 == 8
    static constexpr int ThreadsCols = blockSizeC / VectorSize; // 256/8 == 32
    static_assert(ThreadsCols < Threads, "Only support for advancing in row dimension for now");

    static constexpr int ThreadsRows = Threads / ThreadsCols ; // 256 / 2 = 128, or 256/16 = 16, 256/32 == 8
    static_assert(std::is_same<T, half>::value, "Only support half for now");
    static_assert(VectorSize*sizeof(T)*8 == 128, "Only support 128 bits load for now");

    using MatrixType = Matrix<T, blockSizeR, blockSizeC>;

    using Fragment =  T[kIteration][VectorSize];
    HOST_DEVICE GloablMemoryIterator(int threadId, Coords blockOffset, Coords matrixSize, int lda, void *address):
        mTid{threadId}, mAddress{reinterpret_cast<T*>(address)}, mLda(lda), mBlockOffset(blockOffset), mMatrixSize(matrixSize)
    {
    }

    HOST_DEVICE void load(Fragment& fragment) const
    {
        #pragma unroll kIteration
        for (int i = 0; i < kIteration; ++i)
        { 
            //TODO: this assumes the global matrix is row major(A: MxK, B: KxN), add a variant to column major
            int const r = mBlockOffset.x+row()+i*ThreadStrideR;
            int const c = mBlockOffset.y+col()*VectorSize;
            if(r >= mMatrixSize.x || c >= mMatrixSize.y)
            {
            #define HANG_BUG_FIXED 0
            #if HANG_BUG_FIXED
                for(int v=0; i<VectorSize; ++v) 
                    fragment[i][v] = T(0);
            #endif
            }
            else
            {
                int64_t offset = r * mLda + c; 
                *reinterpret_cast<uint4*>(&fragment[i][0]) = *reinterpret_cast<uint4*>(&reinterpret_cast<half*>(mAddress)[offset]);
            }
        #if DEBUG
            if(blockIdx.x == 0 && blockIdx.y == 0 && ThreadsCols== 32)
            {
                if(threadIdx.x == 0 )
                {
                    printf("== tid: %d, iter:%d row:%d, col:%d, globalR:%d, globalC:%d, val:[%f %f] \n", threadIdx.x, i, row(), col(), r, c, toFloat(fragment[i][0]), toFloat(fragment[i][7]));
                }
            }
        #endif
        }
    }

    HOST_DEVICE void store(Fragment &fragment, MatrixType& dstMatrix) const
    {
        if(dstMatrix.layout() == Layout::RowMajor)
        {
            #pragma unroll kIteration
            for(int i=0; i<kIteration; ++i)
            {
                auto &dst = dstMatrix.at(row() + i*ThreadStrideR, col()*VectorSize);
                *reinterpret_cast<uint4 *>(&dst) = *reinterpret_cast<uint4 *>(&fragment[i][0]);
            }
        }
        else
        {
            #pragma unroll kIteration
            for(int i=0; i<kIteration; ++i)
            {
                int const r = mBlockOffset.x+row()+i*ThreadStrideR;
                int const c = mBlockOffset.y + col()*VectorSize;
        #if DEBUG
                if(blockIdx.x == 0 && blockIdx.y == 0)
                {
                    if(threadIdx.x == 0 )
                    {
                        printf("tid: %d, iter:%d row:%d, col:%d, globalR:%d, globalC:%d, val:[%f %f] \n", threadIdx.x, i, row(), col(), r, c, toFloat(fragment[i][0]), toFloat(fragment[i][7]));
                    }
                }
        #endif

                #pragma unroll VectorSize 
                for(int v=0; v < VectorSize; ++v)
                {
                    dstMatrix.at(row() + i*ThreadStrideR, col()*VectorSize+v) = fragment[i][v];
                }
            }
        }
    }

private:
    //! The row number of this thread in the structed block
    HOST_DEVICE int row() const
    {
        return ThreadMapRowMajor ?  mTid/ThreadsCols : mTid% ThreadsRows;
    }

    //! The col number of this thread in the structed block
    HOST_DEVICE int col() const 
    {
        return ThreadMapRowMajor ?  mTid%ThreadsCols : mTid/ ThreadsRows;
    }

    int mTid; // The thread id
    T* mAddress; // start adress of the complete matrix (not the start of this cta)
    int mLda; // lead dimension of the global memory
    Coords mBlockOffset; // This block's offset
    Coords mMatrixSize; // The complete matrix' size
};

// D = alpha * (A @ B + C) + beta
// @ means matric multiply
template<int BlockSize, typename CtaTileT, typename WarpTileT, bool prefetchA=true, bool prefetchB=true>
__global__ void gemm(GemmParams params) {
    int const ctaStartM = blockIdx.x * CtaTileT::M;
    int const ctaStartN = blockIdx.y * CtaTileT::N;
    if(ctaStartM >= params.M || ctaStartN >= params.N)
    {
        return;
    }

    // 8 wrap: 2 x 4;
    int warpId = threadIdx.x / 32;
    assert(blockDim.x == 256);
    constexpr int WARP_ROW = 2;
    constexpr int WARP_COL = 4;
    constexpr int WARP_TILE_M = CtaTileT::M / WARP_ROW; // The tile size in M computed by a warp
    constexpr int WARP_TILE_N = CtaTileT::N / WARP_COL; // The tile size in N computed by a warp

    constexpr int WMMA_M = WarpTileT::M;
    constexpr int WMMA_N = WarpTileT::N;
    constexpr int WMMA_K = WarpTileT::K;

    const int wrapStartM = (warpId / WARP_COL) * WARP_TILE_M;
    const int wrapStartN = (warpId % WARP_COL) * WARP_TILE_N;

    __shared__ half tileA[CtaTileT::M * CtaTileT::K];
    __shared__ half tileB[CtaTileT::N * CtaTileT::K];

    constexpr int MemoryAccessBits = 128;
    constexpr int VectorSize = MemoryAccessBits/(sizeof(half)*8);

    using IteratorA = GloablMemoryIterator<BlockSize, CtaTileT::M, CtaTileT::K, half, VectorSize>;
    // Using the RowMajorMap for B has worse performance.
    using IteratorB = GloablMemoryIterator<BlockSize, CtaTileT::K, CtaTileT::N, half, VectorSize, false/*ThreadMapRowMajor*/>;

    typename  IteratorA::MatrixType matrixA_shared(&tileA[0], Layout::RowMajor);
    typename  IteratorB::MatrixType matrixB_shared(&tileB[0], Layout::ColMajor);
    alignas(MemoryAccessBits/8) typename IteratorA::Fragment fragementA;
    alignas(MemoryAccessBits/8) typename IteratorB::Fragment fragementB;

    wmma::fragment <wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[prefetchA? 2: 1];
    wmma::fragment <wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[prefetchB? 2: 1];
    wmma::fragment <wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[
            WARP_TILE_M / WMMA_M][WARP_TILE_N / WMMA_N];

    wmma::fragment <wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag_epilogue[
            WARP_TILE_M / WMMA_M][WARP_TILE_N / WMMA_N];

    static_assert(CtaTileT::K % WMMA_K == 0);
    // Initialize the output to zero
    for (int m = 0; m < WARP_TILE_M / WMMA_M; ++m) {
        for (int n = 0; n < WARP_TILE_N / WMMA_N; ++n) {
            wmma::fill_fragment(c_frag[m][n], 0);
        }
    }

    for (int ctaK = 0; ctaK < params.K; ctaK += CtaTileT::K) {
        //1. load A, B to shmem by all threads in this CTA

        // Assuming A is row major
        IteratorA iteratorA(static_cast<int32_t>(threadIdx.x), { static_cast<int32_t>(blockIdx.x) * CtaTileT::M, ctaK}, {params.M, params.K}, params.lda, params.ptrA);
        iteratorA.load(fragementA);
        iteratorA.store(fragementA, matrixA_shared);

        // Assuming B is row major
        IteratorB iteratorB(static_cast<int32_t>(threadIdx.x), {ctaK, static_cast<int32_t>(blockIdx.y) * CtaTileT::N}, {params.K, params.N}, params.ldb, params.ptrB);
        iteratorB.load(fragementB);
        iteratorB.store(fragementB, matrixB_shared);

        //2. sync cta
        __syncthreads();

#if DEBUG
        if(ctaK == 0 && blockIdx.x ==0 && blockIdx.y == 0 && threadIdx.x == 0)
        {
            printf("iter M: %d, iter N: %d\n", WARP_TILE_M/WMMA_M, WARP_TILE_N/WMMA_N);
            printf("======== Matrix A ===============\n");
            matrixA_shared.dump();
            printf("======== Matrix B ===============\n");
            matrixB_shared.dump();
        }
#endif

        static_assert(WARP_TILE_N/WMMA_N <= 2 || !prefetchB, "TODO: the kernel will fail when warp n iteration > 2 and prefetch b");

        for(int wmmaK = 0; wmmaK < CtaTileT::K;  wmmaK += WMMA_K)
        {
            if(prefetchA) {
                wmma::load_matrix_sync(a_frag[0], &matrixA_shared.at(wrapStartM, wmmaK), matrixA_shared.ldm());
            }
            if(prefetchB) {
                wmma::load_matrix_sync(b_frag[0], &matrixB_shared.at(wmmaK, wrapStartN), matrixB_shared.ldm());
            }
            for (int m = 0; m < WARP_TILE_M / WMMA_M; m += 1) {
                if(prefetchA) {
                    if (m < WARP_TILE_M / WMMA_M - 1) {
                        wmma::load_matrix_sync(a_frag[(m + 1) % 2], &matrixA_shared.at(wrapStartM + WMMA_M * (m + 1), wmmaK), matrixA_shared.ldm());
                    }
                    for (int n = 0; n < WARP_TILE_N / WMMA_N; n += 1) {
                        if(prefetchB)
                        {
                            // Load the inputs
                            if (n < WARP_TILE_N / WMMA_N - 1) {
                                wmma::load_matrix_sync(b_frag[(n + 1) % 2],
                                                    &matrixB_shared.at(wmmaK, wrapStartN + WMMA_N * (n + 1)), matrixB_shared.ldm());
                            }
                            // Perform the matrix multiplication
                            wmma::mma_sync(c_frag[m][n], a_frag[m % 2], b_frag[n % 2], c_frag[m][n]);
                        }
                        else
                        {
                            wmma::load_matrix_sync(b_frag[0], &matrixB_shared.at(wmmaK, wrapStartN + WMMA_N*n), matrixB_shared.ldm());
                            wmma::mma_sync(c_frag[m][n], a_frag[m % 2], b_frag[0], c_frag[m][n]);
                        }
                    }
                } else {
                    for (int n = 0; n < WARP_TILE_N / WMMA_N; n += 1) {
                        wmma::load_matrix_sync(a_frag[0], &matrixA_shared.at(wrapStartM + WMMA_M*m,  wmmaK), matrixA_shared.ldm());
                        wmma::load_matrix_sync(b_frag[0], &matrixB_shared.at(wmmaK, wrapStartN + WMMA_N*n), matrixB_shared.ldm());
                        wmma::mma_sync(c_frag[m][n], a_frag[0], b_frag[0], c_frag[m][n]);
                    }
                }
            }
        }
        //This is essential, otherwise, some threads will override the shared memory while other ones using it
        __syncthreads();
    };

    // D = alpha * C + beta
    for (int m = 0; m < WARP_TILE_M / WMMA_M; m += 1) {
        for (int n = 0; n < WARP_TILE_N / WMMA_N; n += 1) {
            for(int rr=0; rr < c_frag[m][n].num_elements; ++rr)
            {
                c_frag_epilogue[m][n].x[rr] = fromFloat<half>(c_frag[m][n].x[rr]);
            }
            int globalOffsetM = ctaStartM + wrapStartM + WMMA_M * m;
            int globalOffsetN = ctaStartN + wrapStartN + WMMA_N * n;
            if(globalOffsetM >= params.M || globalOffsetN >= params.N)
            {
                continue;
            }
            auto c = reinterpret_cast<half *>(params.ptrD) + globalOffsetM * params.N + globalOffsetN;
            wmma::store_matrix_sync(c, c_frag_epilogue[m][n], params.N, wmma::mem_row_major);
        }
    }
}

template<int BlockSize, typename CtaTileT, typename WarpTileT, bool prefetchA=true, bool prefetchB=true>
struct GemmKernel
{
    void operator()(GemmParams params, cudaStream_t stream)
    {
        auto const M = params.M;
        auto const N = params.N;
        dim3 grid = {static_cast<uint32_t>(divUp(M, CtaTileT::M)), static_cast<uint32_t>(divUp(N, CtaTileT::N)), 1};
        gemm<BlockSize, CtaTileT, WarpTileT, prefetchA, prefetchB><<<grid, BlockSize, 0, stream>>>(params);
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

//! return true if the kernel can pass ref check with cutclass
template<typename Kernel>
bool testKernel(Kernel kernelFunc, int M, int N, int K)
{
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
    kernelFunc(params, nullptr/*stream*/);

    CUDA_PERROR(cudaDeviceSynchronize());
    CUDA_PERROR(cudaMemcpy(hC.data(), C, hC.size()*sizeof(half), cudaMemcpyDeviceToHost));

    constexpr bool CUTLASS_USE_TENSOR_CORE {true};
    if(CUTLASS_USE_TENSOR_CORE)
    {
        // cutlass Tensor core gemm has some bug here???
        CUDA_PERROR(CutlassHgemmTT_TensorCore(M, N, K, alpha, reinterpret_cast<cutlass::half_t const*>(A), K, 
                    reinterpret_cast<cutlass::half_t const*>(B), N, beta,  reinterpret_cast<cutlass::half_t*>(RefC), N));
    }
    else
    {
        CUDA_PERROR(CutlassHgemmTT(M, N, K, alpha, A, K, B, N, beta, RefC, N));
    }

    CUDA_PERROR(cudaDeviceSynchronize());
    CUDA_PERROR(cudaMemcpy(hRefC.data(), RefC, hRefC.size()*sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_PERROR(cudaFree(A));
    CUDA_PERROR(cudaFree(B));
    CUDA_PERROR(cudaFree(C));
    CUDA_PERROR(cudaFree(RefC));

    bool hasErr = false;
    for(int i=0; i < hC.size(); ++i)
    {
        auto a = toFloat(hC[i]);
        auto b = toFloat(hRefC[i]);
        auto const err = std::abs(a-b);
        if(err >=1e-3 && err >= 0.05*std::max(std::abs(a), std::abs(b)))
        {
#if DEBUG
            printf("i %d: Result: %f, Ref: %f\n", i, a, b);
#endif
            hasErr = true;
        }
    }
    return !hasErr;
}

int main(int argc, char**argv)
{
    int M = 4096;
    int N = 4096;
    int K = 256;

    for(int argIndex=0; argIndex<argc; ++argIndex)
    {
        if(std::string(argv[argIndex]) == "-m")
        {
            assert(argIndex+1 < argc);
            M = std::stoi(argv[argIndex+1]);
        }
        else if(std::string(argv[argIndex]) == "-n")
        {
            assert(argIndex+1 < argc);
            N = std::stoi(argv[argIndex+1]);
        }
        else if(std::string(argv[argIndex]) == "-k")
        {
            assert(argIndex+1 < argc);
            K = std::stoi(argv[argIndex+1]);
        }
    }

    using hgemm_128x128_16x16x16 = GemmKernel<256, CtaTile<128, 128, 16>, WarpTile<16, 16, 16>>;
    using hgemm_256x128_16x16x16 = GemmKernel<256, CtaTile<256, 128, 16>, WarpTile<16, 16, 16>>;
    using hgemm_128x256_16x16x16 = GemmKernel<256, CtaTile<128, 256, 16>, WarpTile<16, 16, 16>, true, false>;
    using hgemm_256x256_16x16x16 = GemmKernel<256, CtaTile<256, 256, 16>, WarpTile<16, 16, 16>, true, false>;
    auto double_buffer_kernel = [](GemmParams params, cudaStream_t stream)
    {
        constexpr int BLOCK_SIZE = 256;
        using CtaTileShape = CtaTile<128, 128, 16>;
        using ThreadTileShape = ThreadTile<8, 8, 1>;
        dim3 grid = {static_cast<uint32_t>(divUp(params.M, CtaTileShape::M)), static_cast<uint32_t>(divUp(params.N, CtaTileShape::N)), 1};
        gemm_double_buffer_shmem<CtaTileShape, ThreadTileShape> <<<grid, BLOCK_SIZE, 0>>>(params);
    };

#define TEST_KERNEL(Kernel) \
    do { \
        printf("Test %s with [M,N,K] as [%d,%d,%d] ", #Kernel, M, N, K); \
        bool passed = testKernel(Kernel, M, N, K); \
        printf("%s\n", (passed ? "PASSED": "FAILED")); \
    } while(false)

    TEST_KERNEL(hgemm_128x128_16x16x16());
    TEST_KERNEL(hgemm_256x128_16x16x16());
    TEST_KERNEL(hgemm_128x256_16x16x16());
    TEST_KERNEL(hgemm_256x256_16x16x16());

#define DOUBLE_BUFFER_PARTIAL_CTA_FIXED 0
#if DOUBLE_BUFFER_PARTIAL_CTA_FIXED
    TEST_KERNEL(double_buffer_kernel);
#endif

    using hgemm_512x512_16x16x16 = GemmKernel<256, CtaTile<512, 512, 16>, WarpTile<16, 16, 16>, true, false>;
    TEST_KERNEL(hgemm_512x512_16x16x16());

    using hgemm_128x128x32_16x16x16 = GemmKernel<256, CtaTile<128, 128, 32>, WarpTile<16, 16, 16>>;
    TEST_KERNEL(hgemm_128x128x32_16x16x16());

#undef TEST_KERNEL
}