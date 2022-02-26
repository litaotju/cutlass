#pragma once


#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <iostream>
#include "cuda_fp16.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

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
    int lda;
    int ldb;
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

inline __host__ __device__ int divUp(int a, int b)
{
    return (a+b-1)/b;
}

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmTT(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassHgemmTT(
  int M,
  int N,
  int K,
  float alpha,
  half const *A,
  int lda,
  half const *B,
  int ldb,
  float beta,
  half *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using RowMajor = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<half,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  half,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  half,        // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassHgemmTT_TensorCore(
  int M,
  int N,
  int K,
  float alpha,
  cutlass::half_t const *A,
  int lda,
  cutlass::half_t const *B,
  int ldb,
  float beta,
  cutlass::half_t *C,
  int ldc) {

  using ElementOutput = cutlass::half_t;
  using ElementComputeEpilogue = float;

  // The code section below describes matrix layout of input and output matrices. Column Major for
  // Matrix A, Row Major for Matrix B and Row Major for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm70;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
      cutlass::gemm::GemmShape<128, 128, 32>;
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

  // This code section describes the epilogue part of the kernel
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,                                     // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                         // memory access. For a byte, it's 16
                                                         // elements. This becomes the vector width of
                                                         // math instructions in the epilogue too
      float,                                // <- data type of accumulator
      ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

  // Number of pipelines you want to use
  constexpr int NumStages = 2;

  using Gemm = cutlass::gemm::device::Gemm<cutlass::half_t,
                                           LayoutInputA,
                                           cutlass::half_t,
                                           LayoutInputB,
                                           cutlass::half_t,
                                           LayoutOutput,
                                           float,
                                           MMAOp,
                                           SmArch,
                                           ShapeMMAThreadBlock,
                                           ShapeMMAWarp,
                                           ShapeMMAOp,
                                           EpilogueOp, 
                                           SwizzleThreadBlock,
                                           NumStages>;

    // Define a CUTLASS GEMM type
    Gemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    Gemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                         {A, lda},    // Tensor-ref for source matrix A
                         {B, ldb},    // Tensor-ref for source matrix B
                         {C, ldc},    // Tensor-ref for source matrix C
                         {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                         {alpha, beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //

    if(gemm_operator.can_implement(args) != cutlass::Status::kSuccess || gemm_operator.get_workspace_size(args)!=0)
    {
      return cudaErrorUnknown;
    }

    cutlass::Status status = gemm_operator(args);

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}


template<typename T>
__host__ __device__ inline T fromFloat(float val)
{
  return  static_cast<T>(val);
}

template<>
__host__ __device__ inline half fromFloat(float val)
{
  return  __float2half(val);
}

template<typename T>
__host__ __device__ inline float toFloat(T val)
{
  return  static_cast<float>(val);
}

template<>
__host__ __device__ inline float toFloat(half val)
{
  return  __half2float(val);
}

/// Kernel to initialize a matrix with small integers.
template<typename T>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);
    if(offset %  7==0 || offset % 3 ==0) value = 0;
    matrix[offset] = fromFloat<T>(value);
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template<typename T>
cudaError_t InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<T> <<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
template<typename T>
cudaError_t AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}