#include <cstdio>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;

// size of the tile (for simplicity use 16×16×16)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void init(half *A, half *B)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    A[tid] = tid % 10; // only allow numbers form 0 - 9
    B[tid] = tid % 10;
}

// A: (M×K) ; B: (K×N) ; C: (M×N)
__global__ void tensorCoreGemmKernel(half *A, half *B, float *C, int M, int N, int K)
{
    // each warp computes one tile of output C
    int warpNum = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // warp number
    int warpRow = warpNum / 4;                                  // warp row
    int warpCol = warpNum % 4;                                  // warp column
    int mTile = warpRow * WMMA_M;
    int nTile = warpCol * WMMA_N;

    if (mTile >= M || nTile >= N)
        return;

    // Declare the fragments. wmma stands for warp matrix multiply add
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> aFrag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> bFrag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    // Initialize the output to zero
    fill_fragment(cFrag, 0.0f);

    // Compute C(warpRow,warpCol)
    for (int i = 0; i < 4; i++)
    {
        int aInd = warpRow * K * WMMA_M + i * WMMA_K;
        int bInd = warpCol * WMMA_N + i * N * WMMA_K;

        load_matrix_sync(aFrag, A + aInd, K); // load submatrix A  into registers
        load_matrix_sync(bFrag, B + bInd, N); // load submatrix B into registers
        mma_sync(cFrag, aFrag, bFrag, cFrag); // do the multiplication.
    }

    // write final result
    store_matrix_sync(C + mTile * N + nTile, cFrag, N, mem_row_major);
}

void cpu_matrix_mult(float *A, float *B, float *C, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

int main()
{
    /**
     * @brief: half - one bit for sign, 5 bit for exponent, 10 bits for fraction/mantissa.
     *         single - one bit for sign, 8 bits for exponent, 23 bits for fraction/mantissa.
     *         Tensor core instructions work on fixed size fragments, therefore dimension of
     *         matrix aligned to that fixed size give better performance. Align matrix dimension
     *         to multiple of 16 with padding if required for tensor core operation.
     **/
    int M = 64, N = 64, K = 64;
    half *devA;
    half *devB;
    float *devC, *hostC;
    half *hostTempA, *hostTempB;
    float *hostA, *hostB;
    float *serHostC; // serial computed matrix multiplication

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate device memory of matrices
    cudaMalloc(&devA, M * K * sizeof(half));
    cudaMalloc(&devB, K * N * sizeof(half));
    cudaMalloc(&devC, M * N * sizeof(float));

    hostTempA = (half *)malloc(M * K * sizeof(half));
    hostTempB = (half *)malloc(K * N * sizeof(half));
    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(K * N * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serHostC = (float *)malloc(M * N * sizeof(float));

    // intialize device matrix A and B
    cudaError_t err = cudaGetLastError();
    init<<<4, 1024>>>(devA, devB); // since 64 * 64 = 4 * 1024
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    cudaMemcpy(hostTempA, devA, sizeof(half) * M * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostTempB, devB, sizeof(half) * K * N, cudaMemcpyDeviceToHost);

    cudaEventRecord(start, 0);                                       // starttime recorded.
    tensorCoreGemmKernel<<<1, 16 * 32>>>(devA, devB, devC, M, N, K); // 16 warps to compute matrix C since there are 16 tiles in total
    cudaEventRecord(stop, 0);                                        // endtime recoorded
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f milli seconds\n", elapsedTime);
    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("%s\n", cudaGetErrorString(err));

    cudaMemcpy(hostC, devC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // convert half to float for comparison
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = __half2float(hostTempA[i]);
    }

    for (int i = 0; i < K * N; i++)
    {
        hostB[i] = __half2float(hostTempB[i]);
    }

    cpu_matrix_mult(hostA, hostB, serHostC, M, K, N);

    float total_diff = 0;

    for (int i = 0; i < M * N; i++)
    {
        total_diff += fabs(serHostC[i] - hostC[i]);
    }

    printf("Total difference in precision = %f\n", total_diff);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    free(hostTempA);
    free(hostTempB);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serHostC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
