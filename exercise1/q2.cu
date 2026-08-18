#include <iostream>
#include <cuda.h>

__device__ int blockCounter = 0;

/**
 * @brief A barrier for all threads of a CUDA kernel
 *        if the number of blocks <= Number of SMs
 *        else there would be a deadlock
 * 
 * @note A warp can be pre-empted but a block cannot be pre-empted in CUDA
 *       If multiple blocks can fit in a SM then it is possible for grid-wise sync
 *       using __syncthreads() and atomic operations. It all depends on the amount 
 *       of resources available on the SM
 * 
 * @param totalBlocks 
 * 
 */
__global__ void kernel(int totalBlocks)
{
    // Except the thread 0 all threads in block do not execute this block
    if (threadIdx.x == 0)
    {
        atomicAdd(&blockCounter, 1);
    }

    __syncthreads();

    // Wait till blockCounter reaches the totalBlocks
    while (atomicAdd(&blockCounter,0) != totalBlocks){}

    __syncthreads();
}

int main()
{
    int totalBlocks = 32;
    int threadPerBlock = 1024;

    kernel<<<totalBlocks, threadPerBlock>>>(totalBlocks);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    int hBlockCounter;

    cudaMemcpyFromSymbol(&hBlockCounter, &blockCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);

    std::cout << "Block Counter value: " << hBlockCounter << std::endl;

    return 0;
}
