#include <cuda.h>
#include <stdio.h>

__global__ void dkernel(int *dptr)
{
    dptr[threadIdx.x] = 1;
}

int main()
{
    int *dptr;
    cudaMalloc((void **)dptr, sizeof(int) * 10);

    dkernel<<<1, 10>>>(dptr);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        exit(1);
    }

    return 0;
}