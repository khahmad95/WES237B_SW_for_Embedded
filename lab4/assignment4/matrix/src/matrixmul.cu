#include <stdio.h>
#include <stdlib.h>

#include "matrixmul.h"
#include "timer.h"

#define BLOCK_SIZE 16

__global__ void block_mm_kernel(const float* A, const float* B, float* output, int M, int N) 
{
	// TODO: complete the block matrix kernel function
    const int threadRow = blockIdx.y*blockDim.y + threadIdx.y;
    const int threadCol = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0;
    for(int idx = 0; idx < M; idx++){
        sum += A[threadRow*M + idx] * B[idx * N + threadCol]; 
    }
    output[threadRow*N + threadCol] = sum; 
}


inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}


float run_mm_gpu(const float* A, const float* B, float* C, int M, int N)
{
	Timer gpu_timer;
	gpu_timer.start();

	//TODO: launch the kernel function
    const int grid_x = divup(N, BLOCK_SIZE);
    const int grid_y = divup(N, BLOCK_SIZE);
    
    dim3 blocksPerGrid(grid_x, grid_y, 1);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    
	block_mm_kernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, M, N);

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	gpu_timer.stop();
	float gpu_time = gpu_timer.getElapsed();
	gpu_timer.end();

	return gpu_time;
}


