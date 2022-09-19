#include <stdio.h>
#include <stdlib.h>

#include "matrixmul.h"
#include "timer.h"

#define BLOCK_SIZE 16

__global__ void block_mm_kernel(const float* A, const float* B, float* output, int M, int N) 
{
	// TODO: complete the block matrix kernel function
    const int threadRow = blockIdx.x*blockDim.x + threadIdx.x;
    const int globaly_id = blockIdx.y*blockDim.y + threadIdx.y;
    int global_idx = globalx_idx * N + globaly_idx;
    output[global_idx] = A[globaly_idx * N + globalx_idx] * B[globalx_idx * N + globaly_idx]; 
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
    
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    
	block_mm_kernel<<<grid,block>>>(A, B, C, M, N);

	CudaCheckError();
	CudaSafeCall(cudaDeviceSynchronize());
	gpu_timer.stop();
	float gpu_time = gpu_timer.getElapsed();
	gpu_timer.end();

	return gpu_time;
}


