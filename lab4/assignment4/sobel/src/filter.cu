#include "filter.h"
#include "timer.h"

#include <iostream>

using namespace std;

// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}



// =================== CPU Functions ===================
void sobel_filter_cpu(const uchar * input, uchar * output, const uint height, const uint width)
{
    float sobel_x[3][3] =
    { { -1, 0, 1 },
      { -2, 0, 2 },
      { -1, 0, 1 } };

    float sobel_y[3][3] =
    { { -1, -2, -1 },
      { 0,  0,  0 },
      { 1,  2,  1 } };
      
    int mag_x, mag_y;
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            int idx = i*width + j;
            mag_x = 0;
            mag_y = 0;
            for(int x_idx = -1;x_idx <= 1; x_idx++){
                for(int y_idx = -1;y_idx <= 1; y_idx++){
                    int currentx = i + x_idx;
                    int currenty = j + y_idx;
                    
                    if(currentx >= 0 && currentx <= width && currenty >= 0 && currenty <= height){
                         mag_x += input[currentx*width + currenty]*sobel_x[x_idx+1][y_idx+1];
                         mag_y += input[currentx*width + currenty]*sobel_y[x_idx+1][y_idx+1];
                    }
                }
            }
            output[idx] = sqrt((mag_x * mag_x) + (mag_y * mag_y));
        }
    }
}

// =================== GPU Kernel Functions ===================
__global__
void kernel_sobel_filter(const uchar * input, uchar * output, const uint height, const uint width)
{            
    float sobel_x[3][3] =
    { { -1, 0, 1 },
      { -2, 0, 2 },
      { -1, 0, 1 } };

    float sobel_y[3][3] =
    { { -1, -2, -1 },
      { 0,  0,  0 },
      { 1,  2,  1 } };
    
    const int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
    
    int mag_x, mag_y;
    int idx = y_idx*width + x_idx;
    mag_x = 0;
    mag_y = 0;
    for(int sx = -1;sx <= 1; sx++){
        for(int sy = -1;sy <= 1; sy++){
            int currentx = x_idx + sx;
            int currenty = y_idx + sy;

            if(currentx >= 0 && currentx <= width && currenty >= 0 && currenty <= height){
                 mag_x += input[currenty*width + currentx]*sobel_x[sx+1][sx+1];
                 mag_y += input[currenty*width + currentx]*sobel_y[sy+1][sy+1];
            }
        }
    }
    output[idx] = sqrt((float)(mag_x * mag_x) + (mag_y * mag_y));
}

// =================== GPU Host Functions ===================
void sobel_filter_gpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	//TODO: launch kernel function
        
    // define grid size
    const int grid_x = 64;
    const int grid_y = 64;
    
    //define grid and block configuration
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(divup(width, grid_x), divup(height, grid_y), 1);
    
    //run GPU kernel function using grid and block config
    kernel_sobel_filter<<<grid,block>>>(input, output, height, width);
    
    //sync parallel processing
    cudaDeviceSynchronize();
}
