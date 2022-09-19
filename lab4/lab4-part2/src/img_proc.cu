#include "img_proc.h"

// =================== Helper Functions ===================
inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

// =================== CPU Functions ===================

void img_rgb2gray_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int channels)
{
    //TODO: Convert a 3 channel RGB image to grayscale
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            int idx = i*width + j;
            int rgb_idx = idx*channels;
            int sum = 0;
            for(int chan_idx = 0; chan_idx < channels; chan_idx++){
                sum += in[rgb_idx + chan_idx];
            }
            
            out[idx] = sum / channels;
        }
    }
}

void img_invert_cpu(uchar* out, const uchar* in, const uint width, const uint height)
{
    //TODO: Invert a 8bit image
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            int idx = i*width + j;          
            out[idx] = 255 - in[idx];
        }
    }
}

void img_blur_cpu(uchar* out, const uchar* in, const uint width, const uint height, const int blur_size)
{
    //TODO: Average out blur_size pixels
    int sum;
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            int idx = i*width + j;
            sum = 0;
            for(int x_idx = -2;x_idx <= 2; x_idx++){
                for(int y_idx = -2;y_idx <= 2; y_idx++){
                    int currentx = i + x_idx;
                    int currenty = j + y_idx;
                    
                    if(currentx >= 0 && currentx <= width && currenty >= 0 && currenty <= height){
                        sum += in[currentx*width + currenty];
                    }
                }
            }
            out[idx] = sum/(blur_size*blur_size);
        }
    }    
}

// =================== GPU Kernel Functions ===================
/*
TODO: Write GPU kernel functions for the above functions
   */
__global__
void kernel_img_rgb2gray(uchar* out, const uchar* in, const uint height, const uint width, const int channels){
    //Defining indexes
    const int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
    
    int out_idx, in_idx;
    out_idx = y_idx * width + x_idx;
    in_idx = out_idx * channels;
    
    //block and thread indexes for r,g,b channels
    uchar r = in[in_idx];
    uchar g = in[in_idx + 1];
    uchar b = in[in_idx + 2];
    
    uchar gray = (r + g + b)/3;
    out[out_idx] = gray;
}

__global__
void kernel_img_invert(uchar* out, const uchar* in, const uint height, const uint width){
    //Defining indexes
    const int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
    
    int idx = y_idx * width + x_idx;

    out[idx] = 255-in[idx];

}

__global__
void kernel_img_blur(uchar* out, const uchar* in, const uint height, const uint width, const int blur_size){
    //Defining indexes
    const int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y*blockDim.y + threadIdx.y;
    
    int idx = y_idx * width + x_idx;
    int sum = 0;
    for(int fx = -2;fx <= 2; fx++){
        for(int fy = -2;fy <= 2; fy++){
            int currentx = x_idx + fx;
            int currenty = y_idx + fy;
            if(currentx >= 0 && currentx <= width && currenty >= 0 && currenty <= height){
                sum += in[currenty*width + currentx];
            }
        }
    }
    out[idx] = sum/(blur_size*blur_size);
}

// =================== GPU Host Functions ===================
/* 
TODO: Write GPU host functions that launch the kernel functions above
   */

//HOST FUNCTION FOR GRAYSCALE
void img_rgb2gray(uchar* out, const uchar* in, const uint height, const uint width, const int channels){
    // define grid size
    const int grid_x = 64;
    const int grid_y = 64;
    
    //define grid and block configuration
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(divup(width, grid_x), divup(height, grid_y), 1);
    
    //run GPU kernel function using grid and block config
    kernel_img_rgb2gray<<<grid,block>>>(out, in, height, width, channels);
    
    //sync parallel processing
    cudaDeviceSynchronize();    
}

//HOST FUNCTION FOR INVERT
void img_invert(uchar* out, const uchar* in, const uint height, const uint width){
    // define grid size
    const int grid_x = 64;
    const int grid_y = 64;
    
    //define grid and block configuration
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(divup(width, grid_x), divup(height, grid_y), 1);
    
    //run GPU kernel function using grid and block config
    kernel_img_invert<<<grid,block>>>(out, in, height, width);
    
    //sync parallel processing
    cudaDeviceSynchronize();
    
}

//HOST FUNCTION FOR BLUR
void img_blur(uchar* out, const uchar* in, const uint height, const uint width, int size){
    // define grid size
    const int grid_x = 64;
    const int grid_y = 64;
    
    //define grid and block configuration
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(divup(width, grid_x), divup(height, grid_y), 1);
    
    //run GPU kernel function using grid and block config
    kernel_img_blur<<<grid,block>>>(out, in, height, width, size);
    
    //sync parallel processing
    cudaDeviceSynchronize();
    
}