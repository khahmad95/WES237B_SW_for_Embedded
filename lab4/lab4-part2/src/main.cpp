#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "img_proc.h"
#include "timer.h"

#define OPENCV 0
#define CPU 1
#define GPU 2

#define BLUR_SIZE 5

//#define UNIFIED_MEM 

using namespace std;
using namespace cv;

int usage()
{
	cout << "Usage: ./lab4 <mode> <WIDTH> <HEIGHT>" <<endl;
	cout << "mode: 0 OpenCV" << endl;
	cout << "      1 CPU" << endl;
	cout << "      2 GPU" << endl;
	return 0;
}

int use_mode(int mode)
{
	string descr;
	switch(mode)
	{
		case OPENCV:
			descr = "OpenCV Functions";
			break;
		case CPU:
			descr = "CPU Implementations";
			break;
		case GPU:
			descr = "GPU Implementations";
			break;
		default:
			descr = "None";
			return usage();
	}	
	
	cout << "Using " << descr.c_str() <<endl;
	return 1;
}

int main(int argc, const char *argv[]) 
{

	int mode = 0;

	if(argc >= 2)
	{
		mode = atoi(argv[1]);	
	}
	
	if(use_mode(mode) == 0)
		return 0;

	VideoCapture cap("input.raw");

	int WIDTH  = 768;
	int HEIGHT = 768;
	int CHANNELS = 3;

	// 1 argument on command line: WIDTH = HEIGHT = arg
	if(argc >= 3)
	{
		WIDTH = atoi(argv[2]);
		HEIGHT = WIDTH;
	}
	// 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
	if(argc >= 4)
	{
		HEIGHT = atoi(argv[3]);
	}

	// Profiling framerate
	LinuxTimer timer;
	LinuxTimer fps_counter;
	double time_elapsed = 0;

#ifndef UNIFIED_MEM
    //TODO: Allocate memory on the GPU device.
    unsigned char* rgb_device;
    unsigned char* gray_device;
    unsigned char* invert_device;
    unsigned char* blur_device;
    cudaMalloc((void **)&rgb_device,WIDTH*HEIGHT*sizeof(unsigned char)*CHANNELS);
    cudaMalloc((void **)&gray_device,WIDTH*HEIGHT*sizeof(unsigned char));
    cudaMalloc((void **)&invert_device,WIDTH*HEIGHT*sizeof(unsigned char));
    cudaMalloc((void **)&blur_device,WIDTH*HEIGHT*sizeof(unsigned char));
    //TODO: Declare the host image result matrices
    Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3);
    Mat grayed = Mat(HEIGHT, WIDTH, CV_8U);
    Mat inverted = Mat(HEIGHT, WIDTH, CV_8U);
    Mat blurred = Mat(HEIGHT, WIDTH, CV_8U);
    
#else
    //TODO: Allocate unified memory for the necessary matrices
    unsigned char* rgb_device;
    unsigned char* gray_device;
    unsigned char* invert_device;
    unsigned char* blur_device;
    cudaMallocManaged(&rgb_device,HEIGHT*WIDTH*sizeof(unsigned char)*CHANNELS);
    cudaMallocManaged(&gray_device,HEIGHT*WIDTH*sizeof(unsigned char));
    cudaMallocManaged(&invert_device,HEIGHT*WIDTH*sizeof(unsigned char));
    cudaMallocManaged(&blur_device,HEIGHT*WIDTH*sizeof(unsigned char));
//TODO: Declare the image matrices which point to the unified memory
    Mat rgb = Mat(HEIGHT, WIDTH, CV_8UC3, rgb_device);
    Mat grayed = Mat(HEIGHT, WIDTH, CV_8U, gray_device);
    Mat inverted = Mat(HEIGHT, WIDTH, CV_8U, invert_device);
    Mat blurred = Mat(HEIGHT, WIDTH, CV_8U, blur_device);
#endif

	//Matrix for OpenCV inversion
	Mat ones = Mat::ones(HEIGHT, WIDTH, CV_8U)*255;

	Mat frame;	
	char key=0;
	int count = 0;

	while (key != 'q')
	{
		cap >> frame;
		if(frame.empty())
		{
			waitKey();
			break;
		}

		resize(frame, rgb, Size(WIDTH, HEIGHT));

		imshow("Original", rgb);

		timer.start();
		switch(mode)
		{
			case OPENCV:
#ifdef OPENCV4
				cvtColor(rgb, grayed, COLOR_BGR2GRAY);
                bitwise_not(grayed, inverted);
                blur(grayed,blurred,Size(BLUR_SIZE,BLUR_SIZE));
#else
				cvtColor(rgb, grayed, CV_BGR2GRAY);
                inverted = bitwise_not(grayed);
                blurred = blur(grayed,(BLUR_SIZE,BLUR_SIZE));
#endif
				break;
			case CPU:
                // TODO: 1) Call the CPU functions
                img_rgb2gray_cpu(grayed.ptr<uchar>(),rgb.ptr<uchar>(),HEIGHT,WIDTH,CHANNELS);
                img_invert_cpu(inverted.ptr<uchar>(),grayed.ptr<uchar>(),HEIGHT,WIDTH);
                img_blur_cpu(blurred.ptr<uchar>(),grayed.ptr<uchar>(),HEIGHT,WIDTH,BLUR_SIZE);
				break;

			case GPU:
#ifndef UNIFIED_MEM
                /* TODO: */
                // 1) Copy data from host to device
                 cudaMemcpy(rgb_device,rgb.ptr<uchar>(),WIDTH*HEIGHT*sizeof(unsigned char)*CHANNELS,cudaMemcpyHostToDevice);
                // 2) Call GPU host function with device data
                 img_rgb2gray(gray_device, rgb_device, HEIGHT, WIDTH, CHANNELS);
                // 3) Copy data from device to host
                 cudaMemcpy(grayed.ptr<uchar>(),gray_device,WIDTH*HEIGHT*sizeof(unsigned char),cudaMemcpyDeviceToHost);
                // Invert: Call GPU host function with device data
                 img_invert(invert_device, gray_device, HEIGHT, WIDTH);
                // Invert: Copy data from device to host
                 cudaMemcpy(inverted.ptr<uchar>(),invert_device,WIDTH*HEIGHT*sizeof(unsigned char),cudaMemcpyDeviceToHost);
                // Blur: Call GPU host function with device data
                 img_blur(blur_device, gray_device, HEIGHT, WIDTH, BLUR_SIZE);
                // Blur: Copy data from device to host
                 cudaMemcpy(blurred.ptr<uchar>(),blur_device,WIDTH*HEIGHT*sizeof(unsigned char),cudaMemcpyDeviceToHost);
#else
                /* TODO: */
                // 1) Call GPU host function with unified memory allocated data
                img_rgb2gray(grayed.ptr<uchar>(), rgb.ptr<uchar>(), HEIGHT, WIDTH, CHANNELS);
                img_invert(inverted.ptr<uchar>(), grayed.ptr<uchar>(), HEIGHT, WIDTH);                
                img_blur(blurred.ptr<uchar>(), grayed.ptr<uchar>(), HEIGHT, WIDTH, BLUR_SIZE);                
#endif
				break;
		}
		timer.stop();

		size_t time_rgb2gray = timer.getElapsed();
		
		count++;
		time_elapsed += (timer.getElapsed())/10000000000.0;

		if (count % 10 == 0)
		{
			cout << "Execution Time (s) = " << time_elapsed << endl;
			time_elapsed = 0;
		}

		imshow("Gray", grayed);
        imshow("Inverted", inverted);
        imshow("Blurred", blurred);
        key = waitKey(1);
	}
}
