#include "simple_test.h"
#include <stdio.h>
#include <iostream>

__global__ void helloCudaKernel()
{
	printf("Hello from your device :-)");
}

void helloCuda()
{
	cudaSetDevice(0);
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	
	printf("test ... \n");
	helloCudaKernel<<<1,1>>>();
	
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
	cudaDeviceSynchronize(); // synchronize the printf
}