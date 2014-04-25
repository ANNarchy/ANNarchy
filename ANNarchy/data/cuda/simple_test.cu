#include "simple_test.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <omp.h>

#include "memory_management.h"

__global__ void helloCudaKernel()
{
	printf("Hello from your device :-) \n");
}

void helloCuda()
{
	cudaSetDevice(0);
	
	helloCudaKernel<<<1,1>>>();
	cudaDeviceSynchronize(); // synchronize the printf
}

template<class T, unsigned int blockSize>
__global__ void
weightReduce(
		T *pr,		// neurons preynaptic layer
		T *w,		// weights matrix per neuron
		int *idx,	// index matrix per neuron
		int c,		// number of connections
		T *result	// write back result
	  ) {

	unsigned int tid = threadIdx.x;
    unsigned int i = tid;

	extern T __shared__ sdata[];
	T mySum = 0.0;

	while(i < c) {
		mySum += pr[idx[i]] * w[i];

		i+= blockSize;
	}

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
		volatile T* smem = sdata;

        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }

	}

    // write result for this block to global mem
    if (tid == 0)
    {
        *result = sdata[0];
    }
}

WeightSumData* WeightSumData::weightSumData_ = NULL;

DATA_TYPE weightedSum(std::vector<int> rank, std::vector<DATA_TYPE> value, std::vector<DATA_TYPE> preRates)
{
	int N = preRates.size();
	int C = rank.size();

	for ( int i = 0; i < rank.size(); i++ )
		std::cout << rank[i] << ", ";
	std::cout << std::endl;
	for ( int i = 0; i < value.size(); i++ )
		std::cout << value[i] << ", ";
	std::cout << std::endl;
	for ( int i = 0; i < preRates.size(); i++ )
		std::cout << preRates[rank[i]] << ", ";
	std::cout << std::endl;

	double start1 = omp_get_wtime();
	cudaMemcpy( WeightSumData::instance(N, C)->getWeightPtr(), value.data(), sizeof(DATA_TYPE) * C, cudaMemcpyHostToDevice );
	cudaMemcpy( WeightSumData::instance(N, C)->getRatePtr(), preRates.data(), sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy( WeightSumData::instance(N, C)->getIndexPtr(), rank.data(), sizeof(int) * C, cudaMemcpyHostToDevice);
	std::cout << "Copying data ("<< N <<" pre-rates, "<< C << " synapses): "<< (omp_get_wtime() - start1)*1000.0 << " ms "<< std::endl;

	int numBlocks = (int)ceil(double(rank.size())/32.0);
	int smemSize = 64*sizeof(DATA_TYPE);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	std::cout << "Compute kernel ... "<< std::endl;
	cudaEventRecord(start, 0);

	weightReduce<DATA_TYPE,32><<<numBlocks, 32, smemSize>>>(WeightSumData::instance(N, C)->getRatePtr(),
															WeightSumData::instance(N, C)->getWeightPtr(),
															WeightSumData::instance(N, C)->getIndexPtr(),
															C,
															WeightSumData::instance(N, C)->getResultPtr()
															);

	cudaDeviceSynchronize();
	cudaError_t status = cudaGetLastError();
	if ( status != cudaSuccess )
		std::cout << cudaGetErrorString(status) << std::endl;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime = 0.0;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << "Time for kernel ("<< C <<" synapses): "<< elapsedTime << " ms "<< std::endl;

	DATA_TYPE sum = 0.0;
	cudaMemcpy(&sum, WeightSumData::instance(N, C)->getResultPtr(), sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

	return sum;
}
