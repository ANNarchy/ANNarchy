#ifndef __MEMORY_MANAGEMENT_H__
#define __MEMORY_MANAGEMENT_H__

/**
 * 	\brief		manages gpu data
 */
class WeightSumData
{
public:
	static WeightSumData* instance(int N)
	{
		if( weightSumData_ == NULL ) // initialize if not already done
		{
			weightSumData_ = new WeightSumData(N);
		}

		return weightSumData_;
	}

	void resize(int N)
	{
		if ( gpuRates_ != NULL)
		{
			cudaFree(gpuRates_);
			gpuRates_ = NULL;
		}
		if ( gpuWeights_ != NULL)
		{
			cudaFree(gpuWeights_);
			gpuWeights_ = NULL;
		}
		if ( gpuIdx_ != NULL)
		{
			cudaFree(gpuIdx_);
			gpuIdx_ = NULL;
		}

		std::cout << "Resize from " << nbElements_ << " to " << N << std::endl;
		cudaDeviceSynchronize();
		cudaMalloc((void**)&gpuWeights_, sizeof(DATA_TYPE) * N);
		cudaMalloc((void**)&gpuRates_, sizeof(DATA_TYPE) * N);
		cudaMalloc((void**)&gpuIdx_, sizeof(int) * N);

		nbElements_ = N;
	}

	DATA_TYPE* getRatePtr() { return gpuRates_; }
	DATA_TYPE* getWeightPtr() { return gpuWeights_; }
	int* getIndexPtr() { return gpuIdx_; }
	DATA_TYPE* getResultPtr() { return gpuResult_; }

protected:
	WeightSumData(int N)
	{
		gpuRates_ = NULL;
		gpuWeights_ = NULL;
		gpuResult_ = NULL;
		gpuIdx_ = NULL;
		nbElements_ = NULL;

		cudaMalloc((void**)&gpuResult_, sizeof(DATA_TYPE));
		resize(N);
	}

	DATA_TYPE *gpuRates_;
	DATA_TYPE *gpuWeights_;
	DATA_TYPE *gpuResult_;
	int *gpuIdx_;
	int nbElements_;

	static WeightSumData* weightSumData_;
};

/**
 * 	\brief		manages gpu data
 * 	\details	NOT FULLY IMPLEMENTED
 */
class WeightSumDataThreaded
{
public:
	static WeightSumDataThreaded* instance(int threadId, int N)
	{
		while ( weightSumData_.size() <= threadId ) // resize if needed
		{
			weightSumData_.push_back(NULL);
		}

		if( weightSumData_[threadId] == NULL ) // initialize if not already done
		{
			weightSumData_[threadId] = new WeightSumDataThreaded(N);
		}

		return weightSumData_[threadId];
	}

	void resize(int N)
	{
		std::cout << "Resize from " << nbElements_ << " to " << N << std::endl;
		cudaDeviceSynchronize();
		cudaMalloc((void**)&gpuWeights_, sizeof(DATA_TYPE) * N);
		cudaMalloc((void**)&gpuRates_, sizeof(DATA_TYPE) * N);
		cudaMalloc((void**)&gpuIdx_, sizeof(int) * N);

		nbElements_ = N;
	}

	DATA_TYPE* getRatePtr() { return gpuRates_; }
	DATA_TYPE* getWeightPtr() { return gpuWeights_; }
	int* getIndexPtr() { return gpuIdx_; }
	DATA_TYPE* getResultPtr() { return gpuResult_; }

protected:
	WeightSumDataThreaded(int N)
	{
		gpuRates_ = NULL;
		gpuWeights_ = NULL;
		gpuResult_ = NULL;
		gpuIdx_ = NULL;
		nbElements_ = NULL;

		cudaMalloc((void**)&gpuResult_, sizeof(DATA_TYPE));
		resize(N);
	}

	DATA_TYPE *gpuRates_;
	DATA_TYPE *gpuWeights_;
	DATA_TYPE *gpuResult_;
	int *gpuIdx_;
	int nbElements_;

	static std::vector<WeightSumDataThreaded*> weightSumData_;
};

#endif

