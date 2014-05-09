#ifndef __MEMORY_MANAGEMENT_H__
#define __MEMORY_MANAGEMENT_H__

/**
 * 	\brief		manages gpu data
 */
class WeightSumData
{
public:
	static WeightSumData* instance(int N, int C)
	{
		if( weightSumData_ == NULL ) // initialize if not already done
		{
			weightSumData_ = new WeightSumData(N, C);
		}

		return weightSumData_;
	}

	void resize(int N, int C)
	{
		if ( nbRates_ <= N)
		{
			std::cout << "Resize from " << nbRates_ << " to " << N << std::endl;
			if ( gpuRates_ != NULL)
			{
				cudaFree(gpuRates_);
				gpuRates_ = NULL;
			}

			cudaMalloc((void**)&gpuRates_, sizeof(DATA_TYPE) * N);
			nbRates_ = N;
		}

		if ( nbWeights_ <= C )
		{
			std::cout << "Resize from " << nbWeights_ << " to " << C << std::endl;
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

			cudaMalloc((void**)&gpuWeights_, sizeof(DATA_TYPE) * C);
			cudaMalloc((void**)&gpuIdx_, sizeof(int) * C);
			nbWeights_ = C;
		}
	}

	DATA_TYPE* getRatePtr() { return gpuRates_; }
	DATA_TYPE* getWeightPtr() { return gpuWeights_; }
	int* getIndexPtr() { return gpuIdx_; }
	DATA_TYPE* getResultPtr() { return gpuResult_; }

protected:
	WeightSumData(int N, int C)
	{
		gpuRates_ = NULL;
		gpuWeights_ = NULL;
		gpuResult_ = NULL;
		gpuIdx_ = NULL;

		nbRates_ = 0;
		nbWeights_ = 0;

		cudaMalloc((void**)&gpuResult_, sizeof(DATA_TYPE));
		resize(N, C);
	}

	DATA_TYPE *gpuRates_;
	DATA_TYPE *gpuWeights_;
	DATA_TYPE *gpuResult_;
	int *gpuIdx_;

	int nbRates_;
	int nbWeights_;

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
		nbElements_ = 0;

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

