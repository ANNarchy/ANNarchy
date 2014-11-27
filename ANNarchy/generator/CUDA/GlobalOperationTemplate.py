global_operation_templates = {
   'max' : {
        'header' : """void max_value(double* result, double *gpu_array, int N);""",
        'body' : """// Computes the maximum value of an array
__global__ void cuMaxValue(double* result, double *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern double __shared__ sdata[];
    double localMax = FLT_MIN;

    while(i < N)
    {
        localMax = fmaxf(localMax, gpu_array[i]);
        i+= blockDim.x;
    }

    sdata[tid] = localMax;
    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localMax = fmaxf(localMax, sdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localMax = fmaxf(localMax, sdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localMax = fmaxf(localMax, sdata[tid +  64]); } __syncthreads(); }

    if (tid < 32)
    {
        volatile double* smem = sdata;

        if (blockDim.x >=  64) { smem[tid] = localMax = fmaxf(localMax, smem[tid + 32]); }
        if (blockDim.x >=  32) { smem[tid] = localMax = fmaxf(localMax, smem[tid + 16]); }
        if (blockDim.x >=  16) { smem[tid] = localMax = fmaxf(localMax, smem[tid +  8]); }
        if (blockDim.x >=   8) { smem[tid] = localMax = fmaxf(localMax, smem[tid +  4]); }
        if (blockDim.x >=   4) { smem[tid] = localMax = fmaxf(localMax, smem[tid +  2]); }
        if (blockDim.x >=   2) { smem[tid] = localMax = fmaxf(localMax, smem[tid +  1]); }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        *result = sdata[0];
    }
}

void max_value(double* result, double *gpu_array, int N)
{
    // TODO: determine correct kernel sizes
    int sharedMemSize = 64 * 8;
    cuMaxValue <<< 1, 32, sharedMemSize >>> ( result, gpu_array, N );
}
""",
        'call' : """
    if ( pop%(id)s._active ) {
        max_value( tmp, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._max_%(var)s, tmp, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    },

    'min' : {
        'header' : """void min_value(double* result, double *gpu_array, int N);""",
        'body' : """// Computes the minimum value of an array
__global__ void cuMinValue(double* result, double *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern double __shared__ sdata[];
    double localMin = FLT_MAX;

    while(i < N)
    {
        localMin = fminf(localMin, gpu_array[i]);
        i+= blockDim.x;
    }

    sdata[tid] = localMin;
    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localMin = fminf(localMin, sdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localMin = fminf(localMin, sdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localMin = fminf(localMin, sdata[tid +  64]); } __syncthreads(); }

    if (tid < 32)
    {
        volatile double* smem = sdata;

        if (blockDim.x >=  64) { smem[tid] = localMin = fminf(localMin, smem[tid + 32]); }
        if (blockDim.x >=  32) { smem[tid] = localMin = fminf(localMin, smem[tid + 16]); }
        if (blockDim.x >=  16) { smem[tid] = localMin = fminf(localMin, smem[tid +  8]); }
        if (blockDim.x >=   8) { smem[tid] = localMin = fminf(localMin, smem[tid +  4]); }
        if (blockDim.x >=   4) { smem[tid] = localMin = fminf(localMin, smem[tid +  2]); }
        if (blockDim.x >=   2) { smem[tid] = localMin = fminf(localMin, smem[tid +  1]); }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        *result = sdata[0];
    }
}

void min_value(double* result, double *gpu_array, int N)
{
    // TODO: determine correct kernel sizes
    int sharedMemSize = 64 * 8;
    cuMinValue <<< 1, 32, sharedMemSize >>> ( result, gpu_array, N );
}
""",
        'call' : """
    if ( pop%(id)s._active ) {
        min_value( tmp, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._min_%(var)s, tmp, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    },

    'mean' : {
        'header' : """void mean_value(double* result, double *gpu_array, int N);""",
        'body' : """// Computes the mean value of an array
__global__ void cuMeanValue(double* result, double *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(i < N)
    {
        localSum += gpu_array[i];
        i+= blockDim.x;
    }

    sdata[tid] = localSum;
    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localSum = localSum + sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localSum = localSum + sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localSum = localSum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        volatile double* smem = sdata;

        if (blockDim.x >=  64) { smem[tid] = localSum = localSum + smem[tid + 32]; }
        if (blockDim.x >=  32) { smem[tid] = localSum = localSum + smem[tid + 16]; }
        if (blockDim.x >=  16) { smem[tid] = localSum = localSum + smem[tid +  8]; }
        if (blockDim.x >=   8) { smem[tid] = localSum = localSum + smem[tid +  4]; }
        if (blockDim.x >=   4) { smem[tid] = localSum = localSum + smem[tid +  2]; }
        if (blockDim.x >=   2) { smem[tid] = localSum = localSum + smem[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        *result = sdata[0] / (double)N;
    }
}

void mean_value(double* result, double *gpu_array, int N)
{
    // TODO: determine correct kernel sizes
    int sharedMemSize = 64 * 8;
    cuMeanValue <<< 1, 32, sharedMemSize >>> ( result, gpu_array, N );
}
""",
        'call' : """
    if ( pop%(id)s._active ) {
        mean_value( tmp, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._mean_%(var)s, tmp, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    }
    
    #
    # NOT IMPLEMENTED YET:
    # 
    
    #===========================================================================
    # norm1_template = """
    # // Computes the L1-norm of an array
    # double norm1_value(std::vector<double> &array)
    # {
    #     double sum = 0.0;
    #     #pragma omp parallel reduction(+:sum)
    #     for(int i=0; i<array.size(); i++)
    #     {
    #         sum += fabs(array[i]);
    #     }
    # 
    #     return sum;
    # }
    # """
    # norm2_template = """
    # // Computes the L2-norm (Euclidian) of an array
    # double norm2_value(std::vector<double> &array)
    # {
    #     double sum = 0.0;
    #     #pragma omp parallel reduction(+:sum)
    #     for(int i=0; i<array.size(); i++)
    #     {
    #         sum += pow(array[i], 2.0);
    #     }
    # 
    #     return sqrt(sum);
    # }
    # """
    #===========================================================================
}