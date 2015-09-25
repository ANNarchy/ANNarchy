global_operation_templates_openmp = {
    'max' : """
// Computes the maximum value of an array
double max_value(const double* array, int n)
{
    double max = array[0];
    for(int i=0; i<n; i++)
    {
        if(array[i] > max)
            max = array[i];
    }

    return max;
}
    """,
    'min' : """
// Computes the minimum value of an array
double min_value(const double* array, int n)
{
    double min = array[0];
    for(int i=0; i<n; i++)
    {
        if(array[i] < min)
            min = array[i];
    }

    return min;
}
    """,
    'mean' : """
// Computes the mean value of an array
double mean_value(const double* array, int n)
{
    double sum = 0.0;
    %(omp)s#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++)
    {
        sum += array[i];
    }
    return sum/(double)n;
}
    """,
    'norm1' : """
// Computes the L1-norm of an array
double norm1_value(const double* array, int n)
{
    double sum = 0.0;
    %(omp)s#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++)
    {
        sum += fabs(array[i]);
    }

    return sum;
}
    """,
    'norm2' : """
// Computes the L2-norm (Euclidian) of an array
double norm2_value(const double* array, int n)
{
    double sum = 0.0;
    %(omp)s#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++)
    {
        sum += pow(array[i], 2.0);
    }

    return sqrt(sum);
}
    """
}

global_operation_templates_extern = {
    'max': "extern double max_value(const double*, int);\n",
    'min': "extern double min_value(const double*, int);\n",
    'mean': "extern double mean_value(const double*, int);\n",
    'norm1': "extern double norm1_value(const double*, int);\n",
    'norm2': "extern double norm2_value(const double*, int);\n"
}

#
# determine correct kernel sizes
#
global_operation_templates_cuda = {
   'max' : {
        'header' : """__global__ void cuMaxValue(double* result, double *gpu_array, int N);""",
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
""",
        'call' : """
    if ( pop%(id)s._active ) {
        cuMaxValue <<< 1, 32, 64 * 8 >>> ( pop%(id)s._gpu_%(op)s_%(var)s, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._max_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    },

    'min' : {
        'header' : """__global__ void cuMinValue(double* result, double *gpu_array, int N);""",
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
""",
        'call' : """
    if ( pop%(id)s._active ) {
        cuMinValue <<< 1, 32, 64 * 8 >>> ( pop%(id)s._gpu_%(op)s_%(var)s, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._min_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    },

    'mean' : {
        'header' : """__global__ void cuMeanValue(double* result, double *gpu_array, int N);""",
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
""",
    'call' : """
    if ( pop%(id)s._active ) {
        cuMeanValue <<< 1, 32, 64 * 8 >>> ( pop%(id)s._gpu_%(op)s_%(var)s, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._%(op)s_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    },

    # Computes the L1-norm of an array
    'norm1' : {
        'header' : """__global__ void cuNorm1(double* result, double *gpu_array, int N);""",
        'body' : """// Computes the L1-norm value of an array
__global__ void cuNorm1(double* result, double *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(i < N)
    {
        localSum += fabs(gpu_array[i]);
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

    // write back result
    if (tid == 0)
    {
        *result = sdata[0];
    }
}
""",
    'call' : """
    if ( pop%(id)s._active ) {
        cuNorm1 <<< 1, 32, 64 * 8 >>> ( pop%(id)s._gpu_%(op)s_%(var)s, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._%(op)s_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    },

    # Computes the L2-norm (Euclidian) of an array
    'norm2' : {
        'header' : """__global__ void cuNorm2(double* result, double *gpu_array, int N);""",
        'body' : """// Computes the L2-norm value of an array
__global__ void cuNorm2(double* result, double *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(i < N)
    {
        localSum += pow(gpu_array[i], 2);
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

    // write back result
    if (tid == 0)
    {
        *result = sqrt(sdata[0]);
    }
}
""",
    'call' : """
    if ( pop%(id)s._active ) {
        cuNorm2 <<< 1, 32, 64 * 8 >>> ( pop%(id)s._gpu_%(op)s_%(var)s, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._%(op)s_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(double), cudaMemcpyDeviceToHost);
    }
"""
    }
}