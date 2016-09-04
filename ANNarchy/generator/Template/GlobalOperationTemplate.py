global_operation_templates_openmp = {
    'max' : """
// Computes the maximum value of an array
%(type)s max_value(const %(type)s* array, int n)
{
    %(type)s max = array[0];
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
%(type)s min_value(const %(type)s* array, int n)
{
    %(type)s min = array[0];
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
%(type)s mean_value(const %(type)s* array, int n)
{
    %(type)s sum = 0.0;
    %(omp)s#pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++)
    {
        sum += array[i];
    }
    return sum/(%(type)s)n;
}
    """,
    'norm1' : """
// Computes the L1-norm of an array
%(type)s norm1_value(const %(type)s* array, int n)
{
    %(type)s sum = 0.0;
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
%(type)s norm2_value(const %(type)s* array, int n)
{
    %(type)s sum = 0.0;
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
    'max': "extern %(type)s max_value(const %(type)s*, int);\n",
    'min': "extern %(type)s min_value(const %(type)s*, int);\n",
    'mean': "extern %(type)s mean_value(const %(type)s*, int);\n",
    'norm1': "extern %(type)s norm1_value(const %(type)s*, int);\n",
    'norm2': "extern %(type)s norm2_value(const %(type)s*, int);\n"
}

#
# determine correct kernel sizes
#
global_operation_templates_cuda = {
   'max' : {
        'header' : """__global__ void cuMaxValue(%(type)s* result, %(type)s *gpu_array, int N);""",
        'body' : """// Computes the maximum value of an array
__global__ void cuMaxValue(%(type)s* result, %(type)s *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern %(type)s __shared__ sdata[];
    %(type)s localMax = FLT_MIN;

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
        volatile %(type)s* smem = sdata;

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
        cudaMemcpy(&pop%(id)s._max_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    },

    'min' : {
        'header' : """__global__ void cuMinValue(%(type)s* result, %(type)s *gpu_array, int N);""",
        'body' : """// Computes the minimum value of an array
__global__ void cuMinValue(%(type)s* result, %(type)s *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern %(type)s __shared__ sdata[];
    %(type)s localMin = FLT_MAX;

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
        volatile %(type)s* smem = sdata;

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
        cudaMemcpy(&pop%(id)s._min_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    },

    'mean' : {
        'header' : """__global__ void cuMeanValue(%(type)s* result, %(type)s *gpu_array, int N);""",
        'body' : """// Computes the mean value of an array
__global__ void cuMeanValue(%(type)s* result, %(type)s *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern %(type)s __shared__ sdata[];
    %(type)s localSum = 0.0;

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
        volatile %(type)s* smem = sdata;

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
        *result = sdata[0] / (%(type)s)N;
    }
}
""",
    'call' : """
    if ( pop%(id)s._active ) {
        cuMeanValue <<< 1, 32, 64 * 8 >>> ( pop%(id)s._gpu_%(op)s_%(var)s, pop%(id)s.gpu_%(var)s, pop%(id)s.size );
        cudaMemcpy(&pop%(id)s._%(op)s_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    },

    # Computes the L1-norm of an array
    'norm1' : {
        'header' : """__global__ void cuNorm1(%(type)s* result, %(type)s *gpu_array, int N);""",
        'body' : """// Computes the L1-norm value of an array
__global__ void cuNorm1(%(type)s* result, %(type)s *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern %(type)s __shared__ sdata[];
    %(type)s localSum = 0.0;

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
        volatile %(type)s* smem = sdata;

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
        cudaMemcpy(&pop%(id)s._%(op)s_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    },

    # Computes the L2-norm (Euclidian) of an array
    'norm2' : {
        'header' : """__global__ void cuNorm2(%(type)s* result, %(type)s *gpu_array, int N);""",
        'body' : """// Computes the L2-norm value of an array
__global__ void cuNorm2(%(type)s* result, %(type)s *gpu_array, int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;

    extern %(type)s __shared__ sdata[];
    %(type)s localSum = 0.0;

    while(i < N)
    {
        localSum += gpu_array[i]*gpu_array[i];
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
        volatile %(type)s* smem = sdata;

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
        cudaMemcpy(&pop%(id)s._%(op)s_%(var)s, pop%(id)s._gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    }
}