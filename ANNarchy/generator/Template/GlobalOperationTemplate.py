"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

global_operation_templates_st = {
    'max' : """
// Computes the maximum value of an array
%(type)s max_value(const %(type)s* array, int n)
{
    %(type)s max = array[0];
    for(int i=1; i<n; i++)
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
    for(int i=1; i<n; i++)
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
    %(type)s sum = array[0];
    for(int i=1; i<n; i++)
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
    %(type)s sum = fabs(array[0]);
    for(int i=1; i<n; i++)
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
    %(type)s sum = array[0] * array[0];
    for(int i=1; i<n; i++)
    {
        sum += array[i] * array[i];
    }
    return sqrt(sum);
}
"""
}

global_operation_templates_st_extern = {
    'max': "extern %(type)s max_value(const %(type)s*, int);\n",
    'min': "extern %(type)s min_value(const %(type)s*, int);\n",
    'mean': "extern %(type)s mean_value(const %(type)s*, int);\n",
    'norm1': "extern %(type)s norm1_value(const %(type)s*, int);\n",
    'norm2': "extern %(type)s norm2_value(const %(type)s*, int);\n"
}

global_operation_templates_openmp = {
    'max' : """// Computes the maximum value of an array
%(type)s max_value(const %(type)s* array, const int beg, const int end)
{
    %(type)s max = array[beg];
    for(int i=beg+1; i<end; i++)
    {
        if(array[i] > max)
            max = array[i];
    }
    return max;
}
""",
    'min' : """
// Computes the minimum value of an array
%(type)s  min_value(const %(type)s* array, const int beg, const int end)
{
    %(type)s min = array[beg];
    for(int i=beg+1; i<end; i++)
    {
        if(array[i] < min)
            min = array[i];
    }
    return min;
}
""",
    'mean' : """
// Computes the mean value of an array
%(type)s mean_value(const %(type)s* array, const int beg, const int end, const int n)
{
    %(type)s sum = array[beg];
    for(int i=beg+1; i<end; i++)
    {
        sum += array[i];
    }
    return sum/static_cast<%(type)s>(n);
}

%(type)s mean_value(const %(type)s* array, const int beg, const int end) 
{
    return mean_value(array, beg, end, end);
}
""",
    'norm1' : """
// Computes the L1-norm of an array
%(type)s norm1_value(const %(type)s* array, const int beg, const int end)
{
    %(type)s sum = fabs(array[beg]);
    for(int i=beg+1; i<end; i++)
    {
        sum += fabs(array[i]);
    }
    return sum;
}
""",
    'norm2' : """
// Computes the L2-norm (Euclidian) of an array
%(type)s norm2_value(const %(type)s* array, const int beg, const int end)
{
    %(type)s sum = array[beg] * array[beg];
    for(int i=beg+1; i<end; i++)
    {
        sum += array[i] * array[i];
    }

    return sum;
}
"""
}

# Will be called if number of elements is too small to be computed in parallel
global_operation_templates_st_call = {
    'max': """#pragma omp task
{
    _max_%(var)s = max_value(%(var)s.data(), 0, %(var)s.size());
}
""",
    'min': """#pragma omp task
{
    _min_%(var)s = min_value(%(var)s.data(), 0, %(var)s.size());
}
""",
    'mean': """#pragma omp task
{
    _mean_%(var)s = mean_value(%(var)s.data(), 0, %(var)s.size());
}
""",
    'norm1': """#pragma omp task
{
    _norm1_%(var)s = norm1_value(%(var)s.data(), 0, %(var)s.size());
}
""",
    'norm2': """#pragma omp task
{
    _norm2_%(var)s = sqrt(norm2_value(%(var)s.data(), 0, %(var)s.size()));
}
"""
}

global_operation_templates_omp_call = {
    'max': """#pragma omp master
{
    _max_%(var)s = %(var)s[0];
}
auto local_max_%(var)s = max_value(%(var)s.data(), chunks_[tid], chunks_[tid+1]);
""",
    'min': """#pragma omp master
{
    _min_%(var)s = %(var)s[0];
}
auto local_min_%(var)s = min_value(%(var)s.data(), chunks_[tid], chunks_[tid+1]);
""",
    'mean': """#pragma omp master
{
    _mean_%(var)s = 0.0;
}
auto local_mean_%(var)s = mean_value(%(var)s.data(), chunks_[tid], chunks_[tid+1], %(var)s.size());
""",
    'norm1': """#pragma omp master
{
    _norm1_%(var)s = 0.0;
}
auto local_norm1_%(var)s = norm1_value(%(var)s.data(), chunks_[tid], chunks_[tid+1]);
""",
    'norm2': """#pragma omp master
{
    _norm2_%(var)s = 0.0;
}
auto local_norm2_%(var)s = norm2_value(%(var)s.data(), chunks_[tid], chunks_[tid+1]);
"""
}

global_operation_templates_omp_reduce = {
    'max': """#pragma omp for schedule(static, 1)
for (int t = 0; t < nt; t++)
{
    #pragma omp critical
    {
        if ( local_max_%(var)s > _max_%(var)s )
            _max_%(var)s = local_max_%(var)s;
    }
}
""",
    'min': """#pragma omp for schedule(static, 1)
for (int t = 0; t < nt; t++)
{
    #pragma omp critical
    {
        if ( local_min_%(var)s < _min_%(var)s )
            _min_%(var)s = local_min_%(var)s;
    }
}
""",
    'mean': """#pragma omp for reduction(+: _mean_%(var)s)
for (int t = 0; t < nt; t++)
{
    _mean_%(var)s += local_mean_%(var)s;
}
""",
    'norm1': """#pragma omp for reduction(+: _norm1_%(var)s)
for (int t = 0; t < nt; t++)
{
    _norm1_%(var)s += local_norm1_%(var)s;
}
""",
    'norm2': """#pragma omp for reduction(+: _norm2_%(var)s)
for (int t = 0; t < nt; t++)
{
    _norm2_%(var)s += local_norm2_%(var)s;
}
#pragma omp master
{
    _norm2_%(var)s = sqrt(_norm2_%(var)s);
}
"""
}

global_operation_templates_omp_extern = {
    'max': "%(type)s max_value(const %(type)s*, const int, const int);\n",
    'min': "extern %(type)s min_value(const %(type)s*, const int, const int);\n",
    'mean': "extern %(type)s mean_value(const %(type)s*, const int, const int, const int);\nextern %(type)s mean_value(const %(type)s*, const int, const int);\n",
    'norm1': "extern %(type)s norm1_value(const %(type)s*, const int, const int);\n",
    'norm2': "extern %(type)s norm2_value(const %(type)s*, const int, const int);\n"
}

#
# determine correct kernel sizes
#
global_operation_templates_cuda = {
   'max' : {
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
        'invoke': """void cuda_max_value(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N) {
    cuMaxValue <<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>> ( result, gpu_array, N );
}
""",
        'header': """void cuda_max_value(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N);""",

        'call' : """
    if ( pop%(id)s->_active ) {
        cuda_max_value(RunConfig(1, 32, 64 * sizeof(%(type)s), pop%(id)s->stream), pop%(id)s->_gpu_%(op)s_%(var)s, pop%(id)s->gpu_%(var)s, pop%(id)s->size );
        cudaMemcpy(&pop%(id)s->_max_%(var)s, pop%(id)s->_gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    #ifdef _DEBUG
        auto glob_%(op)s_pop%(id)s_err = cudaGetLastError();
        if (glob_%(op)s_pop%(id)s_err != cudaSuccess)
            std::cerr << "Global operation '%(op)s' (PopStruct%(id)s): " <<  << std::endl;
    #endif
    }
"""
    },

    'min' : {
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
        'invoke' : """
void cuda_min_value(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N) {
    cuMinValue <<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>> ( result, gpu_array, N );
}
""",
        'header' : """void cuda_min_value(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N);""",
        'call' : """
    if ( pop%(id)s->_active ) {
        cuda_min_value(RunConfig(1, 32, 64 * sizeof(%(type)s), pop%(id)s->stream), pop%(id)s->_gpu_%(op)s_%(var)s, pop%(id)s->gpu_%(var)s, pop%(id)s->size );
        cudaMemcpy(&pop%(id)s->_min_%(var)s, pop%(id)s->_gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    },

    'mean' : {
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
        'invoke' : """void cuda_mean_value(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N) {
    cuMeanValue <<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>> ( result, gpu_array, N );
}
""",
        'header' : """void cuda_mean_value(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N);""",
        'call' : """
    if ( pop%(id)s->_active ) {
        cuda_mean_value(RunConfig(1, 32, 64 * sizeof(%(type)s), pop%(id)s->stream), pop%(id)s->_gpu_%(op)s_%(var)s, pop%(id)s->gpu_%(var)s, pop%(id)s->size );
        cudaMemcpy(&pop%(id)s->_%(op)s_%(var)s, pop%(id)s->_gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    },

    # Computes the L1-norm of an array
    'norm1' : {
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
        'invoke': """
void cuda_norm1(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N) {
    cuNorm1<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(result, gpu_array, N);
}
""",
        'header' : """void cuda_norm1(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N);""",
        'call' : """
    if ( pop%(id)s->_active ) {
        cuda_norm1(RunConfig(1, 32, 64 * sizeof(%(type)s), pop%(id)s->stream), pop%(id)s->_gpu_%(op)s_%(var)s, pop%(id)s->gpu_%(var)s, pop%(id)s->size );
        cudaMemcpy(&pop%(id)s->_%(op)s_%(var)s, pop%(id)s->_gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    },

    # Computes the L2-norm (Euclidian) of an array
    'norm2' : {
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
        'invoke': """
void cuda_norm2(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N) {
    cuNorm2<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(result, gpu_array, N);
}
""",
        'header' : """void cuda_norm2(RunConfig cfg, %(type)s* result, %(type)s *gpu_array, int N);""",
        'call' : """
    if ( pop%(id)s->_active ) {
        cuda_norm2(RunConfig(1, 32, 64 * sizeof(%(type)s), pop%(id)s->stream), pop%(id)s->_gpu_%(op)s_%(var)s, pop%(id)s->gpu_%(var)s, pop%(id)s->size );
        cudaMemcpy(&pop%(id)s->_%(op)s_%(var)s, pop%(id)s->_gpu_%(op)s_%(var)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
    }
"""
    }
}
