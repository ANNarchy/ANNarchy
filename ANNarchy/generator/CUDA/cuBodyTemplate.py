cu_body_template=\
"""
#include "cuANNarchy.h"

%(kernel_config)s

/****************************************
 * inline functions                     *
 ****************************************/
__device__ __forceinline__ double positive( double x ) { return (x>0) ? x : 0; }

/****************************************
 * updating neural variables            *
 ****************************************/
%(pop_kernel)s
 
/****************************************
 * weighted sum kernels                 *
 ****************************************/
%(psp_kernel)s

/****************************************
 * update synapses kernel               *
 ****************************************/
%(syn_kernel)s

/****************************************
 * call kernels                         *
 ****************************************/
%(call_kernel)s
"""

psp_kernel=\
"""
template<unsigned int blockSize>
__global__ void cuPop%(pre)s_Pop%(post)s_%(target)s_psp( int size, int* pre_rank, int *nb_synapses, int* offsets, double *r, double* w, double *sum_%(target)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int i = tid+offsets[blockIdx.x];

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(i < nb_synapses[blockIdx.x]+offsets[blockIdx.x])
    {
        localSum += %(psp)s

        i+= blockSize;
    }

    sdata[tid] = localSum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = localSum = localSum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = localSum = localSum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = localSum = localSum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile double* smem = sdata;

        if (blockSize >=  64) { smem[tid] = localSum = localSum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = localSum = localSum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = localSum = localSum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = localSum = localSum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = localSum = localSum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = localSum = localSum + smem[tid +  1]; }

    }

    // write result for this block to global mem
    if (tid == 0)
    {
        sum_%(target)s[blockIdx.x] = sdata[0];
    }

}

void Pop%(pre)s_Pop%(post)s_%(target)s_psp( int size, int* pre_rank, int* nb_synapses, int *offsets, double *r, double* w, double *sum_%(target)s ) {
    int sharedMemSize = pop%(pre)s_pop%(post)s_%(target)s * 64;

    cuPop%(pre)s_Pop%(post)s_%(target)s_psp<pop%(pre)s_pop%(post)s_%(target)s><<<size, pop%(pre)s_pop%(post)s_%(target)s, sharedMemSize >>>( size, pre_rank, nb_synapses, offsets, r, w, sum_%(target)s );
}
"""