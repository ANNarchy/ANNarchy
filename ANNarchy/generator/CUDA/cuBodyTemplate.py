cu_body_template=\
"""
#include "cuANNarchy.h"
#include <float.h>

%(kernel_config)s

/****************************************
 * init random states                   *
 ****************************************/
__global__ void rng_setup_kernel( int N, curandState* states, unsigned long seed )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < N )
    {
        curand_init( seed, tid, 0, &states[ tid ] );
    }
}

void init_curand_states( int N, curandState* states, unsigned long seed ) {
    int numThreads = 64;
    int numBlocks = ceil (double(N) / double(numThreads));

    rng_setup_kernel<<< numBlocks, numThreads >>>( N, states, seed);
}

/****************************************
 * inline functions                     *
 ****************************************/
__device__ __forceinline__ double positive( double x ) { return (x>0) ? x : 0; }
__device__ __forceinline__ double negative( double x ) { return x<0.0? x : 0.0; }
__device__ __forceinline__ double clip(double x, double a, double b) { return x<a? a : (x>b? b :x); }

/****************************************
 * custom functions                     *
 ****************************************/
%(custom_func)s

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
 * global operations kernel             *
 ****************************************/
%(glob_ops_kernel)s
"""

cu_header_template=\
"""#ifndef __CUDA_KERNEL__
#define __CUDA_KERNEL__
#include <curand_kernel.h>

// initialize cuda RNG states
void init_curand_states( int N, curandState* states, unsigned long seed );

// population step prototypes
%(neuron)s

// compute_psp prototypes
%(compute_sum)s

// projection step prototypes
%(synapse)s

// global operations
%(glob_ops)s

#endif
"""

pop_kernel=\
"""
// gpu device kernel for population %(id)s
__global__ void cuPop%(id)s_step(double dt%(tar)s%(var)s%(par)s)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    // Updating global variables of population %(id)s
%(global_eqs)s

    // Updating local variables of population %(id)s
    if ( i < %(pop_size)s )
    {
%(local_eqs)s
    }
}

// host calls device kernel for population %(id)s
void Pop%(id)s_step(cudaStream_t stream, double dt%(tar)s%(var)s%(par)s)
{
    int nb = ceil ( double(%(pop_size)s) / (double)pop%(id)s );

    cuPop%(id)s_step<<<nb, pop%(id)s, 0, stream>>>(dt%(tar2)s%(var2)s%(par2)s);
}
"""

pop_kernel_call =\
"""
    // Updating the local and global variables of population %(id)s
    if ( pop%(id)s._active ) {
        Pop%(id)s_step(/* default arguments */
              pop%(id)s.stream, dt
              /* population targets */
              %(tar)s
              /* kernel gpu arrays */
              %(var)s
              /* kernel constants */
              %(par)s );
    }
"""

syn_kernel=\
"""
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_step( /* default params */
                              int *post_rank, int *pre_rank, int* nb_synapses, int* offsets, double dt
                              /* additional params */
                              %(var)s%(par)s )
{
    int i = blockIdx.x;
    int j = offsets[i] + threadIdx.x;
    int C = offsets[i]+ nb_synapses[i];
    int rk_post = post_rank[i];

    // Updating global variables of projection %(id)s
    if ( threadIdx.x == 0)
    {
%(global_eqs)s
    }

    // Updating local variables of projection %(id)s
    while ( j < C )
    {
        int rk_pre = pre_rank[j];
        
%(local_eqs)s

        j += blockDim.x;
    }
}

// host calls device kernel for population %(id)s
void Proj%(id)s_step(cudaStream_t stream, int size, int* post_rank, int *pre_rank, int *offsets, int *nb_synapses, double dt%(var)s%(par)s)
{

    cuProj%(id)s_step<<<size, pop%(pre)s_pop%(post)s_%(target)s, 0, stream>>>(post_rank, pre_rank, nb_synapses, offsets, dt%(var2)s%(par2)s);
}
"""

syn_kernel_call =\
"""
    // Updating the variables of projection %(id)s
    if ( proj%(id)s._learning && pop%(id)s._active )
    {
        Proj%(id)s_step(/* kernel config */
                  proj%(id)s.stream, proj%(id)s.post_rank.size(), proj%(id)s.gpu_post_rank, proj%(id)s.gpu_pre_rank, proj%(id)s.gpu_off_synapses, proj%(id)s.gpu_nb_synapses, dt
                  /* kernel gpu arrays */
                  %(local)s
                  /* kernel constants */
                  %(glob)s);
    }
"""

psp_kernel=\
"""
__global__ void cuPop%(pre)s_Pop%(post)s_%(target)s_psp( int* rank_pre, int *nb_synapses, int* offsets, double *pre_r, double* w, double *sum_%(target)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int j = tid+offsets[blockIdx.x];

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(j < nb_synapses[blockIdx.x]+offsets[blockIdx.x])
    {
        localSum += %(psp)s

        j+= blockDim.x;
    }

    sdata[tid] = localSum;
    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localSum = localSum + sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localSum = localSum + sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localSum = localSum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
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
        sum_%(target)s[blockIdx.x] = sdata[0];
    }

}

void Pop%(pre)s_Pop%(post)s_%(target)s_psp( cudaStream_t stream, int size, int* pre_rank, int* nb_synapses, int *offsets, double *r, double* w, double *sum_%(target)s ) {
    int sharedMemSize = pop%(pre)s_pop%(post)s_%(target)s * 64;

    cuPop%(pre)s_Pop%(post)s_%(target)s_psp<<<size, pop%(pre)s_pop%(post)s_%(target)s, sharedMemSize, stream >>>( pre_rank, nb_synapses, offsets, r, w, sum_%(target)s );
}
"""

psp_kernel_call =\
"""
    // proj%(id)s: pop%(pre)s -> pop%(post)s
    if ( pop%(post)s._active ) {
        Pop%(pre)s_Pop%(post)s_%(target)s_psp( proj%(id)s.stream, pop%(post)s.size,
                       /* ranks and offsets */
                       proj%(id)s.gpu_pre_rank, proj%(id)s.gpu_nb_synapses, proj%(id)s.gpu_off_synapses,
                       /* computation data */
                       pop%(pre)s.gpu_r, proj%(id)s.gpu_w,
                       /* result */
                       pop%(post)s.gpu_sum_%(target)s );
    }
"""

proj_basic_data =\
"""
    // Initialize device memory for proj%(id)s

    // post_rank
    cudaMalloc((void**)&proj%(id)s.gpu_post_rank, proj%(id)s.post_rank.size() * sizeof(int));
    cudaMemcpy(proj%(id)s.gpu_post_rank, proj%(id)s.post_rank.data(), proj%(id)s.post_rank.size() * sizeof(int), cudaMemcpyHostToDevice);

    // nb_synapses
    proj%(id)s.flat_idx = flattenIdx<int>(proj%(id)s.pre_rank);
    cudaMalloc((void**)&proj%(id)s.gpu_nb_synapses, proj%(id)s.flat_idx.size() * sizeof(int));
    cudaMemcpy(proj%(id)s.gpu_nb_synapses, proj%(id)s.flat_idx.data(), proj%(id)s.flat_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    proj%(id)s.overallSynapses = 0;
    std::vector<int>::iterator it%(id)s;
    for ( it%(id)s = proj%(id)s.flat_idx.begin(); it%(id)s != proj%(id)s.flat_idx.end(); it%(id)s++)
        proj%(id)s.overallSynapses += *it%(id)s;

    // off_synapses
    proj%(id)s.flat_off = flattenOff<int>(proj%(id)s.pre_rank);
    cudaMalloc((void**)&proj%(id)s.gpu_off_synapses, proj%(id)s.flat_off.size() * sizeof(int));
    cudaMemcpy(proj%(id)s.gpu_off_synapses, proj%(id)s.flat_off.data(), proj%(id)s.flat_off.size() * sizeof(int), cudaMemcpyHostToDevice);

    // pre_rank
    std::vector<int> flat_proj%(id)s_pre_rank = flattenArray<int>(proj%(id)s.pre_rank);
    cudaMalloc((void**)&proj%(id)s.gpu_pre_rank, flat_proj%(id)s_pre_rank.size() * sizeof(int));
    cudaMemcpy(proj%(id)s.gpu_pre_rank, flat_proj%(id)s_pre_rank.data(), flat_proj%(id)s_pre_rank.size() * sizeof(int), cudaMemcpyHostToDevice);
    flat_proj%(id)s_pre_rank.clear();

"""

stream_setup=\
"""
cudaStream_t streams[%(nbStreams)s];

void stream_setup()
{
    for ( int i = 0; i < %(nbStreams)s; i ++ )
    {
        cudaStreamCreate( &streams[i] );
    }
}

void stream_assign()
{
%(pop_assign)s

%(proj_assign)s
}

void stream_destroy()
{
    for ( int i = 0; i < %(nbStreams)s; i ++ )
    {
        // all work finished
        cudaStreamSynchronize( streams[i] );

        // destroy
        cudaStreamDestroy( streams[i] );
    }
}
"""