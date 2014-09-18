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

psp_kernel=\
"""
__global__ void cuPop%(pre)s_Pop%(post)s_%(target)s_psp( int* pre_rank, int *nb_synapses, int* offsets, double *r, double* w, double *sum_%(target)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int i = tid+offsets[blockIdx.x];

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(i < nb_synapses[blockIdx.x]+offsets[blockIdx.x])
    {
        localSum += %(psp)s

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
        for ( auto it = proj%(id)s.flat_idx.begin(); it != proj%(id)s.flat_idx.end(); it++)
            proj%(id)s.overallSynapses += *it;

        // off_synapses
        proj%(id)s.flat_off = flattenOff<int>(proj%(id)s.pre_rank);
        cudaMalloc((void**)&proj%(id)s.gpu_off_synapses, proj%(id)s.flat_off.size() * sizeof(int));
        cudaMemcpy(proj%(id)s.gpu_off_synapses, proj%(id)s.flat_off.data(), proj%(id)s.flat_off.size() * sizeof(int), cudaMemcpyHostToDevice);

        // pre_rank
        auto flat_proj%(id)s_pre_rank = flattenArray<int>(proj%(id)s.pre_rank);
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