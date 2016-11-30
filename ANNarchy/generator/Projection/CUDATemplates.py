#===============================================================================
#
#     CUDATemplates.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
projection_header = """#pragma once

#include "pop%(id_pre)s.hpp"
#include "pop%(id_post)s.hpp"
%(include_additional)s
%(include_profile)s

extern PopStruct%(id_pre)s pop%(id_pre)s;
extern PopStruct%(id_post)s pop%(id_post)s;
%(struct_additional)s

/////////////////////////////////////////////////////////////////////////////
// proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct%(id_proj)s{
    // Number of dendrites
    int size;

    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;

%(declare_connectivity_matrix)s
%(declare_inverse_connectivity_matrix)s
%(declare_delay)s
%(declare_event_driven)s
%(declare_rng)s
%(declare_parameters_variables)s
%(declare_cuda_stream)s
%(declare_additional)s
%(declare_profile)s

    // Method called to initialize the projection
    void init_projection() {
        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

%(init_connectivity_matrix)s

        // Inverse the connectivity matrix if spiking neurons
        inverse_connectivity_matrix();

%(init_event_driven)s
%(init_parameters_variables)s
%(init_rng)s
%(init_additional)s
%(init_profile)s
    }

    // Spiking networks: inverse the connectivity matrix
    void inverse_connectivity_matrix() {
%(init_inverse_connectivity_matrix)s
    }

    // Accessors for default attributes
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }

    // Additional access methods
%(access_connectivity_matrix)s
%(access_parameters_variables)s
%(access_additional)s

    // Memory transfers
    void host_to_device() {
%(host_to_device)s
    }
    
    void device_to_host() {
%(device_to_host)s
    }

%(cuda_flattening)s
};
"""




cuda_stream = """
    // stream
    cudaStream_t stream;
"""

#
# Set of functions for convertion of LIL in CSR
cuda_flattening = """
    /*
     * (De-)Flattening of LIL structures
     */
    void genRowPtr( ) {
        std::vector<std::vector<int> >::iterator pre_it = pre_rank.begin();
        std::vector<int>::iterator post_it = post_rank.begin();

        row_ptr = std::vector<int>(pop%(id_post)s.size, 0);

        int curr_off = 0;
        for(int i = 0; i < pop%(id_post)s.size; i++) {
            row_ptr[i] = curr_off;
            if ( i == *post_it ) {
                curr_off += pre_it->size();
                pre_it++;
                post_it++;
            }
        }
        row_ptr.push_back(curr_off);
        overallSynapses = curr_off;
    }

    template<typename T>
    std::vector<T> flattenArray(std::vector<std::vector<T> > in)
    {
        std::vector<T> flatVec = std::vector<T>();
        typename std::vector<std::vector<T> >::iterator it;

        for ( it = in.begin(); it != in.end(); it++)
        {
            flatVec.insert(flatVec.end(), it->begin(), it->end());
        }

        return flatVec;
    }

    template<typename T>
    std::vector<std::vector<T> > deFlattenArray( std::vector<T> in )
    {
        std::vector<std::vector<T> > deFlatVec = std::vector<std::vector<T> >();
        std::vector<int>::iterator it;

        int t=0;
        for ( int i = 0; i < pop%(id_post)s.size; i++)
        {
            if ( row_ptr[i] != row_ptr[i+1] ) {
                int num_syn = row_ptr[i+1]-row_ptr[i];
                std::vector<T> tmp = std::vector<T>(in.begin()+t, in.begin()+t+num_syn);
                t += num_syn;

                deFlatVec.push_back(tmp);
            }
        }

        return deFlatVec;
    }
"""

######################################
### Summation CUDA
######################################
# Comment to if (tid < 32) block:
#
# now that we are using warp-synchronous programming (below)
# we need to declare our shared memory volatile so that the compiler
# doesn't reorder stores to it and induce incorrect behavior.
cuda_psp_kernel = {
    'body': """
__global__ void cu_proj%(id_proj)s_psp( int post_size, %(conn_args)s%(add_args)s, double* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern double __shared__ sdata[];
    
    while( bid < post_size ) {
        unsigned int j = tid+row_ptr[bid];
    
        double localSum = 0.0;
        while(j < row_ptr[bid+1])
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
        if (blockDim.x >=  64) { if (tid <  32) { sdata[tid] = localSum = localSum + sdata[tid +  32]; } __syncthreads(); }
    
        if (tid < 16)
        {
            volatile double* smem = sdata;
    
            smem[tid] = localSum = localSum + smem[tid + 16];
            smem[tid] = localSum = localSum + smem[tid +  8];
            smem[tid] = localSum = localSum + smem[tid +  4];
            smem[tid] = localSum = localSum + smem[tid +  2];
            smem[tid] = localSum = localSum + smem[tid +  1];
        }
    
        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[bid] += sdata[0];
        }
        
        bid += gridDim.x;
    }
}
""",
    'one2one': """
// gpu device kernel for projection %(id)s
__global__ void cu_proj%(id)s_psp( double dt, bool plasticity, int *spiked, %(conn_arg)s %(kernel_args)s ) {
    int syn_idx = spiked[blockIdx.x]; // one2one: syn_idx = n_idx

    if(threadIdx.x == 0) {
        g_target[syn_idx] += w[syn_idx];
        if ( g_target[syn_idx] > max_trans[syn_idx] )
            g_target[syn_idx] = max_trans[syn_idx];
    }
}
""",
    'header': """__global__ void cu_proj%(id)s_psp( int post_size, %(conn_args)s%(add_args)s, double* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active ) {
        int sharedMemSize = __pop%(id_pre)s_pop%(id_post)s_%(target)s_tpb__ * sizeof(double);

        cu_proj%(id_proj)s_psp<<< __pop%(id_pre)s_pop%(id_post)s_%(target)s_nb__, __pop%(id_pre)s_pop%(id_post)s_%(target)s_tpb__, sharedMemSize>>>(
                       pop%(id_post)s.size,
                       /* ranks and offsets */
                       %(conn_args)s
                       /* computation data */
                       %(add_args)s
                       /* result */
                       %(target_arg)s );

    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "cu_proj%(id_proj)s_psp: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }
"""

}

cuda_spike_psp_kernel = {
    'body': """// gpu device kernel for projection %(id)s
__global__ void cu_proj%(id)s_psp( double dt, bool plasticity, int *spiked, %(conn_arg)s %(kernel_args)s ) {

%(prefix)s

    // over all afferent connections
    while( %(row_desc)s ) {
%(event_driven)s
%(psp)s
%(pre_event)s

        syn_idx += blockDim.x;
    }
}
""",
    'header': """__global__ void cu_proj%(id)s_psp( double dt, bool plasticity, int *spiked, %(conn_header)s %(kernel_args)s );
""",
    'call': """
    if ( pop%(id_pre)s._active) {
        int num_events = pop%(id_pre)s.num_events;
        int tpb = __pop%(id_pre)s_pop%(id_post)s_%(target)s__;

    #ifdef _DEBUG
        std::cout << t << ": " << num_events << " event(s)." << std::endl;
    #endif
        if ( num_events > 0 ) {
            cu_proj%(id_proj)s_psp<<< num_events, tpb, 0, streams[%(stream_id)s] >>>( dt, proj%(id_proj)s._plasticity, pop%(id_pre)s.gpu_spiked, %(conn_args)s %(kernel_args)s );

        #ifdef _DEBUG
            cudaDeviceSynchronize();
            cudaError_t err_psp_proj%(id_proj)s = cudaGetLastError();
            if( err_psp_proj%(id_proj)s != cudaSuccess) {
                std::cout << "proj%(id_proj)s_psp (" << t << "): " << std::endl;
                std::cout << "   " << cudaGetErrorString(err_psp_proj%(id_proj)s) << std::endl;
                std::cout << "   kernel_config: " << num_events << ", " << tpb << std::endl;
            }
        #endif
        }
    }
"""
}

######################################
### Update synaptic variables CUDA
######################################
cuda_synapse_kernel = {
    # Update of global synaptic equations, consist of body (annarchyDevice.cu),
    # header and call semantic (take place in ANNarchyHost.cu)
    'global': {
        'body': """
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_global_step( /* default params */
                              int post_size, int *pre_rank, int *row_ptr, double dt
                              /* additional params */
                              %(kernel_args)s,
                              /* plasticity enabled */
                              bool plasticity )
{
    int rk_post = threadIdx.x + blockIdx.x*blockDim.x;

    while ( rk_post < post_size ) {
%(global_eqs)s

        rk_post += gridDim.x * blockDim.x;
    }
}
""",
    'header': """__global__ void cuProj%(id)s_global_step( int post_size, int *pre_rank, int *row_ptr, double dt %(kernel_args)s, bool plasticity);
""",
    'call': """
        // global update
        int nb_blocks = ceil( double(proj%(id_proj)s.post_rank.size()) / double(__pop%(pre)s_pop%(post)s_%(target)s_tpb__));
        cuProj%(id_proj)s_global_step<<< nb_blocks, __pop%(pre)s_pop%(post)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream>>>(
            proj%(id_proj)s.post_rank.size(),
            /* default args*/
            proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t global_step = cudaGetLastError();
        if ( global_step != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString( global_step ) << std::endl;
        }
    #endif
"""
    },

    # Update of local synaptic equations, consist of body (annarchyDevice.cu),
    # header and call semantic (take place in ANNarchyHost.cu)
    'local': {
        'body': """
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_local_step( /* default params */
                              int *pre_rank, int *row_ptr, double dt
                              /* additional params */
                              %(kernel_args)s,
                              /* plasticity enabled */
                              bool plasticity )
{
    int rk_post = blockIdx.x;
    int j = row_ptr[rk_post] + threadIdx.x;
    int C = row_ptr[rk_post+1];

    // Updating local variables of projection %(id)s
    while ( j < C )
    {
        int rk_pre = pre_rank[j];

%(local_eqs)s

        j += blockDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id)s_local_step( int *pre_rank, int *row_ptr, double dt %(kernel_args)s, bool plasticity);
""",
        'call': """
        // local update
        cuProj%(id_proj)s_local_step<<< pop%(id_post)s.size, __pop%(id_pre)s_pop%(id_post)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream>>>(
            /* default args*/
            proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t local_step = cudaGetLastError();
        if ( local_step != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString( local_step) << std::endl;
        }
    #endif
""",
    },

    # call semantic for global and local kernel
    'call': """
    // proj%(id_proj)s: pop%(pre)s -> pop%(post)s
    if ( proj%(id_proj)s._transmission && proj%(id_proj)s._update && proj%(id_proj)s._plasticity && ( (t - proj%(id_proj)s._update_offset)%%proj%(id_proj)s._update_period == 0L)) {
        double _dt = dt * proj%(id_proj)s._update_period;
%(global_call)s
%(local_call)s
    }
"""
}

######################################
### post-event update CUDA
######################################
cuda_spike_postevent_kernel = {
    'body': """// Projection %(id_proj)s: post-synaptic events
__global__ void cuProj%(id_proj)s_postevent( double dt, bool plasticity, int* spiked, %(conn_args)s double* w %(add_args)s ) {
    int i = spiked[blockIdx.x];                // post-synaptic
    int j = row_ptr[i]+threadIdx.x;    // pre-synaptic

    while ( j < row_ptr[i+1] ) {
%(event_driven)s
%(post_code)s

        j+= blockDim.x;
    }
}
""",
    'header': """__global__ void cuProj%(id_proj)s_postevent( double dt, bool plasticity, int* spiked, %(conn_args)s double* w %(add_args)s );
""",
    'call': """
    if ( proj%(id_proj)s._transmission && pop%(id_post)s._active) {
        if (pop%(id_post)s.num_events > 0 ) {
            cuProj%(id_proj)s_postevent<<< pop%(id_post)s.num_events, __pop%(id_pre)s_pop%(id_post)s_%(target)s__ >>>(
                dt, proj%(id_proj)s._plasticity, pop%(id_post)s.gpu_spiked
                /* connectivity */
                %(conn_args)s
                /* weights */
                , proj%(id_proj)s.gpu_w
                /* other variables */
                %(add_args)s
            );
        #ifdef _DEBUG
            cudaDeviceSynchronize();
            cudaError_t proj%(id_proj)s_postevent = cudaGetLastError();
            if (proj%(id_proj)s_postevent != cudaSuccess) {
                std::cout << "proj%(id_proj)s_postevent: " << cudaGetErrorString(proj%(id_proj)s_postevent) << std::endl;
            }
        #endif
        }
    }
"""
}

cuda_templates = {
    # base stuff
    'projection_header': projection_header,
    'cuda_stream': cuda_stream,
    'flattening': cuda_flattening,

    # operations
    'computesum_rate': cuda_psp_kernel,
    'computesum_spiking': cuda_spike_psp_kernel,
    'post_event': cuda_spike_postevent_kernel,
    'synapse_update': cuda_synapse_kernel
}
