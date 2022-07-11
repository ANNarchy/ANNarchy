#===============================================================================
#
#     CSR.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
#     Julien Vitay <julien.vitay@gmail.com>
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

# Code which should be added prior to kernels
# (directly imported by CodeGenerator if needed)
additional_global_functions = """
// warp reduce as in Dinkelbach et al. 2012 / Harris (CUDA Webinar 2)
template<typename DATA_TYPE>
__device__ void half_warp_reduce_sum(volatile DATA_TYPE* data, unsigned int tid) {
    data[tid] += data[tid + 16];
    data[tid] += data[tid +  8];
    data[tid] += data[tid +  4];
    data[tid] += data[tid +  2];
    data[tid] += data[tid +  1];
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Alternative implementation for Keplar and upwards
    // https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
    #define FULL_WARP_MASK 0xFFFFFFFF
    template<class ValueType, unsigned int WARP_SIZE>
    __device__ ValueType warp_reduce (ValueType val)
    {
        for(int offset = WARP_SIZE/2; offset > 0; offset /= 2)
            val += __shfl_down_sync(FULL_WARP_MASK, val, offset, 32);

        return val;
    }
#endif
"""

init_launch_config = """
        // Generate the kernel launch configuration
        _threads_per_block = 64;
        _nb_blocks = std::min<unsigned int>(nb_dendrites(), 65535);
    
    #ifdef _DEBUG
        std::cout << "Kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif
"""

attribute_decl = {
    'local': """
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
    %(type)s* gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
""",
    'semiglobal': """
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
    %(type)s* gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
""",
    'global': {
        'parameter': """
    // Global %(attr_type)s %(name)s
    %(type)s %(name)s;
""",
        'variable': """
    // Global %(attr_type)s %(name)s
    %(type)s %(name)s;
    %(type)s* gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
"""
    }
}

attribute_cpp_init = {
    'local': """
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(%(init)s);
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        %(name)s_host_to_device = true;
        %(name)s_device_to_host = t;
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(%(init)s);
        gpu_%(name)s = init_vector_variable_gpu<%(type)s>(%(name)s);
        %(name)s_host_to_device = true;
        %(name)s_device_to_host = t;
""",
    'global': {
        'parameter': """
        // Global %(attr_type)s %(name)s
        %(name)s = 0.0;
    """,
        'variable': """
        // Global %(attr_type)s %(name)s
        %(name)s = static_cast<%(type)s>(%(init)s);
        cudaMalloc((void**)&gpu_%(name)s, sizeof(%(type)s));
        %(name)s_host_to_device = true;
        %(name)s_device_to_host = t;
"""
    }
}

attribute_cpp_size = {
    'local': """
        // Local %(attr_type)s %(name)s
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(%(ctype)s*);
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(%(ctype)s*);
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'global': """
        // Global
        size_in_bytes += sizeof(bool);
        size_in_bytes += sizeof(%(ctype)s*);
        size_in_bytes += sizeof(%(ctype)s);
"""
}

attribute_cpp_delete = {
    'local': """
        // %(name)s - host
        %(name)s.clear();
        %(name)s.shrink_to_fit();

        // %(name)s - device
        cudaFree(gpu_%(name)s);
""",
    'semiglobal': """
        // %(name)s - host
        %(name)s.clear();
        %(name)s.shrink_to_fit();

        // %(name)s - device
        cudaFree(gpu_%(name)s);
""",
    'global': ""
}

attribute_host_to_device = {
    'local': """
        // %(name)s: local
        if ( %(name)s_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), num_non_zeros_ * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'semiglobal': """
        // %(name)s: semiglobal
        if ( %(name)s_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), post_ranks_.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'global': """
        // %(name)s: global
        if ( %(name)s_host_to_device )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, &%(name)s, sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
"""
}

attribute_device_to_host = {
    'local': """
        // %(name)s: local
        if ( %(name)s_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, num_non_zeros_ * sizeof( %(type)s ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
            %(name)s_device_to_host = t;
        }
""",
    'semiglobal': """
            // %(name)s: semiglobal
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, post_ranks_.size() * sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
""",
    'global': """
            // %(name)s: global
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( &%(name)s, gpu_%(name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
"""
}

delay = {
    'uniform': {
        'declare': """
    // Uniform delay
    int delay ;""",
        'pyx_struct': """
        # Non-uniform delay
        int delay""",
        'init': "delay = delays[0][0];",
        'pyx_wrapper_init': """
        proj%(id_proj)s.delay = syn.uniform_delay""",
        'pyx_wrapper_accessor': """
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay
    def set_delay(self, value):
        print("set delay", value)
        proj%(id_proj)s.delay = value
"""
    },
    'nonuniform_rate_coded': {
        'declare': """
    // Non-uniform delay
    std::vector< std::vector< int > > delay ;""",
        'pyx_struct': """
        # Non-uniform delay
        vector[vector[int]] delay""",
    
        'init': "",

        'pyx_wrapper_init': """
        proj%(id_proj)s.delay = syn.delay""",
    
        'pyx_wrapper_accessor': """
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay[idx]
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""
    },
    'nonuniform_spiking': None
}

event_driven = {
    'declare': """
    std::vector< long > _last_event;
    long* _gpu_last_event;
""",
    'cpp_init': """
    _last_event = init_matrix_variable<long>(-10000);
    _gpu_last_event = init_matrix_variable_gpu<long>(_last_event);    
""",
    'pyx_struct': """
        vector[long] _last_event
""",
    'pyx_wrapper_init':
"""
        proj%(id_proj)s._last_event = vector[long]( syn._matrix.num_elements(), -10000)
"""
}

#
# Implement the continuous transmission for rate-coded synapses.
#
rate_psp_kernel = {
    # Comment to if (tid < 32) block:
    #
    # now that we are using warp-synchronous programming (below)
    # we need to declare our shared memory volatile so that the compiler
    # doesn't reorder stores to it and induce incorrect behavior.
    'body': {
        'sum':"""
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        
        %(idx_type)s rk_post = rank_post[bid];

        // thread-local sum store result in shared memory
        sdata[tid] = %(thread_init)s;
        for (%(size_type)s j = row_ptr[rk_post] + tid; j < row_ptr[rk_post+1]; j+= blockDim.x) {
            sdata[tid] += %(psp)s
        }

        __syncthreads();

        // reduction in shared mem across warps
        if (blockDim.x == 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { sdata[tid] += sdata[tid +  32]; } __syncthreads(); }

        // reduction in shared mem within one warp
        if (tid < 16) { half_warp_reduce_sum<%(float_prec)s>(sdata, tid); }

        // write result for this block to global mem
        if (tid == 0) { %(target_arg)s[rk_post] += sdata[0]; }

        bid += gridDim.x;
    }
}
""",
    'min':"""
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        
        %(idx_type)s rk_post = rank_post[bid];
        %(size_type)s j = row_ptr[rk_post] + threadIdx.x;
        %(size_type)s C = row_ptr[rk_post+1];

        // Init all threads with max. value
        %(float_prec)s localMin = %(thread_init)s;

        // Iterate with chunks over the array
        while(j < C)
        {
            auto tmp = %(psp)s;
            if (tmp < localMin)
                localMin = tmp;

            j+= blockDim.x;
        }

        sdata[tid] = localMin;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 512) { if (tid < 256) { if ( sdata[tid] > sdata[tid + 256] ) sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { if ( sdata[tid] > sdata[tid + 128] ) sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { if ( sdata[tid] > sdata[tid + 64] ) sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { if ( sdata[tid] > sdata[tid + 32] ) sdata[tid] = sdata[tid + 32]; } __syncthreads(); }

        if (tid < 16)
        {
            volatile %(float_prec)s* smem = sdata;

            // if other value is smaller, copy
            if ( smem[tid] > smem[tid + 16] ) smem[tid] = smem[tid + 16];
            if ( smem[tid] > smem[tid +  8] ) smem[tid] = smem[tid + 8];
            if ( smem[tid] > smem[tid +  4] ) smem[tid] = smem[tid + 4];
            if ( smem[tid] > smem[tid +  2] ) smem[tid] = smem[tid + 2];
            if ( smem[tid] > smem[tid +  1] ) smem[tid] = smem[tid + 1];
        }

        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[rk_post] += sdata[0];
        }

        bid += gridDim.x;
    }
}
""",
    'max':"""
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        %(idx_type)s rk_post = rank_post[bid];
        %(size_type)s j = row_ptr[rk_post] + threadIdx.x;
        %(size_type)s C = row_ptr[rk_post+1];

        // Init all threads with min. value
        %(float_prec)s localMax = %(thread_init)s;

        // Iterate with chunks over the array
        while(j < C)
        {
            %(float_prec)s tmp = %(psp)s;
            if (tmp > localMax)
                localMax = tmp;

            j+= blockDim.x;
        }

        sdata[tid] = localMax;
        __syncthreads();

        // do reduction in shared mem
        if (blockDim.x >= 512) { if (tid < 256) { if ( sdata[tid] < sdata[tid + 256] ) sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { if ( sdata[tid] < sdata[tid + 128] ) sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { if ( sdata[tid] < sdata[tid + 64] ) sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { if ( sdata[tid] < sdata[tid + 32] ) sdata[tid] = sdata[tid + 32]; } __syncthreads(); }

        if (tid < 16)
        {
            volatile %(float_prec)s* smem = sdata;

            // if other value is larger, copy
            if ( smem[tid] < smem[tid + 16] ) smem[tid] = smem[tid + 16];
            if ( smem[tid] < smem[tid +  8] ) smem[tid] = smem[tid + 8];
            if ( smem[tid] < smem[tid +  4] ) smem[tid] = smem[tid + 4];
            if ( smem[tid] < smem[tid +  2] ) smem[tid] = smem[tid + 2];
            if ( smem[tid] < smem[tid +  1] ) smem[tid] = smem[tid + 1];
        }

        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[rk_post] += sdata[0];
        }

        bid += gridDim.x;
    }
}
""",
    # Technically a sum operation, but the result is normalized with the number of connection entries
    'mean': """
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];

    while( bid < post_size ) {
        
        %(idx_type)s rk_post = rank_post[bid];
        %(size_type)s j = row_ptr[rk_post] + threadIdx.x;
        %(size_type)s C = row_ptr[rk_post+1];

        // thread-local sum store result in shared memory
        sdata[tid] = %(thread_init)s;
        while(j < C) {
            sdata[tid] += %(psp)s
            j += blockDim.x;
        }
        __syncthreads();

        // reduction in shared mem across warps
        if (blockDim.x == 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
        if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
        if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
        if (blockDim.x >=  64) { if (tid <  32) { sdata[tid] += sdata[tid +  32]; } __syncthreads(); }

        // do reduction in shared mem within one warp
        if (tid < 16) { half_warp_reduce_sum<%(float_prec)s>(sdata, tid); }

        // write result for this block to global mem
        if (tid == 0) { %(target_arg)s[rk_post] += sdata[0] / (%(float_prec)s(row_ptr[bid+1]-row_ptr[bid])); }

        bid += gridDim.x;
    }
}
"""
    },
    'header': """__global__ void cu_proj%(id)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        int sharedMemSize = proj%(id_proj)s._threads_per_block * sizeof(%(float_prec)s);
        cu_proj%(id_proj)s_psp<<< proj%(id_proj)s._nb_blocks, proj%(id_proj)s._threads_per_block, sharedMemSize>>>(
            /* ranks and offsets */
            %(conn_args)s
            /* computation data */
            %(add_args)s
            /* result */
            %(target_arg)s 
        );

    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "cu_proj%(id_proj)s_psp: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }
""",
    'thread_init': {
        'float': {
            'sum': "0.0f",
            'min': "FLT_MAX",
            'max': "FLT_MIN",
            'mean': "0.0f"
        },
        'double': {
            'sum': "0.0",
            'min': "DBL_MAX",
            'max': "DBL_MIN",
            'mean': "0.0"
        }
    }
}

rate_psp_kernel_multi_warp = {
    # Adapted from Bell and Garland 2008, taken from https://code.google.com/archive/p/cusp-library/downloads (26. mar. 2018)
    #
    #   HD (2019): I needed to add a volatile variable, otherwise the results were wrong ...
    #   HD (2020): I replaced the local reduction by a warp primitive introduced with CUDA 9.0
    #   HD (2021): I replaced the WARP_SIZE by hard-coded 32 (all present architectures still have 32 threads as warp size)
    #              I replaced the BLOCK_SIZE by hard-coded 64 ... this should be a template parameter but this is not working currently
    'body': {
        'sum':"""
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    __shared__ %(size_type)s ptrs[64/32][2];

    const %(idx_type)s thread_id   = 64 * blockIdx.x + threadIdx.x;  // global thread index
    const %(idx_type)s thread_lane = threadIdx.x & (32-1);            // thread index within the warp
    const %(idx_type)s warp_id     = thread_id   / 32;                // global warp index
    const %(idx_type)s warp_lane   = threadIdx.x / 32;                // warp index within the CTA
    const %(idx_type)s num_warps   = (64 / 32) * gridDim.x;   // total number of active warps

    for(%(idx_type)s row = warp_id; row < post_size; row += num_warps){

        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = row_ptr[row + thread_lane];
        const %(size_type)s row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
        const %(size_type)s row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

        // compute local sum
        %(float_prec)s sum = 0;
        for(%(size_type)s j = row_start + thread_lane; j < row_end; j += 32)
            sum += %(psp)s

        // reduce local sums to row sum (ASSUME: warpsize 32)
        sum = warp_reduce<%(float_prec)s, 32>(sum);

        // first thread writes warp result
        if (thread_lane == 0)
            %(target_arg)s[rank_post[row]] = sum;
    }
}
"""
    },
    'header': """__global__ void cu_proj%(id)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        const unsigned int BLOCK_SIZE = 64;
        const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
        const unsigned int MAX_BLOCKS = MAX_THREADS / BLOCK_SIZE;
        const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO( proj%(id_proj)s.nb_dendrites(), WARPS_PER_BLOCK));

        cu_proj%(id_proj)s_psp<<< proj%(id_proj)s._nb_blocks, 64>>>(
            /* ranks and offsets */
            %(conn_args)s
            /* computation data */
            %(add_args)s
            /* result */
            %(target_arg)s
        );

    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "cu_proj%(id_proj)s_psp: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }
""",
    'thread_init': {
        'float': {
            'sum': "0.0f",
            'min': "FLT_MAX",
            'max': "FLT_MIN",
            'mean': "0.0f"
        },
        'double': {
            'sum': "0.0",
            'min': "DBL_MAX",
            'max': "DBL_MIN",
            'mean': "0.0"
        }
    }
}

spike_event_transmission = {
    'post_to_pre': {
        'body': """// gpu device kernel for projection %(id)s
__global__ void cu_proj%(id)s_psp( const long int t, const %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_arg)s %(kernel_args)s ) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    while ( bid < *num_events ) {
        int pre_index = spiked[bid];

        int j = col_ptr[pre_index] + tid;

        while ( j < col_ptr[pre_index+1] ) {
            int syn_idx = inv_idx[j];
            int post_rank = row_idx[j];

            // event-driven
%(event_driven)s

            // increase of conductance
%(psp)s

            // pre-spike statements
%(pre_event)s

            j += blockDim.x;
        }

        bid += gridDim.x;
    }
}
""",
        'header': """__global__ void cu_proj%(id)s_psp( const long int t, const %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_header)s %(kernel_args)s);
""",
        'call': """
    if ( pop%(id_pre)s._active && (pop%(id_pre)s.spike_count > 0) && proj%(id_proj)s._transmission ) {
    #if defined (__proj%(id_proj)s_%(target)s_nb__)
        unsigned int tpb = __proj%(id_proj)s_%(target)s_tpb__;
    #else
        unsigned int tpb = proj%(id_proj)s._threads_per_block;
    #endif
        unsigned int nb = static_cast<unsigned int>(pop%(id_pre)s.spike_count);

        // compute psp using backward view ...
        cu_proj%(id_proj)s_psp<<< nb, tpb, 0, proj%(id_proj)s.stream >>>( 
            t, dt, proj%(id_proj)s._plasticity, pop%(id_pre)s.gpu_spiked, pop%(id_pre)s.gpu_spike_count, 
            /* connectivity */
            %(conn_args)s
            /* kernel config */
            %(kernel_args)s
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t err_psp_proj%(id_proj)s = cudaGetLastError();
        if( err_psp_proj%(id_proj)s != cudaSuccess) {
            std::cout << "proj%(id_proj)s_psp (" << t << "): " << std::endl;
            std::cout << "   " << cudaGetErrorString(err_psp_proj%(id_proj)s) << std::endl;
        }
    #endif
    }
"""
    },
    #
    # This kernel computes the post-synaptic potential for the pre1st structures.
    'pre_to_post': {
        'body': """// gpu device kernel for projection %(id)s
    __global__ void cu_proj%(id)s_psp( %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* spike_count, %(conn_arg)s %(kernel_args)s ) {
        int idx = threadIdx.x;
        int b_idx = blockIdx.x;
        
        while (b_idx < *spike_count) {
            idx += row_ptr[spiked[b_idx]];
            while (idx < row_ptr[spiked[b_idx]+1]){
                atomicAdd(&g_target[col_idx[idx]], w[idx]);
			    //g_target[col_idx[idx]] += w[idx];
                idx += blockDim.x;
		    }
            b_idx += gridDim.x;
        }

    }
    """,
        'header': """__global__ void cu_proj%(id)s_psp( %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* spike_count,  %(conn_header)s %(kernel_args)s);
    """,
        'call': """
        if ( pop%(id_pre)s._active && (pop%(id_pre)s.spike_count > 0) && proj%(id_proj)s._transmission) {
            int tpb = 1024;//__pop%(id_pre)s_pop%(id_post)s_%(target)s_tpb__;
            int nbBlocks = pop%(id_pre)s.spike_count;
            
            cu_proj%(id_proj)s_psp<<< nbBlocks, tpb, 0, proj%(id_proj)s.stream >>>( dt, proj%(id_proj)s._plasticity, pop%(id_pre)s.gpu_spiked, pop%(id_pre)s.gpu_spike_count, %(conn_args)s %(kernel_args)s);

        #ifdef _DEBUG
            cudaDeviceSynchronize();
            cudaError_t err_psp_proj%(id_proj)s = cudaGetLastError();
            if( err_psp_proj%(id_proj)s != cudaSuccess) {
                std::cout << "proj%(id_proj)s_psp (" << t << "): " << std::endl;
                std::cout << "   " << cudaGetErrorString(err_psp_proj%(id_proj)s) << std::endl;
            }
        #endif
        }
    """
    }

}

spike_continous_transmission = {
    #
    # This kernel computes the post-synaptic potential for continous
    # transmission using the forward view of connectivty data.
    #
    # ATTENTION: post_idx and post_rank diverge in case of non-existant
    #            dendrites
    #
    # TODO: it might be more effective to split this kernel into two functions ...
    'post_to_pre': {
        'body': """// gpu device kernel for projection %(id_proj)s
__global__ void cu_proj%(id_proj)s_cont_psp( %(float_prec)s dt, bool plasticity, int post_size, int* post_ranks, 
                                            /* connectivity */
                                            int* row_ptr, int* col_idx, %(float_prec)s *w
                                            /* additional arguments */
                                            %(kernel_args)s
                                            /* target */
                                            , %(float_prec)s* %(target_arg)s ) 
{
    int post_idx = blockIdx.x;
    int tid = threadIdx.x;
    extern %(float_prec)s __shared__ sdata[];
    while ( post_idx < post_size ) {
        // which dendrite we are working on
        int post_rank = post_ranks[post_idx];
        size_t syn_idx = row_ptr[post_rank] + tid;
        %(float_prec)s localSum = 0.0;
        while( syn_idx < row_ptr[post_rank+1] ) {
            localSum += %(psp)s
            syn_idx += blockDim.x;
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
            volatile %(float_prec)s* smem = sdata;
            smem[tid] = localSum = localSum + smem[tid + 16];
            smem[tid] = localSum = localSum + smem[tid +  8];
            smem[tid] = localSum = localSum + smem[tid +  4];
            smem[tid] = localSum = localSum + smem[tid +  2];
            smem[tid] = localSum = localSum + smem[tid +  1];
        }
        // write result for this block to global mem
        if (tid == 0)
        {
            %(target_arg)s[post_rank] += sdata[0];
        }
        __syncthreads();
        post_idx += gridDim.x;
    }
}
""",
        'header': """__global__ void cu_proj%(id)s_event_psp( %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, int* col_ptr, int* row_idx, int* inv_idx, %(float_prec)s *w %(kernel_args)s);
__global__ void cu_proj%(id)s_cont_psp( %(float_prec)s dt, bool plasticity, int post_size, int* post_ranks, int* row_ptr, int* col_idx, %(float_prec)s *w %(kernel_args)s, %(float_prec)s* %(target_arg)s );
""",
        'call': """
    if ( pop%(id_pre)s._active && proj%(id_proj)s._transmission ) {
    #if defined (__proj%(id_proj)s_%(target)s_nb__)
        unsigned int tpb = __proj%(id_proj)s_%(target)s_tpb__;
    #else
        unsigned int tpb = proj%(id_proj)s._threads_per_block;
    #endif

        // compute continous transmission using forward view ...
        cu_proj%(id_proj)s_cont_psp<<< proj%(id_proj)s.nb_dendrites(), tpb, tpb*sizeof(%(float_prec)s), proj%(id_proj)s.stream >>>( 
            dt, proj%(id_proj)s._plasticity, proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.gpu_post_rank, 
            /* connectivity */
            proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_w
            /* additional arguments */ 
            %(kernel_args)s
            /* target */
            %(target_arg)s
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        cudaError_t err_psp_proj%(id_proj)s = cudaGetLastError();
        if( err_psp_proj%(id_proj)s != cudaSuccess) {
            std::cout << "proj%(id_proj)s_psp (" << t << "): " << std::endl;
            std::cout << "   " << cudaGetErrorString(err_psp_proj%(id_proj)s) << std::endl;
        }
    #endif
    }
"""
    }
}

# Update of global synaptic equations, consist of body (annarchyDevice.cu),
# header and call semantic (take place in ANNarchyHost.cu)
global_synapse_update = {
    'body': """
// gpu device kernel for projection %(id_proj)s
__global__ void cuProj%(id_proj)s_global_step(
    /* default params */
    const long int t, const %(float_prec)s dt
    /* additional params */
    %(kernel_args)s,
    /* plasticity enabled */
    bool plasticity )
{
%(pre_loop)s
%(global_eqs)s
}
""",
    'header': """__global__ void cuProj%(id_proj)s_global_step(const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'call': """
        // global update
        cuProj%(id_proj)s_global_step<<< 1, 1, 0, proj%(id_proj)s.stream>>>(
            /* default args*/
            t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( global_step != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString( err ) << std::endl;
        }
    #endif
"""
}

# Update of semiglobal synaptic equations, consist of body (annarchyDevice.cu),
# header and call semantic (take place in ANNarchyHost.cu)
semiglobal_synapse_update = {
    'body': """
// gpu device kernel for projection %(id_proj)s
__global__ void cuProj%(id_proj)s_semiglobal_step(
    const %(idx_type)s post_size, const %(idx_type)s* __restrict__ rank_post,
    /* default params */
    const long int t, const %(float_prec)s dt
    /* additional params */
    %(kernel_args)s,
    /* plasticity enabled */
    bool plasticity
) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;

%(pre_loop)s
    while ( i < post_size ) {
%(semiglobal_eqs)s

        i += gridDim.x * blockDim.x;
    }
}
""",
    'header': """__global__ void cuProj%(id_proj)s_semiglobal_step(%(idx_type)s post_size, const %(idx_type)s* __restrict__ rank_post, const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'call': """
        // semiglobal update
    #if defined (__proj%(id_proj)s_%(target)s_tpb__)
        cuProj%(id_proj)s_semiglobal_step<<< __proj%(id_proj)s_%(target)s_nb__, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream >>>(
            proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.gpu_post_rank,
            /* default args*/
            t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );
    #else
        nb_blocks = ceil( %(float_prec)s(proj%(id_proj)s.nb_dendrites()) / %(float_prec)s(proj%(id_proj)s._threads_per_block));
        cuProj%(id_proj)s_semiglobal_step<<< nb_blocks, proj%(id_proj)s._threads_per_block, 0, proj%(id_proj)s.stream >>>(
            proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.gpu_post_rank,
            /* default args*/
            t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );
    #endif

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess) {
            std::cout << "proj%(id_proj)s_semiglobal_step: " << cudaGetErrorString( err ) << std::endl;
        }
    #endif
"""
}

# Update of local synaptic equations, consist of body (annarchyDevice.cu),
# header and call semantic (take place in ANNarchyHost.cu)
local_synapse_update = {
    'body': """
// gpu device kernel for projection %(id_proj)s
__global__ void cuProj%(id_proj)s_local_step(
    /* connectivity */
    %(conn_args)s,
    /* default params */
    const long int t, const %(float_prec)s dt
    /* additional params */
    %(kernel_args)s,
    /* plasticity enabled */
    bool plasticity 
) {
    int i = blockIdx.x;
    %(idx_type)s rk_post = rank_post[i];
    %(size_type)s j = row_ptr[rk_post] + threadIdx.x;
    %(size_type)s C = row_ptr[rk_post+1];
%(pre_loop)s

    // Updating local variables of projection %(id_proj)s
    while ( j < C )
    {
%(local_eqs)s

        j += blockDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id_proj)s_local_step(%(conn_args)s, const long int t, const %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
        'call': """
        // local update
    #if defined (__proj%(id_proj)s_%(target)s_tpb__)
        cuProj%(id_proj)s_local_step<<< __proj%(id_proj)s_nb__, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream >>>(
            pop%(id_post)s.size,
            /* connectivity */
            %(conn_args_call)s
            /* default args*/
            , t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );
    #else
        cuProj%(id_proj)s_local_step<<< proj%(id_proj)s._nb_blocks, proj%(id_proj)s._threads_per_block, 0, proj%(id_proj)s.stream >>>(
            /* connectivity */
            %(conn_args_call)s
            /* default args*/
            , t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );
    #endif

    #ifdef _DEBUG
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if ( err != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString( err ) << std::endl;
        }
    #endif
""",
}

# call semantic for global, semiglobal and local kernel
synapse_update_call = """
    // proj%(id_proj)s: pop%(pre)s -> pop%(post)s
    if ( proj%(id_proj)s._transmission && proj%(id_proj)s._update && proj%(id_proj)s._plasticity && ( (t - proj%(id_proj)s._update_offset)%%proj%(id_proj)s._update_period == 0L)) {
        %(float_prec)s _dt = dt * proj%(id_proj)s._update_period;
#ifdef _DEBUG
    cudaError_t err;
#endif
%(global_call)s
int nb_blocks;
%(semiglobal_call)s
%(local_call)s
    }
"""

#
# Evaluation of post-event equations
#
spike_postevent = {
    'post_to_pre': {
        #
        # Called if storage_order is 'post_to_pre'. The vector pop%(id).gpu_spiked must be interpreted
        # as a boolean array. The parallelization happens across pop%(id).spike_count blocks.
        #
        'body': """// Projection %(id_proj)s: post-synaptic events
__global__ void cuProj%(id_proj)s_postevent( const long int t, const %(float_prec)s dt, bool plasticity, int *post_rank, int* spiked, %(conn_args)s %(float_prec)s* w %(add_args)s ) {
    int i = spiked[blockIdx.x]; // post-synaptic
    int j = row_ptr[i]+threadIdx.x;        // pre-synaptic

    while ( j < row_ptr[i+1] ) {
        int rk_post = post_rank[i];

    // event-driven
%(event_driven)s
    // post-event
%(post_code)s
        j+= blockDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id_proj)s_postevent( const long int t, const %(float_prec)s dt, bool plasticity, int *post_rank, int* spiked, %(conn_args)s %(float_prec)s* w %(add_args)s );
""",
        # Each cuda block compute one of the spiking post-synaptic neurons
        'call': """
    if ( proj%(id_proj)s._transmission && pop%(id_post)s._active && (pop%(id_post)s.spike_count > 0) ) {
    #if defined (__proj%(id_proj)s_%(target)s_nb__)
        int tpb = __proj%(id_proj)s_%(target)s_tpb__;
    #else
        int tpb = proj%(id_proj)s._threads_per_block;
    #endif

        cuProj%(id_proj)s_postevent<<< pop%(id_post)s.spike_count, tpb >>>(
            t, dt, proj%(id_proj)s._plasticity, proj%(id_proj)s.gpu_post_rank, pop%(id_post)s.gpu_spiked
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
"""
    }
}

conn_templates = {
    # connectivity representation
    'conn_header' : "const %(idx_type)s post_size, const %(idx_type)s* __restrict__  rank_post, const %(size_type)s* __restrict__ row_ptr, const %(idx_type)s* __restrict__ rank_pre",
    'conn_call' : "proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.gpu_post_rank, proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank",

    # launch config
    'launch_config': init_launch_config,

    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete':attribute_cpp_delete,
    'host_to_device': attribute_host_to_device,
    'device_to_host': attribute_device_to_host,
    'delay': delay,
    'event_driven': event_driven,

    # operations
    'rate_psp': rate_psp_kernel,
    #'rate_psp': rate_psp_kernel_multi_warp, # Alternative implementation of continuous transmission
    'spike_transmission': {
        'event_driven': spike_event_transmission,
        'continous': spike_continous_transmission,
    },
    'synapse_update': {
        'global': global_synapse_update,
        'semiglobal': semiglobal_synapse_update,
        'local': local_synapse_update,
        'call': synapse_update_call
    },
    'post_event': spike_postevent
}

conn_ids = {
    'local_index': "[j]",
    'semiglobal_index': '[i]',
    'global_index': '[0]',
    'pre_index': '[rank_pre[j]]',
    'post_index': '[rank_post[i]]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_',
    'delay_nu' : '[delay[j]-1]', # non-uniform delay
}
