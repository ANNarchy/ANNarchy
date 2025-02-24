"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
# (directly imported by CodeGenerator if needed)
additional_global_functions = """
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
"""

launch_config = {
    'init': """
        _threads_per_block = 64;
        _nb_blocks = std::min<unsigned int>(ceil(static_cast<double>(nb_dendrites())/static_cast<double>(_threads_per_block)), 65535);
    
    #ifdef _DEBUG
        std::cout << "Kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif
""",
    'update': """
    #ifdef _DEBUG
        std::cout << "The CSR vector implementation is fixed to a specific thread configuration." << std::endl;
    #endif
"""
}

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

rate_psp_kernel = {
    # Adapted from Bell and Garland 2008, taken from https://code.google.com/archive/p/cusp-library/downloads (26. mar. 2018)
    #
    #   HD (2019): I needed to add a volatile variable, otherwise the results were wrong ...
    #   HD (2020): I replaced the local reduction by a warp primitive introduced with CUDA 9.0
    'device_kernel': {
        'sum':"""
template<unsigned int BLOCK_SIZE, unsigned int WARP_SIZE>
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    __shared__ %(size_type)s ptrs[BLOCK_SIZE/WARP_SIZE][2];

    const %(idx_type)s thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
    const %(idx_type)s thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
    const %(idx_type)s warp_id     = thread_id   / WARP_SIZE;                // global warp index
    const %(idx_type)s warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    const %(idx_type)s num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

    for(%(idx_type)s row = warp_id; row < post_size; row += num_warps){

        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = row_ptr[row + thread_lane];
        const %(size_type)s row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
        const %(size_type)s row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

        // compute local sum
        %(float_prec)s sum = 0;
        for(%(size_type)s j = row_start + thread_lane; j < row_end; j += WARP_SIZE)
            sum += %(psp)s

        // reduce local sums to row sum (ASSUME: warpsize 32)
        sum = warp_reduce<%(float_prec)s, WARP_SIZE>(sum);

        // first thread writes warp result
        if (thread_lane == 0)
            %(target_arg)s[rank_post[row]] = sum;
    }
}
"""
    },
    'invoke_kernel': """
void launch_proj%(id_proj)s_psp(const unsigned int nb_blocks, const unsigned int threads_per_block, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {

    switch (threads_per_block)
    {
        case 64:
        {
            cu_proj%(id_proj)s_psp<64,32><<< nb_blocks, threads_per_block>>>(
                /* ranks and offsets */
                %(conn_args_call)s
                /* computation data */
                %(add_args_call)s
                /* result */
                %(target_arg_call)s
            );
        }break;

        default:
        {
            std::cerr << "The kernel configuration tpb = " << threads_per_block << " is not supported" << std::endl;
        }
    }
}
""",    
    'kernel_decl': """void launch_proj%(id_proj)s_psp(const unsigned int nb_blocks, const unsigned int threads_per_block, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'host_call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s->_active && proj%(id_proj)s->_transmission ) {
        const unsigned int BLOCK_SIZE = proj%(id_proj)s->_threads_per_block;
        const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
        const unsigned int MAX_BLOCKS = MAX_THREADS / BLOCK_SIZE;
        const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO( proj%(id_proj)s->nb_dendrites(), WARPS_PER_BLOCK));

        launch_proj%(id_proj)s_psp(
            NUM_BLOCKS, BLOCK_SIZE,
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

conn_templates = {
    # connectivity representation
    'conn_header' : "const %(idx_type)s post_size, const %(idx_type)s* __restrict__  rank_post, const %(size_type)s* __restrict__ row_ptr, const %(idx_type)s* __restrict__ rank_pre",
    'conn_call' : "proj%(id_proj)s->nb_dendrites(), proj%(id_proj)s->gpu_post_rank, proj%(id_proj)s->gpu_row_ptr, proj%(id_proj)s->gpu_pre_rank",
    'conn_kernel': "post_size, rank_post, row_ptr, rank_pre",

    # launch config
    'launch_config': launch_config,

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
    'spike_transmission': {
        'event_driven': None,
        'continuous': None,
    },
    'synapse_update': {
        'global': None,
        'semiglobal': None,
        'local': None,
        'call': None
    },
    'post_event': None
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