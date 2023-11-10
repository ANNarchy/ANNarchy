"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
additional_global_functions = ""

init_launch_config = {
    'init': """
        // Generate the kernel launch configuration
        _threads_per_block = get_block_size();
        _nb_blocks = static_cast<unsigned int>(ceil(static_cast<double>(nb_dendrites())/static_cast<double>(_threads_per_block)));

    #ifdef _DEBUG
        std::cout << "Kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif
""",
    'update': """
        std::cerr << "re-configuring kernel config is not yet implemented for SELL" << std::endl;
"""
}

attribute_decl = {
    'local': """
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_host_to_device;
    long int %(name)s_device_to_host;
""",
    'semiglobal': """
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
    %(type)s* gpu_%(name)s;
    bool %(name)s_host_to_device;
    long int %(name)s_device_to_host;
""",
    'global': {
        'parameter':
    """
    // Global parameter %(name)s
    %(type)s %(name)s;
""",
        'variable': """
    // Global variable %(name)s
    %(type)s %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_host_to_device;
    long int %(name)s_device_to_host;
"""
    }
}

attribute_cpp_init = {
    'local': """
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(%(init)s);
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        %(name)s_host_to_device = false;
        %(name)s_device_to_host = t;
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(%(init)s);
        gpu_%(name)s = init_vector_variable_gpu<%(type)s>(%(name)s);
        %(name)s_host_to_device = false;
        %(name)s_device_to_host = t;
""",
    'global': { 
        'parameter': """
        // Global parameter %(name)s
        %(name)s = 0.0;
""",
        'variable': """
        // Global variable %(name)s
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
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), %(name)s.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), %(name)s.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, %(name)s.size() * sizeof( %(type)s ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
            %(name)s_device_to_host = t;
        }
""",
    'semiglobal': "",
    'global': ""
}

#
#   Synaptic delays
#
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
        proj%(id_proj)s.delay = value
"""
    }
}

#
# Implement the continuous transmission for rate-code synapses.
#
rate_psp_kernel = {
    'device_kernel': {
        'sum':"""
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s) {
    //global row
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;   

    if ( row < post_size ) {
        %(float_prec)s tmp = 0.0;
        unsigned int offset = row_ptr[blockIdx.x];
        unsigned int local_max_length = (row_ptr[blockIdx.x + 1] - offset)/block_size;
        
        for (unsigned int i = 0; i < local_max_length; i++) {
            %(idx_type)s idx = i * block_size + threadIdx.x + offset;
            %(idx_type)s rk_pre = rank_pre[idx];
            tmp += %(psp)s
        }
        %(target_arg)s%(post_index)s += tmp;
    }
}
"""
    },
    'invoke_kernel': """
void call_proj%(id_proj)s_psp(RunConfig cfg, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s) {
    cu_proj%(id_proj)s_psp<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        /* ranks and offsets */
        %(conn_args_call)s
        /* computation data */
        %(add_args_call)s
        /* result */
        %(target_arg_call)s
    );
}
""",
    'kernel_decl': """void call_proj%(id_proj)s_psp(RunConfig cfg, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'host_call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        RunConfig proj%(id_proj)s_psp_cfg = RunConfig(proj%(id_proj)s._nb_blocks, proj%(id_proj)s._threads_per_block, 0, proj%(id_proj)s.stream);
        call_proj%(id_proj)s_psp (
            /* kernel config */
            proj%(id_proj)s_psp_cfg,
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
    'conn_header': "const %(idx_type)s post_size, const %(idx_type)s block_size, const %(size_type)s* __restrict__ row_ptr, const %(idx_type)s* __restrict__ rank_pre",
    'conn_call': "proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.get_block_size(), proj%(id_proj)s.d_row_ptr, proj%(id_proj)s.d_col_idx",
    'conn_kernel': "post_size, block_size, row_ptr, rank_pre",

    # launch config
    'launch_config': init_launch_config,
 
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'host_to_device': attribute_host_to_device,
    'device_to_host': attribute_device_to_host,
    'delay': delay,

    # operations
    'rate_psp': rate_psp_kernel,
    'synapse_update': {
        'global': None,
        'semiglobal': None,
        'local': None,
        'call': None
    }
}

conn_ids = {
    'local_index': "[idx]",
    'semiglobal_index': '[i]',
    'global_index': '[0]',
    'pre_index': '[rk_pre]',
    'post_index': '[row]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_'
}