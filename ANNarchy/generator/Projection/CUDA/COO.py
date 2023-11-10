"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
# (directly imported by CodeGenerator if needed)
additional_global_functions = ""

launch_config = {
    'init': """
        _threads_per_block = 192;
        unsigned int tmp_blocks = static_cast<unsigned int>(ceil(static_cast<double>(nb_synapses())/static_cast<double>(_threads_per_block)));
        _nb_blocks = std::min<unsigned int>(65535, tmp_blocks);

    #ifdef _DEBUG
        std::cout << "Initial kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif
""",
    'update': """
        if (nb_blocks != -1) {
            _nb_blocks = static_cast<unsigned int>(nb_blocks);
            _threads_per_block = threads_per_block;
        }else{
            _threads_per_block = threads_per_block;
            unsigned int tmp_blocks = static_cast<unsigned int>(ceil(static_cast<double>(nb_synapses())/static_cast<double>(_threads_per_block)));
            _nb_blocks = std::min<unsigned int>(65535, tmp_blocks);
        }

    #ifdef _DEBUG
        std::cout << "Updated kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
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
    'global': """
    // Global %(attr_type)s %(name)s
    %(type)s %(name)s;
    %(type)s* gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
"""
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
        %(name)s_dirty = true;
""",
    'global': """
        // Global %(attr_type)s %(name)s
        %(name)s = static_cast<%(type)s>(%(init)s);
        cudaMalloc((void**)&gpu_%(name)s, sizeof(%(type)s));
        %(name)s_dirty = true;
"""
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
        // %(name)s
        cudaFree(gpu_%(name)s);
""",
    'semiglobal': """
        // %(name)s
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
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), this->nb_synapses() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), this->nb_dendrites() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, this->nb_synapses() * sizeof( %(type)s ), cudaMemcpyDeviceToHost);
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
        if ( %(name)s_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, this->nb_dendrites() * sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
            %(name)s_device_to_host = t;
        }
""",
    'global': """
        // %(name)s: global
        if ( %(name)s_device_to_host < t ) {
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( &%(name)s, gpu_%(name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
            %(name)s_device_to_host = t;
        }
"""
}

rate_psp_kernel = {
    'device_kernel': {
        'sum': """
__global__ void cu_proj%(id_proj)s_psp_coo(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    %(size_type)s j = segments[blockIdx.x] + threadIdx.x;
    %(size_type)s C = segments[blockIdx.x+1];
    %(size_type)s block_off = segment_size * blockIdx.x;

    extern %(float_prec)s __shared__ sdata[];

    if (threadIdx.x < segment_size)
        sdata[threadIdx.x] = 0.0;
    __syncthreads();

    for( ; j < C; j += blockDim.x ) {

        %(float_prec)s sum = %(psp)s
        %(idx_type)s loc_idx = row_indices[j]-block_off;
        atomicAdd(&(sdata[loc_idx]), sum);
    }

    __syncthreads();
    if (threadIdx.x < segment_size)
        %(target_arg)s[block_off + threadIdx.x] += sdata[threadIdx.x];
}
"""
    },
    'kernel_decl': """__global__ void cu_proj%(id)s_psp_coo(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'host_call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        int sharedMemSize = proj%(id_proj)s.segment_size() * sizeof(%(float_prec)s);
        cu_proj%(id_proj)s_psp_coo<<< proj%(id_proj)s.number_of_segments(), proj%(id_proj)s._threads_per_block, sharedMemSize >>>(
            %(conn_args)s
            /* other variables */
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
    'conn_header': "const %(idx_type)s segment_size, const %(size_type)s *segments, const %(idx_type)s* __restrict__ row_indices, const %(idx_type)s* __restrict__ column_indices",
    'conn_call': "proj%(id_proj)s.segment_size(), proj%(id_proj)s.gpu_segments(), proj%(id_proj)s.gpu_row_indices(), proj%(id_proj)s.gpu_column_indices()",
    'conn_kernel': "",

    # launch config
    'launch_config': launch_config,

    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'host_to_device': attribute_host_to_device,
    'device_to_host': attribute_device_to_host,

    # operations
    'rate_psp': rate_psp_kernel,
}

conn_ids = {
    'local_index': "[j]",
    'semiglobal_index': '[i]',
    'global_index': '[0]',
    'pre_index': '[column_indices[j]]',
    'post_index': '[row_indices[j]]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_',
}