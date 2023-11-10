"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
additional_global_functions = ""

#TODO: maybe it would make more sense to split up the nb_blocks and tpb for ELL and COO
launch_config = {
    'init': """
        _threads_per_block = 64;
        _nb_blocks = std::min<unsigned int>(nb_dendrites(), 65535);

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
            _nb_blocks = std::min<unsigned int>(nb_dendrites(), 65535);
        }

    #ifdef _DEBUG
        std::cout << "Updated kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif
"""
}

attribute_decl = {
    'local': """
    // Local %(attr_type)s %(name)s
    hyb_local<%(type)s>* %(name)s;
    hyb_local_gpu<%(type)s>* gpu_%(name)s;
    long int %(name)s_device_to_host;
    bool %(name)s_host_to_device;
""",
    'semiglobal': "",
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
        %(name)s = init_matrix_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        %(name)s_host_to_device = true;
""",
    'semiglobal': "",
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
    // Local %(name)s
    size_in_bytes += %(name)s->size_in_bytes();
""",
    'semiglobal': "",
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
        %(name)s->clear();

        // %(name)s - device
        gpu_%(name)s->clear();
""",
    'semiglobal': "",
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
            cudaMemcpy( gpu_%(name)s->ell, %(name)s->ell.data(), %(name)s->ell.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            cudaMemcpy( gpu_%(name)s->coo, %(name)s->coo.data(), %(name)s->coo.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_host_to_device = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'semiglobal': "",
    'global': ""
}

attribute_device_to_host = {
    'local': "",
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
    'body': {
        'sum': ""        
    },
    'header': "",
    'host_call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        // ELLPACK - partition
        int nb_dendrites = proj%(id_proj)s.nb_dendrites();
        int nb_blocks = static_cast<int>(ceil(static_cast<double>(nb_dendrites)/32.0));
        cu_proj%(id_proj)s_psp_ell<<< nb_blocks, 32>>>(
                       nb_dendrites,
                       /* ranks and offsets */
                       proj%(id_proj)s.get_ell()->gpu_post_ranks_, proj%(id_proj)s.get_ell()->gpu_col_idx_, proj%(id_proj)s.get_ell()->get_maxnzr(), proj%(id_proj)s.get_ell()->zero_marker()
                       /* computation data */
                       %(add_args_ell)s
                       /* result */
                       %(target_arg)s );

    #ifdef _DEBUG
        auto ell_err = cudaGetLastError();
        if ( ell_err != cudaSuccess ) {
            std::cout << "cu_proj%(id_proj)s_psp (ELL-partition): " << cudaGetErrorString(ell_err) << std::endl;
        }
    #endif

        // Coordinate - partition
        size_t nb_coo_synapses = proj%(id_proj)s.get_coo()->nb_synapses();
        // check if there is something to compute ...
        if (nb_coo_synapses > 0) {
            int sharedMemSize = proj%(id_proj)s.get_coo()->segment_size() * sizeof(%(float_prec)s);
            cu_proj%(id_proj)s_psp_coo<<< proj%(id_proj)s.get_coo()->number_of_segments(), proj%(id_proj)s._threads_per_block, sharedMemSize >>>(
                proj%(id_proj)s.get_coo()->segment_size(), proj%(id_proj)s.get_coo()->gpu_segments(), proj%(id_proj)s.get_coo()->gpu_row_indices(), proj%(id_proj)s.get_coo()->gpu_column_indices()
                /* other variables */
                %(add_args_coo)s
                /* result */
                %(target_arg)s
            );

        #ifdef _DEBUG
            auto err_coo = cudaGetLastError();
            if ( err_coo != cudaSuccess ) {
                std::cout << "cu_proj%(id_proj)s_psp (COO-partition): " << cudaGetErrorString(err_coo) << std::endl;
            }
        #endif
        }
    }    
""",
    'kernel_call': "",
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
    'conn_header': None,    #constructed from ELL+COO
    'conn_call': None,      #constructed from ELL+COO
    'conn_kernel': None,

    # launch config
    'launch_config': launch_config,
 
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'attribute_cpp_size': attribute_cpp_size,
    'attribute_cpp_delete': attribute_cpp_delete,
    'host_to_device': attribute_host_to_device,
    'device_to_host': attribute_device_to_host,
    'delay': delay,

    # operations
    'rate_psp': rate_psp_kernel
}