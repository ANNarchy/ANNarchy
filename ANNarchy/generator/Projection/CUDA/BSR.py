"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
# (directly imported by CodeGenerator if needed)
additional_global_functions = ""

launch_config = {
    'init': """
        _nb_blocks = std::min<unsigned int>(block_row_size(), 65535);

        // must be multiple of tile_size^2
        unsigned int tile_size2 = get_tile_size() * get_tile_size();
        _threads_per_block = (32/tile_size2)*tile_size2;
        if (_threads_per_block == 0) {
            _threads_per_block = tile_size2;
        }

    #ifdef _DEBUG
        std::cout << _nb_blocks << ", " << _threads_per_block << std::endl;
    #endif

""",
    'update': """
        std::cout << "Adjustment of launch configuration for BSR is not supported yet." << std::endl;

    #ifdef _DEBUG
        std::cout << _nb_blocks << ", " << _threads_per_block << std::endl;
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
        if (%(name)s.empty())
            return false;
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        if (gpu_%(name)s == nullptr)
            return false;

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
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), this->tile_data_.size() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, this->tile_data_.size() * sizeof( %(type)s ), cudaMemcpyDeviceToHost);
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

# BSR implementation following Eberhardt & Hoemmen (2016) - row-per-thread
#
# In this variant, each thread computes one row in a dense block and is intended for larger block sizes.
# We ensure, that the number of threads in a CUDA block is equal to the tile-size. Further we assume
# squared dense blocks. Last but not least, each CUDA block computes at least one blocked row in the BSR.
#
rate_psp_kernel_rpt = {
    'device_kernel': {
        'sum': """
__global__ void cu_proj%(id_proj)s_psp_bsr(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    const %(idx_type)s idx = threadIdx.x;
    const %(idx_type)s block_row = blockIdx.x;
    const %(size_type)s tile_size2 = tile_size * tile_size;
    extern %(float_prec)s __shared__ loc_pre_r[];

    for (%(idx_type)s row = block_row; row < n_block_rows; row += gridDim.x) {
        const %(idx_type)s first_block = row_ptr[row];
        const %(idx_type)s last_block = row_ptr[row + 1];

        %(float_prec)s lsum = 0.0;
        for (%(idx_type)s block = first_block; block < last_block; block++)
        {
            __syncthreads();
            loc_pre_r[idx] = pre_r[col_ids[block] * tile_size +idx];
            __syncthreads();

            const %(size_type)s tile_off = block * tile_size2;
            for (%(idx_type)s col = 0; col < tile_size; col++)
                lsum += %(psp)s
        }

        %(target_arg)s[row * tile_size + idx] += lsum;
    }
}
"""
    },
    'device_header': """__global__ void cu_proj%(id)s_psp_bsr(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'host_call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        unsigned int nb_blocks = std::min<unsigned int>(proj%(id_proj)s.block_row_size(), 65535);
        size_t smem_size = proj%(id_proj)s.get_tile_size() * sizeof(%(float_prec)s);
        cu_proj%(id_proj)s_psp_bsr<<<nb_blocks, proj%(id_proj)s.get_tile_size(), smem_size>>>(
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

# BSR implementation following Eberhardt & Hoemmen (2016) - column-by-column
#
# This variant is intended for small block sizes.
#
# Keys - structure:
#
# * device_kernel
#    * sum
# * kernel_decl
# * host_call
# * thread_init
rate_psp_kernel_cbc = {
    'device_kernel': {
        'sum': """
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    unsigned int tIdx = threadIdx.x;
    unsigned int bIdx = blockIdx.x;
    extern %(float_prec)s __shared__ sdata[];
    %(idx_type)s tile_size2 = tile_size *  tile_size;

    int blocks_per_warp = floorf(blockDim.x / tile_size2);
    int block_offset = int(float(tIdx) / float(tile_size2));

    // reset
    sdata[tIdx] = 0;

    // iterate across all columns in this row (determined by blockIdx)
    for (%(idx_type)s col_idx = row_ptr[bIdx]+block_offset; col_idx < row_ptr[bIdx+1]; col_idx+=blocks_per_warp) {
        %(idx_type)s dense_col_idx = (tIdx / tile_size) %% tile_size;
        %(idx_type)s dense_val_idx = tIdx %% tile_size2;

        // which dense column, determine where to access pr
        int bcol_idx = col_ids[col_idx];
        
        // perform dense SpMV (column_major)
        const %(float_prec)s* loc_values = w + col_idx * tile_size2;
        const   %(float_prec)s* loc_pr = pre_r + bcol_idx * tile_size;

        sdata[tIdx] += loc_values[dense_val_idx] * loc_pr[dense_col_idx];
    }

    // reduction to first tile
    if (tIdx < tile_size2) {
        for(%(idx_type)s i = tIdx+tile_size2; i < blockDim.x; i+= tile_size2) {
            sdata[tIdx] += sdata[i];
        }
    }

    // reduction within tile
    // and write back result
    if (tIdx < tile_size) {
        for (int i = tIdx; i < tile_size2; i+= tile_size) {
            %(target_arg)s[bIdx * tile_size+tIdx] += sdata[i];
        }
    }
}
""",
},
    'invoke_kernel': """
void call_proj%(id_proj)s_psp(RunConfig cfg, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    // kernel launch
    cu_proj%(id_proj)s_psp<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(
        %(conn_args_call)s
        /* other variables */
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
        // kernel configuration
        RunConfig proj%(id_proj)s_psp_cfg;
        proj%(id_proj)s_psp_cfg.nb = proj%(id_proj)s._nb_blocks;
        proj%(id_proj)s_psp_cfg.tpb = proj%(id_proj)s._threads_per_block;
        proj%(id_proj)s_psp_cfg.smem_size = proj%(id_proj)s._threads_per_block * sizeof(%(float_prec)s);    // one local variable per thread
        proj%(id_proj)s_psp_cfg.stream = proj%(id_proj)s.stream;

        // invoke kernel
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
    'conn_header': "const %(idx_type)s* __restrict__ row_ptr, const %(idx_type)s* __restrict__ col_ids, const %(idx_type)s n_block_rows, const %(idx_type)s tile_size",
    'conn_call': "proj%(id_proj)s.gpu_block_row_pointer(), proj%(id_proj)s.gpu_block_column_index(), proj%(id_proj)s.block_row_size(), proj%(id_proj)s.get_tile_size()",
    'conn_kernel': "row_ptr, col_ids, n_block_rows, tile_size",

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
    'rate_psp': rate_psp_kernel_cbc,
    #'rate_psp': rate_psp_kernel_rpt,
}

conn_ids = {
    'local_index': "[tile_off + col * tile_size + idx]",
    'semiglobal_index': '[i]',
    'global_index': '[0]',
    'pre_index': '[first_col_x + col]',
    'post_index': '[row_indices[j]]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_',
}