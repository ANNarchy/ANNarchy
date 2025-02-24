"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
# (directly imported by CodeGenerator if needed)
additional_global_functions = ""

launch_config = {
    'init': """
        _threads_per_block = 64;
        _nb_blocks = std::min<unsigned int>(nb_dendrites(), 65535);
    
    #ifdef _DEBUG
        std::cout << "Kernel configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
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
        std::cout << "Updated configuration: " << _nb_blocks << ", " << _threads_per_block << std::endl;
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
    # A single value for all synapses
    'uniform': {
        'declare': """
    // Uniform delay
    int delay;

    int get_delay() { return delay; }
    int get_dendrite_delay(int idx) { return delay; }
    void set_delay(int delay) { this->delay = delay; }
""",
        'init': """
    delay = delays[0][0];
"""
    },
    # An individual value for each synapse
    'nonuniform_rate_coded': None,
    # An individual value for each synapse and a
    # buffer for spike events
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

spike_event_transmission = {
    'device_kernel': """// gpu device kernel for projection %(id_proj)s
__global__ void cu_proj%(id_proj)s_psp(const long int t, %(float_prec)s dt, bool plasticity, int *spiked, unsigned int spike_count, %(conn_args_header)s %(kernel_args_header)s ) {
    int b_idx = blockIdx.x;
    int pre_rank = spiked[b_idx];

    while (b_idx < spike_count) {
        int syn_idx = threadIdx.x + row_ptr[pre_rank];
        while (syn_idx < row_ptr[pre_rank+1]){
            // event-driven
%(event_driven)s

            // increase of conductance
%(psp)s

            // pre-spike statements
%(pre_event)s

            syn_idx += blockDim.x;
        }
        b_idx += gridDim.x;
    }

}
""",
    'invoke_kernel': """
void proj%(id_proj)s_psp(RunConfig cfg,  const long int t, %(float_prec)s dt, bool plasticity, int *spiked, unsigned int spike_count,  %(conn_args_header)s %(kernel_args_header)s) {
    cu_proj%(id_proj)s_psp<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        /* default events */
        t, dt, plasticity,
        /* pre-synaptic events */
        spiked, spike_count,
        /* connectivity */
        %(conn_args_invoke)s
        /* other arguments */
        %(kernel_args_invoke)s
    );
}
""",
    'kernel_decl': """void proj%(id_proj)s_psp(RunConfig cfg,  const long int t, %(float_prec)s dt, bool plasticity, int *spiked, unsigned int spike_count,  %(conn_args_header)s %(kernel_args_header)s);
""",
    'host_call': """
        if ( pop%(id_pre)s->_active && (%(pre_spike_count)s > 0) && proj%(id_proj)s->_transmission) {
            int tpb = 1024;//__pop%(id_pre)s_pop%(id_post)s_%(target)s_tpb__;
            int nbBlocks = %(pre_spike_count)s;
            
            proj%(id_proj)s_psp(
                RunConfig(nbBlocks, tpb, 0, proj%(id_proj)s->stream),
                /* default events */
                t, dt, proj%(id_proj)s->_plasticity,
                /* pre-synaptic spike events */
                %(pre_spike_events)s, %(pre_spike_count)s,
                /* connectivity */
                %(conn_args)s
                /* other arguments */
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
}

# Update of local synaptic equations, consist of body (annarchyDevice.cu),
# header and call semantic (take place in ANNarchyHost.cu)
local_synapse_update = {
    'device_kernel': """
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
    %(idx_type)s rk_pre = blockIdx.x;
    %(size_type)s j = row_ptr[rk_pre] + threadIdx.x;
    %(size_type)s C = row_ptr[rk_pre+1];
%(pre_loop)s

    // Updating local variables of projection %(id_proj)s
    while ( j < C )
    {
%(local_eqs)s

        j += blockDim.x;
    }
}
""",
        'invoke_kernel': """
void proj%(id_proj)s_local_step(RunConfig cfg, %(conn_args)s, const long int t, const %(float_prec)s dt %(kernel_args)s, bool plasticity) {
    cuProj%(id_proj)s_local_step<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        /* connectivity */
        %(conn_args_call)s
        /* default args*/
        , t, dt
        /* kernel args */
        %(kernel_args_call)s
        /* synaptic plasticity */
        , plasticity
    );
}
""",
        'kernel_decl': """void proj%(id_proj)s_local_step(RunConfig cfg, %(conn_args)s, const long int t, const %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
        'host_call': """
        // local update
    #if defined (__proj%(id_proj)s_%(target)s_tpb__)
        RunConfig proj%(id_proj)s_local_step_cfg = RunConfig(__proj%(id_proj)s_nb__, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s->stream);
    #else
        RunConfig proj%(id_proj)s_local_step_cfg = RunConfig(proj%(id_proj)s->_nb_blocks, proj%(id_proj)s->_threads_per_block, 0, proj%(id_proj)s->stream);
    #endif
        proj%(id_proj)s_local_step(
            proj%(id_proj)s_local_step_cfg,
            /* connectivity */
            %(conn_args_call)s
            /* default args*/
            , t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s->_plasticity
        );

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
    if ( proj%(id_proj)s->_transmission && proj%(id_proj)s->_update && proj%(id_proj)s->_plasticity && ( (t - proj%(id_proj)s->_update_offset)%%proj%(id_proj)s->_update_period == 0L)) {
        %(float_prec)s _dt = dt * proj%(id_proj)s->_update_period;
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
    #
    # Called if storage_order is 'post_to_pre'. The vector pop%(id)->gpu_spiked must be interpreted
    # as a boolean array. The parallelization happens across pop%(id)->spike_count blocks.
    #
    'device_kernel': """// Projection %(id_proj)s: post-synaptic events
__global__ void cuProj%(id_proj)s_postevent( const long int t, const %(float_prec)s dt, bool plasticity, int *post_rank, int* spiked, long int* pre_last_spike, %(conn_args)s, %(float_prec)s* w %(add_args)s ) {
    int i = spiked[blockIdx.x];         
    int syn_idx = col_ptr[i]+threadIdx.x;     // post-synaptic entries are a column

    while ( syn_idx < col_ptr[i+1] ) {
        int rk_post = post_rank[i];
        int j = inv_idx[syn_idx];

    // event-driven
%(event_driven)s
    // post-event
%(post_code)s

        syn_idx += blockDim.x;
    }
}
""",
    'invoke_kernel': """
void proj%(id_proj)s_postevent(RunConfig cfg, const long int t, const %(float_prec)s dt, bool plasticity, int *post_rank, int* spiked, long int* pre_last_spike, %(conn_args)s, %(float_prec)s* w %(add_args)s ){
    cuProj%(id_proj)s_postevent<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        t, dt, plasticity, post_rank,
        /* post-spike and pre-spike time points */
        spiked, pre_last_spike,
        /* connectivity */
        %(conn_args_invoke)s
        /* weights */
        , w
        /* other variables */
        %(add_args_invoke)s
    );
}
""",
    'kernel_decl': """void proj%(id_proj)s_postevent(RunConfig cfg, const long int t, const %(float_prec)s dt, bool plasticity, int *post_rank, int* spiked, long int* pre_last_spike, %(conn_args)s, %(float_prec)s* w %(add_args)s );
""",
    # Each cuda block compute one of the spiking post-synaptic neurons
    'host_call': """
    if ( proj%(id_proj)s->_transmission && pop%(id_post)s->_active && (pop%(id_post)s->spike_count > 0) ) {
    #if defined (__proj%(id_proj)s_%(target)s_nb__)
        int tpb = __proj%(id_proj)s_%(target)s_tpb__;
    #else
        int tpb = proj%(id_proj)s->_threads_per_block;
    #endif

        proj%(id_proj)s_postevent(
            RunConfig(pop%(id_post)s->spike_count, tpb, 0, proj%(id_proj)s->stream),
            t, dt, proj%(id_proj)s->_plasticity, proj%(id_proj)s->gpu_post_rank,
            /* post-spike and pre-spike time points */
            pop%(id_post)s->gpu_spiked, pop%(id_pre)s->gpu_last_spike,
            /* connectivity */
            %(conn_args)s
            /* weights */
            , proj%(id_proj)s->gpu_w
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

conn_templates = {
    # connectivity representation
    'conn_header' : "const %(idx_type)s post_size, const %(idx_type)s* __restrict__  rank_post, const %(size_type)s* __restrict__ row_ptr, const %(idx_type)s* __restrict__ col_idx, const %(size_type)s* __restrict__ col_ptr, const %(idx_type)s* __restrict__ row_idx, const %(idx_type)s* __restrict__ inv_idx",
    'conn_call' : "proj%(id_proj)s->nb_dendrites(), proj%(id_proj)s->gpu_post_rank, proj%(id_proj)s->gpu_row_ptr, proj%(id_proj)s->gpu_col_idx, proj%(id_proj)s->gpu_col_ptr, proj%(id_proj)s->gpu_row_idx, proj%(id_proj)s->gpu_inv_idx",
    'conn_kernel': "post_size, rank_post, row_ptr, col_idx, col_ptr, row_idx, inv_idx",

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
    'spike_transmission': {
        'event_driven': spike_event_transmission,
        'continuous': None  # HD: this code path is disabled by sanity check!
    },
    'synapse_update': {
        'global': None,
        'semiglobal': None,
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
    'post_index': '[col_idx[syn_idx]]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_',
    'delay_nu' : '[delay[j]-1]', # non-uniform delay
}