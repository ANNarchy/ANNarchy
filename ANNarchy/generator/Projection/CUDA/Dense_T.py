"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
# (directly imported by CodeGenerator if needed)
additional_global_functions = ""

launch_config = {
    'init': """
        _threads_per_block = 0;
        _nb_blocks = 0;

    #ifdef _DEBUG
        std::cout << "Kernel configuration is a fixed 2D kernel" << std::endl;
    #endif
""",
    'update': """
        // Generate the kernel launch configuration
        _threads_per_block = 0;
        _nb_blocks = 0;

    #ifdef _DEBUG
        std::cout << "Kernel configuration is a fixed 2D kernel" << std::endl;
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
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        %(name)s_host_to_device = true;
        %(name)s_device_to_host = t;
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(static_cast<%(type)s>(%(init)s));
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
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'semiglobal': """
        // Semiglobal %(attr_type)s %(name)s
        size_in_bytes += sizeof(std::vector<%(ctype)s>);
        size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
""",
    'global': """
        // Global %(attr_type)s %(name)s
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
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), this->num_rows_ * this->num_columns_ * sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), this->num_rows_ * sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            //cudaMemcpy( gpu_%(name)s, &%(name)s, sizeof( %(type)s ), cudaMemcpyHostToDevice);
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
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, num_rows_ * num_columns_ * sizeof( %(type)s ), cudaMemcpyDeviceToHost);
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
            //cudaMemcpy( %(name)s.data(), gpu_%(name)s, post_ranks_.size() * sizeof(%(type)s), cudaMemcpyDeviceToHost);
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
            //cudaMemcpy( &%(name)s, gpu_%(name)s, sizeof(%(type)s), cudaMemcpyDeviceToHost);
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

        'pyx_struct':
"""
        # Uniform delay
        int delay""",
        'init': """
    delay = delays[0][0];
""",
        'pyx_wrapper_init':
"""
        proj%(id_proj)s.delay = syn.uniform_delay""",
        'pyx_wrapper_accessor':
"""
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
    'pyx_wrapper_init': """
        proj%(id_proj)s._last_event = vector[long]( syn._matrix.num_elements(), -10000)
"""
}

spike_event_transmission = {
    'device_kernel': """// gpu device kernel for projection %(id_proj)s
__global__ void cu_proj%(id_proj)s_psp( const long int t, const %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_args_header)s %(kernel_args_header)s ) {
    %(idx_type)s rk_pre = spiked[blockIdx.x];    // which neuron spiked
    %(idx_type)s rk_post = threadIdx.x;

    while (rk_post < row_size) {
        %(size_type)s j = rk_pre * row_size + rk_post;    // column-major indexing

        // event-driven
    %(event_driven)s

        // increase of conductance
    %(psp)s

        // pre-spike statements
    %(pre_event)s

        // proceed to the next post-synaptic neuron
        rk_post += blockDim.x;
    }
}
""",
    'invoke_kernel': """
void proj%(id_proj)s_psp(RunConfig cfg, const long int t, const %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_args_header)s %(kernel_args_header)s) {
    cu_proj%(id_proj)s_psp<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(
        t, dt, plasticity,
        /* pre-synaptic events */
        spiked, num_events,
        /* connectivity */
        %(conn_args_invoke)s
        /* kernel config */
        %(kernel_args_invoke)s
    );
}
""",
    'kernel_decl': """void proj%(id_proj)s_psp(RunConfig cfg, const long int t, const %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_args_header)s %(kernel_args_header)s);
""",
    'host_call': """
    if ( pop%(id_pre)s->_active && (pop%(id_pre)s->spike_count > 0) && proj%(id_proj)s->_transmission ) {
        int tpb = 32;
        int nb = pop%(id_pre)s->spike_count;

        if (pop%(id_pre)s->spike_count > 0) {
            // compute psp
            proj%(id_proj)s_psp(
                RunConfig(nb, tpb, 0, proj%(id_proj)s->stream),
                t, dt, proj%(id_proj)s->_plasticity,
                /* pre-synaptic events */
                pop%(id_pre)s->gpu_spiked, pop%(id_pre)s->gpu_spike_count,
                /* connectivity */
                %(conn_args)s
                /* kernel config */
                %(kernel_args)s
            );
        }
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
    %(idx_type)s post_size, %(idx_type)s pre_size, char* mask,
    /* default params */
    const long int t, const %(float_prec)s dt
    /* additional params */
    %(kernel_args)s,
    /* plasticity enabled */
    bool plasticity
) {
    %(idx_type)s rk_pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (rk_pre < pre_size) {
%(pre_loop)s

        // Updating local variables of projection %(id_proj)s
        for (%(idx_type)s rk_post = 0; rk_post < post_size; rk_post++)
        {
            %(size_type)s j = rk_pre * post_size + rk_post;
            if (mask[j]) {
%(local_eqs)s
            }

            j += blockDim.x * gridDim.x;
        }
    }
}
""",
    'invoke_kernel': """
void proj%(id_proj)s_local_step(RunConfig cfg, %(idx_type)s post_size, %(idx_type)s pre_size, char* mask, const long int t, const %(float_prec)s dt %(kernel_args)s, bool plasticity) {
    cuProj%(id_proj)s_local_step<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        /* default args*/
        post_size, pre_size, mask, t, dt
        /* kernel args */
        %(kernel_args_call)s
        /* synaptic plasticity */
        , plasticity
    );
}
""",
    'kernel_decl': """void proj%(id_proj)s_local_step(RunConfig cfg, %(idx_type)s post_size, %(idx_type)s pre_size, char* mask, const long int t, const %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'host_call': """
        // local update
    #if defined (__proj%(id_proj)s_%(target)s_tpb__)
        RunConfig proj%(id_proj)s_local_step_cfg = RunConfig(__proj%(id_proj)s_nb__, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s->stream);
    #else
        proj%(id_proj)s->_threads_per_block = 32;
        proj%(id_proj)s->_nb_blocks = static_cast<int>(ceil ( static_cast<%(float_prec)s>(pop%(id_pre)s.size) / static_cast<%(float_prec)s>(proj%(id_proj)s->_threads_per_block)));
        RunConfig proj%(id_proj)s_local_step_cfg = RunConfig(proj%(id_proj)s->_nb_blocks, proj%(id_proj)s->_threads_per_block, 0, proj%(id_proj)s->stream);
    #endif
        proj%(id_proj)s_local_step(
            proj%(id_proj)s_local_step_cfg,
            /* default args*/
            pop%(id_post)s->size, pop%(id_pre)s->size, proj%(id_proj)s->device_mask(), t, _dt
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

spike_postevent = {
    'device_kernel': """// Projection %(id_proj)s: post-synaptic events
__global__ void cuProj%(id_proj)s_postevent(
    // default constants
    const long int t, const %(float_prec)s dt, bool plasticity,
    // events
    int* spiked, long int* pre_last_spike,
    // connectivity
    %(conn_args)s,
    // weights and other arguments
    %(float_prec)s* w %(add_args)s )
{
    // each CUDA block computes one row
    int rk_post  = spiked[blockIdx.x];
    int rk_pre = threadIdx.x;

    while ( rk_pre < pre_size ) {
        int j = rk_pre * post_size + rk_post;
        if (mask[j]) {
    // event-driven
%(event_driven)s

    // post-event
%(post_code)s
        }
        rk_pre += blockDim.x;
    }
}
""",
    'invoke_kernel': """
void proj%(id_proj)s_postevent(RunConfig cfg, const long int t, const %(float_prec)s dt, bool plasticity, int* spiked, long int* pre_last_spike, %(conn_args)s, %(float_prec)s* w %(add_args)s ){
    cuProj%(id_proj)s_postevent<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        t, dt, plasticity,
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
    'kernel_decl': "void proj%(id_proj)s_postevent(RunConfig cfg, const long int t, const %(float_prec)s dt, bool plasticity, int* spiked, long int* pre_last_spike, %(conn_args)s, %(float_prec)s* w %(add_args)s );",
    'host_call': """
    if ( proj%(id_proj)s->_transmission && pop%(id_post)s->_active && (pop%(id_post)s->spike_count > 0) ) {
    #if defined (__proj%(id_proj)s_%(target)s_nb__)
        int tpb = __proj%(id_proj)s_%(target)s_tpb__;
    #else
        int tpb = 64;
    #endif

        proj%(id_proj)s_postevent(
            RunConfig(pop%(id_post)s->spike_count, tpb, 0, proj%(id_proj)s->stream),
            t, dt, proj%(id_proj)s->_plasticity,
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
    'conn_header': "const %(idx_type)s post_size, const %(idx_type)s pre_size, const char* mask",
    'conn_call': "pop%(id_post)s->size, pop%(id_pre)s->size, proj%(id_proj)s->device_mask()",
    'conn_kernel': "post_size, pre_size, mask",

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
    'event_driven': event_driven,

    #operations
    'rate_psp': None,
    'spike_transmission': {
        'event_driven': spike_event_transmission,
        'continuous': None
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
    'semiglobal_index': '[rk_post]',
    'global_index': '[0]',
    'pre_index': '[rk_pre]',
    'post_index': '[rk_post]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_'
}
