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

#
# Implement the continuous signal transmission for rate-coded synapses.
#
rate_psp_kernel = {
    # Comment to if (tid < 32) block:
    #
    # now that we are using warp-synchronous programming (below)
    # we need to declare our shared memory volatile so that the compiler
    # doesn't reorder stores to it and induce incorrect behavior.
    'device_kernel': {
        'sum':"""
__global__ void cu_proj%(id_proj)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    %(idx_type)s rk_post = blockIdx.y*blockDim.y+threadIdx.y;
    %(float_prec)s __shared__ sdata[16][32];
    unsigned int tid = threadIdx.x;

    while( rk_post < post_size ) {
        sdata[threadIdx.y][threadIdx.x] = 0.0;
        %(size_type)s j = rk_post * pre_size + threadIdx.x;

        for (%(idx_type)s rk_pre = threadIdx.x; rk_pre < pre_size; rk_pre+=blockDim.x, j+= blockDim.x) {
            sdata[threadIdx.y][threadIdx.x] += %(psp)s
        }

        __syncthreads();

        // do reduction in shared mem within one warp
        if (threadIdx.x < 16) {
            volatile %(float_prec)s* data = sdata[threadIdx.y];
            data[tid] += data[tid + 16];
            data[tid] += data[tid +  8];
            data[tid] += data[tid +  4];
            data[tid] += data[tid +  2];
            data[tid] += data[tid +  1];
        }

        if (threadIdx.x == 0) %(target_arg)s%(post_index)s += sdata[threadIdx.y][0];
        __syncthreads();
        rk_post += gridDim.y*blockDim.y;
    }
}
"""
    },
    'invoke_kernel': """
void proj%(id_proj)s_psp(RunConfig cfg, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s) {
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
    'kernel_decl': """void proj%(id_proj)s_psp(RunConfig cfg, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'host_call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        // 2D-Kernel: y number rows, x number columns
        auto num_block_rows = static_cast<%(idx_type)s>(ceil(double(proj%(id_proj)s.num_rows())/16.0));
        auto thread_dim = dim3(32, 16, 1);
        auto block_dim = dim3(1, num_block_rows, 1);

        proj%(id_proj)s_psp(
            RunConfig(block_dim, thread_dim, 0, proj%(id_proj)s.stream),
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
    'device_kernel': """// gpu device kernel for projection %(id_proj)s
__global__ void cu_proj%(id_proj)s_psp( const long int t, const %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_args_header)s %(kernel_args_header)s ) {
    int rk_post = blockIdx.x ;
    int tid = threadIdx.x;

    while (tid < num_events[0]) {
        int rk_pre = spiked[tid];               // which neuron spiked
        int j = rk_post * column_size + rk_pre; // row-major indexing

        // event-driven
    %(event_driven)s

        // increase of conductance
    %(psp)s

        // pre-spike statements
    %(pre_event)s

        // proceed to the next spiking neuron
        tid += blockDim.x;
    }
}
""",
    'invoke_kernel': """
void proj%(id_proj)s_psp(RunConfig cfg, const long int t, const %(float_prec)s dt, bool plasticity, int *spiked, unsigned int* num_events, %(conn_args_header)s %(kernel_args_header)s) {
    cu_proj%(id_proj)s_psp<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
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
    if ( pop%(id_pre)s._active && (pop%(id_pre)s.spike_count > 0) && proj%(id_proj)s._transmission ) {
        int tpb = 32;
        int nb = proj%(id_proj)s.num_rows();

        // compute psp
        proj%(id_proj)s_psp(
            RunConfig(nb, tpb, 0, proj%(id_proj)s.stream),
            t, dt, proj%(id_proj)s._plasticity,
            /* pre-synaptic events */
            pop%(id_pre)s.gpu_spiked, pop%(id_pre)s.gpu_spike_count,
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
}

# Update of global synaptic equations, consist of body (annarchyDevice.cu),
# header and call semantic (take place in ANNarchyHost.cu)
global_synapse_update = {
    'device_kernel': """
// gpu device kernel for projection %(id_proj)s
__global__ void cuProj%(id_proj)s_global_step(
    /* default params */
    const long int, const %(float_prec)s dt
    /* additional params */
    %(kernel_args)s,
    /* plasticity enabled */
    bool plasticity )
{
%(pre_loop)s
%(global_eqs)s
}
""",
    'invoke_kernel': """
void proj%(id_proj)s_global_step(RunConfig cfg, const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity) {
    cuProj%(id_proj)s_global_step<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(
        /* default args*/
        t, dt
        /* kernel args */
        %(kernel_args_call)s
        /* synaptic plasticity */
        , plasticity
    );
}
""",
    'kernel_decl': """void proj%(id_proj)s_global_step(RunConfig cfg, const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'host_call': """
        // global update
        proj%(id_proj)s_global_step(
            RunConfig(1, 1, 0, proj%(id_proj)s.stream),
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
    'device_kernel': """
// gpu device kernel for projection %(id_proj)s
__global__ void cuProj%(id_proj)s_semiglobal_step(
    /* default params */
    const int post_size, const long int t, const %(float_prec)s dt
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
    'invoke_kernel': """
void proj%(id_proj)s_semiglobal_step(RunConfig cfg, const %(idx_type)s post_size, const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity) {
    cuProj%(id_proj)s_semiglobal_step<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
        /* default args*/
        post_size, t, dt
        /* kernel args */
        %(kernel_args_call)s
        /* synaptic plasticity */
        , plasticity
    );
}
""",
    'kernel_decl': """void proj%(id_proj)s_semiglobal_step(RunConfig cfg, const %(idx_type)s post_size, const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'host_call': """
        // semiglobal update
        nb_blocks = ceil( %(float_prec)s(proj%(id_proj)s.num_rows()) / 32.0);
        proj%(id_proj)s_semiglobal_step(
            RunConfig(nb_blocks, 32, 0, proj%(id_proj)s.stream),
            /* default args*/
            proj%(id_proj)s.num_rows(), t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );

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
    %(idx_type)s rk_post = blockIdx.x;
    %(idx_type)s rk_pre = threadIdx.x;
    %(size_type)s j = rk_post*pre_size+rk_pre;

%(pre_loop)s
    // Updating local variables of projection %(id_proj)s
    while ( rk_pre < pre_size )
    {
        if (mask[j]) {
%(local_eqs)s
        }

        j += blockDim.x;
        rk_pre += blockDim.x;
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
        RunConfig proj%(id_proj)s_local_step_cfg = RunConfig(__proj%(id_proj)s_nb__, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream);
    #else
        RunConfig proj%(id_proj)s_local_step_cfg = RunConfig(proj%(id_proj)s.nb_dendrites(), 32, 0, proj%(id_proj)s.stream);
    #endif
        proj%(id_proj)s_local_step(
            proj%(id_proj)s_local_step_cfg,
            /* default args*/
            pop%(id_post)s.size, pop%(id_pre)s.size, proj%(id_proj)s.device_mask(), t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
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
    int rk_pre  = spiked[blockIdx.x];
    int j = rk_pre * pre_size + threadIdx.x;
    int j_end = (rk_pre+1)*pre_size;

    while ( j < j_end) {
        if (mask[j]) {
    // event-driven
%(event_driven)s

    // post-event
%(post_code)s
        }
        j += blockDim.x;
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
    if ( proj%(id_proj)s._transmission && pop%(id_post)s._active && (pop%(id_post)s.spike_count > 0) ) {
    #if defined (__proj%(id_proj)s_%(target)s_nb__)
        int tpb = __proj%(id_proj)s_%(target)s_tpb__;
    #else
        int tpb = 64;
    #endif

        proj%(id_proj)s_postevent(
            RunConfig(pop%(id_post)s.spike_count, tpb, 0, proj%(id_proj)s.stream),
            t, dt, proj%(id_proj)s._plasticity,
            /* post-spike and pre-spike time points */
            pop%(id_post)s.gpu_spiked, pop%(id_pre)s.gpu_last_spike,
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

conn_templates = {
    # connectivity representation
    'conn_header': "const %(idx_type)s post_size, const %(idx_type)s pre_size, const char* mask",
    'conn_call': "pop%(id_post)s.size, pop%(id_pre)s.size, proj%(id_proj)s.device_mask()",
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
    'rate_psp': rate_psp_kernel,
    'spike_transmission': {
        'event_driven': spike_event_transmission,
        'continuous': None
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
    'semiglobal_index': '[rk_post]',
    'global_index': '[0]',
    'pre_index': '[rk_pre]',
    'post_index': '[rk_post]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_'
}
