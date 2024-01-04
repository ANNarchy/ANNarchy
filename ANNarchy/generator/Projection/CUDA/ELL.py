"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Code which should be added prior to kernels
additional_global_functions = ""

launch_config = {
    'init': """
        _threads_per_block = 32;
        auto tmp_blocks = static_cast<unsigned int>(ceil(static_cast<double>(nb_dendrites())/static_cast<double>(_threads_per_block)));
        _nb_blocks = std::min<unsigned int>(tmp_blocks, 65535);

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
            auto tmp_blocks = static_cast<unsigned int>(ceil(static_cast<double>(nb_dendrites())/static_cast<double>(_threads_per_block)));
            _nb_blocks = std::min<unsigned int>(tmp_blocks, 65535);
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
__global__ void cu_proj%(id_proj)s_psp_ell(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    while ( i < post_size ) {
        %(idx_type)s rk_post = rank_post[i];
        %(float_prec)s localSum = %(thread_init)s;

        for (%(size_type)s j =0; j < maxnzr; j++) {
            %(idx_type)s rk_pre = rank_pre[j*post_size+i];
            if (rk_pre == zero_marker)
                break;

            localSum += %(psp)s
        }

        %(target_arg)s%(post_index)s += localSum;

        i += gridDim.x * blockDim.x;
    }
}
"""        
    },
    'invoke_kernel': """
void proj%(id_proj)s_psp(RunConfig cfg, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s) {
    cu_proj%(id_proj)s_psp_ell<<< cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream >>>(
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

        proj%(id_proj)s_psp(
            /* kernel config */
            RunConfig(proj%(id_proj)s._nb_blocks, proj%(id_proj)s._threads_per_block,0, proj%(id_proj)s.stream),
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

# Update of global synaptic equations, consist of body (annarchyDevice.cu),
# header and call semantic (take place in ANNarchyHost.cu)
global_synapse_update = {
    'body': """
// gpu device kernel for projection %(id_proj)s
__global__ void cuProj%(id_proj)s_global_step(
    /* default params */
    const long int t, %(float_prec)s dt
    /* additional params */
    %(kernel_args)s,
    /* plasticity enabled */
    bool plasticity 
) {
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
    /* default params */
    %(idx_type)s post_size, const %(idx_type)s* __restrict__ rank_post, const long int t, %(float_prec)s dt
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
        cuProj%(id_proj)s_semiglobal_step<<< proj%(id_proj)s._nb_blocks, proj%(id_proj)s._threads_per_block, 0, proj%(id_proj)s.stream >>>(
            proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.gpu_post_ranks_,
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
    const long int t, %(float_prec)s dt
    /* additional params */
    %(kernel_args)s,
    /* plasticity enabled */
    bool plasticity 
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
%(pre_loop)s

    // Updating local variables of projection %(id_proj)s
    while ( i < post_size )
    {
        for ( int j = 0; j < maxnzr; j++ ) {
            if (rank_pre[j*post_size+i] == zero_marker)
                break;

%(local_eqs)s
        }

        i += blockDim.x * gridDim.x;
    }
}
""",
        'header': """__global__ void cuProj%(id_proj)s_local_step(%(conn_args)s, const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
        'call': """
        // local update
        cuProj%(id_proj)s_local_step<<< 1, 32, 0, proj%(id_proj)s.stream >>>(
            /* default args*/
            %(conn_args_call)s
            /* default args*/
            , t, _dt
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

conn_templates = {
    # connectivity representation
    'conn_header': "const %(idx_type)s post_size, const %(idx_type)s* __restrict__ rank_post, const %(idx_type)s* __restrict__ rank_pre, const %(idx_type)s maxnzr, const %(idx_type)s zero_marker",
    'conn_call': "proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.gpu_post_ranks_, proj%(id_proj)s.gpu_col_idx_, proj%(id_proj)s.get_maxnzr(), std::numeric_limits<%(idx_type)s>::max()",
    'conn_kernel': "post_size, rank_post, rank_pre, maxnzr, zero_marker",

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
    'rate_psp': rate_psp_kernel,
    'synapse_update': {
        'global': global_synapse_update,
        'semiglobal': semiglobal_synapse_update,
        'local': local_synapse_update,
        'call': synapse_update_call
    }
}

conn_ids = {
    'local_index': "[j*post_size+i]",
    'semiglobal_index': '[i]',
    'global_index': '[0]',
    'pre_index': '[rk_pre]',
    'post_index': '[rk_post]',
    'pre_prefix': 'pre_',
    'post_prefix': 'post_'
}