#===============================================================================
#
#     Dense.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2021  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
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
additional_global_functions = ""

init_launch_config = """
        // Generate the kernel launch configuration
        _threads_per_block = 0;
        _nb_blocks = 0;

    #ifdef _DEBUG
        std::cout << "Kernel configuration is a fixed 2D kernel" << std::endl;
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
    'header': """__global__ void cu_proj%(id)s_psp(%(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        // 2D-Kernel: y number rows, x number columns
        auto num_block_rows = static_cast<%(idx_type)s>(ceil(double(proj%(id_proj)s.num_rows())/16.0));
        auto thread_dim = dim3(32, 16, 1);
        auto block_dim = dim3(1, num_block_rows, 1);

        cu_proj%(id_proj)s_psp<<< block_dim, thread_dim>>>(
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
    'header': """__global__ void cuProj%(id_proj)s_global_step( %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'call': """
        // global update
        cuProj%(id_proj)s_global_step<<< 1, 1, 0, proj%(id_proj)s.stream>>>(
            proj%(id_proj)s.nb_dendrites(),
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
    const %(idx_type)s post_size, const %(idx_type)s* __restrict__ rank_post, const long int t, const %(float_prec)s dt
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
    'header': """__global__ void cuProj%(id_proj)s_semiglobal_step(const %(idx_type)s post_size, const %(idx_type)s* __restrict__ rank_post, const long int t, %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'call': """
        // semiglobal update
        nb_blocks = ceil( %(float_prec)s(proj%(id_proj)s.nb_dendrites()) / 32.0);
        cuProj%(id_proj)s_semiglobal_step<<< nb_blocks, 32, 0, proj%(id_proj)s.stream >>>(
            proj%(id_proj)s.nb_dendrites(), proj%(id_proj)s.gpu_post_rank_,
            /* default args*/
            proj%(id_proj)s.gpu_post_rank, proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank, t, _dt
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
    'header': """__global__ void cuProj%(id_proj)s_local_step(%(idx_type)s post_size, %(idx_type)s pre_size, char* mask, const long int t, const %(float_prec)s dt %(kernel_args)s, bool plasticity);
""",
    'call': """
        // local update
    #if defined (__proj%(id_proj)s_%(target)s_tpb__)
        cuProj%(id_proj)s_local_step<<< __proj%(id_proj)s_nb__, __proj%(id_proj)s_%(target)s_tpb__, 0, proj%(id_proj)s.stream >>>(
            /* default args*/
            pop%(id_post)s.size, pop%(id_pre)s.size, proj%(id_proj)s.device_mask(), t, _dt
            /* kernel args */
            %(kernel_args_call)s
            /* synaptic plasticity */
            , proj%(id_proj)s._plasticity
        );
    #else
        cuProj%(id_proj)s_local_step<<< proj%(id_proj)s.nb_dendrites(), 32, 0, proj%(id_proj)s.stream >>>(
            /* default args*/
            pop%(id_post)s.size, pop%(id_pre)s.size, proj%(id_proj)s.device_mask(), t, _dt
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

conn_templates = {
    # connectivity representation
    'conn_header': "const %(idx_type)s post_size, const %(idx_type)s pre_size",
    'conn_call': "pop%(id_post)s.size, pop%(id_pre)s.size",

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

    #operations
    'rate_psp': rate_psp_kernel,
    'synapse_update': {
        'global': global_synapse_update,
        'semiglobal': semiglobal_synapse_update,
        'local': local_synapse_update,
        'call': synapse_update_call
    }
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
