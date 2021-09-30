#===============================================================================
#
#     COO.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2020-21  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
init_launch_config = """
        // Generate the kernel launch configuration
        _threads_per_block = 192;
        unsigned int tmp_blocks = static_cast<unsigned int>(ceil(static_cast<double>(nb_synapses())/static_cast<double>(_threads_per_block)));
        _nb_blocks = static_cast<unsigned short int>(std::min<unsigned int>(65535, tmp_blocks));
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
    'body': {
        'sum': """
__global__ void cu_proj%(id_proj)s_psp_coo( int nb_synapses, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;

    while( j < nb_synapses ) {

        %(float_prec)s sum = %(psp)s
        atomicAdd(&(%(target_arg)s%(post_index)s), sum);

        j += blockDim.x * gridDim.x;
    }
}
"""
    },
    'header': """__global__ void cu_proj%(id)s_psp_coo( int nb_synapses, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        size_t nb_synapses = proj%(id_proj)s.nb_synapses();

    #if defined (__proj%(id_proj)s_%(target)s_nb__)
        cu_proj%(id_proj)s_psp_coo<<< __proj%(id_proj)s_%(target)s_nb__, __proj%(id_proj)s_%(target)s_tpb__ >>>(
                       nb_synapses,
                       /* connectivity */
                       %(conn_args)s
                       /* other variables */
                       %(add_args)s
                       /* result */
                       %(target_arg)s
        );
    #else
        cu_proj%(id_proj)s_psp_coo<<< proj%(id_proj)s._nb_blocks, proj%(id_proj)s._threads_per_block >>>(
                       nb_synapses,
                       /* connectivity */
                       %(conn_args)s
                       /* other variables */
                       %(add_args)s
                       /* result */
                       %(target_arg)s
        );
    #endif

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
    # launch
    'launch_config': init_launch_config,

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