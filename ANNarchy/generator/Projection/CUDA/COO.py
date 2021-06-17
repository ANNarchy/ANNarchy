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
attribute_decl = {
    'local': """
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'semiglobal': """
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'global': """
    // Global %(attr_type)s %(name)s
    %(type)s %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
"""
}

attribute_acc = {
    'local': """
    // Local %(attr_type)s %(name)s
    std::vector<std::vector< %(type)s > > get_%(name)s() { return get_variable_all<%(type)s>(%(name)s); }
    std::vector<%(type)s> get_dendrite_%(name)s(int rk) { return get_variable_row<%(type)s>(%(name)s, rk); }
    %(type)s get_synapse_%(name)s(int rk_post, int rk_pre) { return get_variable<%(type)s>(%(name)s, rk_post, rk_pre); }
    void set_%(name)s(std::vector<std::vector< %(type)s > >value) {
        update_variable_all(%(name)s, value);
        %(name)s_dirty = true; 
    }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) {
        update_variable_row(%(name)s, rk, value);
        %(name)s_dirty = true; 
    }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) {
        update_variable(%(name)s, rk_post, rk_pre, value);
        %(name)s_dirty = true; 
    }
""",
    'semiglobal': """
    // Semiglobal %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; %(name)s_dirty = true; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; %(name)s_dirty = true; }
""",
    'global': """
    // Global %(attr_type)s %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s( %(type)s value ) { %(name)s = value; %(name)s_dirty = true; }
"""
}

attribute_cpp_init = {
    'local': """
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(%(init)s);
        gpu_%(name)s = init_matrix_variable_gpu<%(type)s>(%(name)s);
        %(name)s_dirty = true;
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

attribute_host_to_device = {
    'local': """
        // %(name)s: local
        if ( %(name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), this->nb_synapses() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_dirty = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'semiglobal': """
        // %(name)s: semiglobal
        if ( %(name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, %(name)s.data(), this->nb_dendrites() * sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_dirty = false;
        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""",
    'global': """
        // %(name)s: global
        if ( %(name)s_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_%(name)s, &%(name)s, sizeof( %(type)s ), cudaMemcpyHostToDevice);
            %(name)s_dirty = false;
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
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, this->nb_synapses() * sizeof( %(type)s ), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_%(name)s = cudaGetLastError();
            if ( err_%(name)s != cudaSuccess )
                std::cout << "  error: " << cudaGetErrorString(err_%(name)s) << std::endl;
        #endif
""",
    'semiglobal': """
            // %(name)s: semiglobal
        #ifdef _DEBUG
            std::cout << "DtoH: %(name)s ( proj%(id)s )" << std::endl;
        #endif
            cudaMemcpy( %(name)s.data(), gpu_%(name)s, this->nb_dendrites() * sizeof(%(type)s), cudaMemcpyDeviceToHost);
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

rate_psp_kernel = {
    'body': {
        'sum': """
__global__ void cu_proj%(id_proj)s_psp( int nb_synapses, int* row_indices, int* column_indices, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s ) {
    int tid = threadIdx.x;
    int j = tid;

    while( j < nb_synapses ) {

        %(float_prec)s sum = %(psp)s
        atomicAdd(&(%(target_arg)s%(post_index)s), sum);

        j += blockDim.x;
    }
}
"""
    },
    'header': """__global__ void cu_proj%(id)s_psp( int nb_synapses, int* row_indices, int* column_indices, %(conn_args)s%(add_args)s, %(float_prec)s* %(target_arg)s );
""",
    'call': """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s
    if ( pop%(id_post)s._active && proj%(id_proj)s._transmission ) {
        
        cu_proj%(id_proj)s_psp<<< 1, __proj%(id_proj)s_%(target)s_tpb__>>>(
                       proj%(id_proj)s.nb_synapses(),
                       /* connectivity */
                       proj%(id_proj)s.gpu_row_indices(), proj%(id_proj)s.gpu_column_indices(),
                       /* default variables */
                       %(conn_args)s
                       /* other variables */
                       %(add_args)s
                       /* result */
                       %(target_arg)s );

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
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_acc': attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'host_to_device': attribute_host_to_device,
    'device_to_host': attribute_device_to_host,

    # operations
    'rate_psp': rate_psp_kernel,
}