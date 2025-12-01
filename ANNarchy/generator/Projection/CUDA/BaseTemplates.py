"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

projection_header = """/*
 *  ANNarchy-version: %(annarchy_version)s
 */
#pragma once

#include "ANNarchy.hpp"
%(sparse_matrix_include)s
%(include_additional)s
%(include_profile)s

extern std::vector<std::mt19937> rng;
extern unsigned long long global_seed;

extern PopStruct%(id_pre)s* pop%(id_pre)s;
extern PopStruct%(id_post)s* pop%(id_post)s;
%(struct_additional)s

/////////////////////////////////////////////////////////////////////////////
// proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
/////////////////////////////////////////////////////////////////////////////
extern struct ProjStruct%(id_proj)s *proj%(id_proj)s;
struct ProjStruct%(id_proj)s : %(sparse_format)s {
    ProjStruct%(id_proj)s() : %(sparse_format)s (%(sparse_format_args)s) {
        // HACK: the object constructor is now called by nanobind, need to update reference in C++ library
        proj%(id_proj)s = this;

    #ifdef _TRACE_INIT
        std::cout << "  ProjStruct%(id_proj)s - this = " << this << " has been allocated." << std::endl;
    #endif
    }

    // Launch configuration
    unsigned int _nb_blocks;
    unsigned int _threads_per_block;

%(connector_call)s

%(declare_connectivity_matrix)s
%(access_connectivity_matrix)s

    // Transmission and plasticity flags
    bool _transmission, _axon_transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;

%(declare_delay)s
%(declare_event_driven)s
%(declare_rng)s
%(declare_parameters_variables)s
%(declare_cuda_stream)s
%(declare_additional)s
%(declare_profile)s

    // Method called to allocate/initialize the variables
    bool init_attributes() {
%(init_event_driven)s
%(init_parameters_variables)s
%(init_rng)s

        return true;
    }

    // Generate the default kernel launch configuration
    void default_launch_config() {
%(init_launch_config)s
    }

    // Override the default kernel launch configuration
    void update_launch_config(int nb_blocks, int threads_per_block) {
%(update_launch_config)s
    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _TRACE_INIT
        std::cout << "  ProjStruct%(id_proj)s::init_projection(post_size = " << pop%(id_post)s->size << ", pre_size = " << pop%(id_pre)s->size << ")" << std::endl;
    #endif

        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

        default_launch_config();

        init_attributes();

%(init_additional)s
%(init_profile)s
    }

    // Additional access methods
%(access_parameters_variables)s
%(access_additional)s

    // Memory management
    size_t size_in_bytes() override final {
        long int size_in_bytes = 0;
%(size_in_bytes)s
        return size_in_bytes;
    }

    void clear() override final {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::clear(this = " << this << ")" << std::endl;
    #endif
%(clear_container)s
    }

    // Memory transfers
    void host_to_device() {
    #if defined(_TRACE_INIT) || defined(_DEBUG)
        std::cout << "  ProjStruct%(id_proj)s::host_to_device() called at t = " << t << " simulation steps." << std::endl;
    #endif    
%(host_to_device)s
    }

    void device_to_host(std::string attr_name) {
%(device_to_host)s
    }
};
"""

attribute_template = {
    "local": """
    std::vector<std::vector<%(ctype)s>> get_local_attribute_all_%(ctype_name)s(std::string name) {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::get_local_attribute_all_%(ctype_name)s(name = "<<name<<")" << std::endl;
    #endif
%(local_get1)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_local_attribute_all_%(ctype_name)s: " << name << " not found" << std::endl;
        return std::vector<std::vector<%(ctype)s>>();
    }

    std::vector<%(ctype)s> get_local_attribute_row_%(ctype_name)s(std::string name, int rk_post) {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::get_local_attribute_row_%(ctype_name)s(name = "<<name<<", rk_post = "<<rk_post<<")" << std::endl;
    #endif
%(local_get2)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_local_attribute_row_%(ctype_name)s: " << name << " not found" << std::endl;
        return std::vector<%(ctype)s>();
    }

    %(ctype)s get_local_attribute_%(ctype_name)s(std::string name, int rk_post, int rk_pre) {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::get_local_attribute_row_%(ctype_name)s(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<")" << std::endl;
    #endif
%(local_get3)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all_%(ctype_name)s(std::string name, std::vector<std::vector<%(ctype)s>> value) {
%(local_set1)s
    }

    void set_local_attribute_row_%(ctype_name)s(std::string name, int rk_post, std::vector<%(ctype)s> value) {
%(local_set2)s
    }

    void set_local_attribute_%(ctype_name)s(std::string name, int rk_post, int rk_pre, %(ctype)s value) {
%(local_set3)s
    }
""",
    "semiglobal": """
    std::vector<%(ctype)s> get_semiglobal_attribute_all_%(ctype_name)s(std::string name) {
%(semiglobal_get1)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_semiglobal_attribute_all_%(ctype_name)s: " << name << " not found" << std::endl;
        return std::vector<%(ctype)s>();
    }

    %(ctype)s get_semiglobal_attribute_%(ctype_name)s(std::string name, int rk_post) {
%(semiglobal_get2)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_semiglobal_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_semiglobal_attribute_all_%(ctype_name)s(std::string name, std::vector<%(ctype)s> value) {
%(semiglobal_set1)s
    }

    void set_semiglobal_attribute_%(ctype_name)s(std::string name, int rk_post, %(ctype)s value) {
%(semiglobal_set2)s
    }
""",
    "global": """
    %(ctype)s get_global_attribute_%(ctype_name)s(std::string name) {
%(global_get)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_global_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute_%(ctype_name)s(std::string name, %(ctype)s value) {
%(global_set)s
    }
"""
}

attribute_acc = {
    #
    # Local attributes
    #
    'local_get_all': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_matrix_variable_all<%(type)s>(%(name)s);
        }
""",
    'local_get_row': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_matrix_variable_row<%(type)s>(%(name)s, rk_post);
        }
""",
    'local_get_single': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_matrix_variable<%(type)s>(%(name)s, rk_post, rk_pre);
        }
""",
    'local_set_all': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            update_matrix_variable_all<%(type)s>(%(name)s, value);
            %(write_dirty_flag)s
            return;
        }
""",
    'local_set_row': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            update_matrix_variable_row<%(type)s>(%(name)s, rk_post, value);
            %(write_dirty_flag)s
            return;
        }
""",
    'local_set_single': """
        // Local %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            update_matrix_variable<%(type)s>(%(name)s, rk_post, rk_pre, value);
            %(write_dirty_flag)s
            return;
        }
""",
    #
    # Semiglobal attributes
    #
    'semiglobal_get_all': """
        // Semiglobal %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_vector_variable_all<%(type)s>(%(name)s);
        }
""",
    'semiglobal_get_single': """
        // Semiglobal %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_vector_variable<%(type)s>(%(name)s, rk_post);
        }
""",
    'semiglobal_set_all': """
        // Semiglobal %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            update_vector_variable_all<%(type)s>(%(name)s, value);
            %(write_dirty_flag)s
            return;
        }
""",
    'semiglobal_set_single': """
        // Semiglobal %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            update_vector_variable<%(type)s>(%(name)s, rk_post, value);
            %(write_dirty_flag)s
            return;
        }
""",
    #
    # Global attributes
    #
    'global_get': """
        // Global %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return %(name)s;
        }
""",
    'global_set': """
        // Global %(attr_type)s %(name)s
        if ( name.compare("%(name)s") == 0 ) {
            %(name)s = value;
            %(write_dirty_flag)s
            return;
        }
"""
}

# Definition for the usage of CUDA device random
# number generators
#
# Parameters:
#
#    rd_name:
#    rd_update:
curand = {
    'local': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, _nb_blocks * _threads_per_block * sizeof(curandState));
        init_curand_states( _nb_blocks, _threads_per_block, gpu_%(rd_name)s, global_seed );
    #ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "proj%(id)s - init_projection: " << cudaGetErrorString(err) << std::endl;
    #endif

"""
    },
    'global': {
        'decl': """
    curandState* gpu_%(rd_name)s;
""",
        'init': """
        cudaMalloc((void**)&gpu_%(rd_name)s, size * sizeof(curandState));
        init_curand_states( size, gpu_%(rd_name)s, global_seed );
    #ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "proj%(id)s - init_projection: " << cudaGetErrorString(err) << std::endl;
    #endif
"""
    }
}

cuda_stream = """
    // stream
    cudaStream_t stream;
"""

# some base stuff
cuda_templates = {
    'projection_header': projection_header,
    'attr_acc': attribute_acc,
    'accessor_template': attribute_template,
    'rng': curand
}
