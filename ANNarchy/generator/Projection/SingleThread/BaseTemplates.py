"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

projection_header = """/*
 *  ANNarchy-version: %(annarchy_version)s
 */
#pragma once

#include "ANNarchy.hpp"
#include "helper_functions.hpp"
%(sparse_matrix_include)s
%(include_additional)s
%(include_profile)s

extern PopStruct%(id_pre)s pop%(id_pre)s;
extern PopStruct%(id_post)s pop%(id_post)s;
extern %(float_prec)s dt;
extern long int t;
%(struct_additional)s
extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct%(id_proj)s : %(sparse_format)s {
    ProjStruct%(id_proj)s() : %(sparse_format)s(%(sparse_format_args)s) {
    }

%(connector_call)s

%(declare_connectivity_matrix)s
%(access_connectivity_matrix)s

    // Transmission and plasticity flags
    bool _transmission, _axon_transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;

%(declare_delays)s
%(declare_event_driven)s
%(declare_rng)s
%(declare_parameters_variables)s
%(declare_additional)s
%(declare_profile)s

    // Method called to allocate/initialize the variables
    bool init_attributes() {
%(init_parameters_variables)s
%(init_event_driven)s
%(init_rng)s

        return true;
    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::init_projection() - this = " << this << std::endl;
    #endif

        _transmission = true;
        _axon_transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

        init_attributes();

%(init_additional)s
%(init_profile)s
    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {
%(reset_ring_buffer)s
    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){
%(update_max_delay)s
    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct%(id_proj)s::compute_psp()" << std::endl;
    #endif
%(psp_prefix)s
%(psp_code)s
    }

    // Draws random numbers
    void update_rng() {
%(update_rng)s
    }

    // Updates synaptic variables
    void update_synapse() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct%(id_proj)s::update_synapse()" << std::endl;
    #endif
%(update_prefix)s
%(update_variables)s
    }

    // Post-synaptic events
    void post_event() {
%(post_event)s
    }

    // Variable/Parameter access methods
%(access_parameters_variables)s

    // Access additional
%(access_additional)s

    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;
%(size_in_bytes)s
        return size_in_bytes;
    }

    // Structural plasticity
%(creating)s
%(pruning)s

    void clear() {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::clear() - this = " << this << std::endl;
    #endif
%(clear_container)s
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
        std::cout << "ProjStruct%(id_proj)s::get_local_attribute_%(ctype_name)s(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<")" << std::endl;
    #endif
%(local_get3)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all_%(ctype_name)s(std::string name, std::vector<std::vector<%(ctype)s>> value) {
    #ifdef _DEBUG
        auto min_value = std::numeric_limits<%(ctype)s>::max();
        auto max_value = std::numeric_limits<%(ctype)s>::min();
        for (auto it = value.cbegin(); it != value.cend(); it++ ){
            auto loc_min = *std::min_element(it->cbegin(), it->cend());
            if (loc_min < min_value)
                min_value = loc_min;
            auto loc_max = *std::max_element(it->begin(), it->end());
            if (loc_max > max_value)
                max_value = loc_max;
        }
        std::cout << "ProjStruct%(id_proj)s::set_local_attribute_all_%(ctype_name)s(name = " << name << ", min(" << name << ")=" <<std::to_string(min_value) << ", max("<<name<<")="<<std::to_string(max_value)<< ")" << std::endl;
    #endif
%(local_set1)s
    }

    void set_local_attribute_row_%(ctype_name)s(std::string name, int rk_post, std::vector<%(ctype)s> value) {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::set_local_attribute_row_%(ctype_name)s(name = "<<name<<", rk_post = " << rk_post << ", min("<<name<<")="<<std::to_string(*std::min_element(value.begin(), value.end())) << ", max("<<name<<")="<<std::to_string(*std::max_element(value.begin(), value.end()))<< ")" << std::endl;
    #endif
%(local_set2)s
    }

    void set_local_attribute_%(ctype_name)s(std::string name, int rk_post, int rk_pre, %(ctype)s value) {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::set_local_attribute_%(ctype_name)s(name = "<<name<<", rk_post = "<<rk_post<<", rk_pre = "<<rk_pre<<", value = " << std::to_string(value) << ")" << std::endl;
    #endif
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

# Definition for the usage of C++11 STL template random
# number generators
#
# Parameters:
#
#    rd_name:
#    rd_update:
cpp_11_rng = {
    'local': {
        'decl': """    std::vector< std::vector< %(float_prec)s > > %(rd_name)s;
    %(template)s dist_%(rd_name)s;
    """,
        'init': """
        %(rd_name)s = std::vector<%(type)s>(size, 0.0);
        dist_%(rd_name)s = %(rd_init)s;
    """,
        'update': """
                %(rd_name)s[i] = dist_%(rd_name)s(rng);
    """
    },
    'global': {
        'decl': """    %(type)s %(rd_name)s;
    %(template)s dist_%(rd_name)s;
    """,
        'init': """
        %(rd_name)s = 0.0;
        dist_%(rd_name)s = %(rd_init)s;
    """,
        'update': """
            %(rd_name)s = dist_%(rd_name)s(rng);
    """
    }
}

single_thread_templates = {
    'projection_header': projection_header,
    'attr_acc': attribute_acc,
    'accessor_template': attribute_template,
    'rng': cpp_11_rng
}
