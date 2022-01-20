#===============================================================================
#
#     OpenMPTemplates.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
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
projection_header = """/*
 *  ANNarchy-version: %(annarchy_version)s
 */
#pragma once

#include "ANNarchy.h"
%(sparse_matrix_include)s
%(include_additional)s
%(include_profile)s

extern PopStruct%(id_pre)s pop%(id_pre)s;
extern PopStruct%(id_post)s pop%(id_post)s;
%(struct_additional)s
extern %(float_prec)s dt;
extern long int t;
extern int global_num_threads;
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
    bool _transmission, _plasticity, _update;
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
%(init_event_driven)s
%(init_parameters_variables)s
%(init_rng)s
        return true;
    }

    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct%(id_proj)s::init_projection()" << std::endl;
    #endif

        _transmission = true;
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
    void compute_psp(const int tid) {
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
    void update_synapse(const int tid) {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct%(id_proj)s::update_synapse()" << std::endl;
    #endif
%(update_prefix)s
%(update_variables)s
    }

    // Post-synaptic events
    void post_event(const int tid) {
%(post_event_prefix)s
%(post_event)s
    }

    // Variable/Parameter access methods
%(access_parameters_variables)s

    // Access additional
%(access_additional)s

    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;
%(determine_size)s
        return size_in_bytes;
    }

    // Structural plasticity
%(creating)s
%(pruning)s

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct%(id_proj)s::clear()" << std::endl;
    #endif
%(clear_container)s
    }
};
"""

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

######################################
### Dense Matrix templates
######################################
dense_summation_operation = {
    'sum' : """
%(pre_copy)s
%(omp_code)s
for(int i = 0; i < pop%(id_post)s.size; i++) {
    sum = 0.0;
    for(int j = 0; j < pop%(id_pre)s.size; j++) {
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s[i] += sum;
}
""",
    'max': """
%(pre_copy)s
%(omp_code)s
for(int i = 0; i < pop%(id_post)s.size; i++){
    int j = 0;
    sum = %(psp)s ;
    for(int j = 1; j < pop%(id_pre)s.size; j++){
        if(%(psp)s > sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s[i] += sum;
}
""",
    'min': """
%(pre_copy)s
%(omp_code)s
for(int i = 0; i < pop%(id_post)s.size; i++){
    int j= 0;
    sum = %(psp)s ;
    for(int j = 1; j < pop%(id_pre)s.size; j++){
        if(%(psp)s < sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s[i] += sum;
}
""",
    'mean': """
%(pre_copy)s
%(omp_code)s
for(int i = 0; i < pop%(id_post)s.size; i++){
    sum = 0.0 ;
    for(int j = 0; j < pop%(id_pre)s.size; j++){
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s[i] += sum / (double)(pop%(id_pre)s.size);
}
"""
}

spiking_summation_fixed_delay_dense_matrix = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){
    // TODO?
} // active
"""

dense_update_variables = {
    'local': """
// Check periodicity
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    // Global variables
    %(global)s
    // Local variables
    %(omp_code)s
    for(int i = 0; i < pop%(id_post)s.size; i++){
        rk_post = i; // dense: ranks are indices
        // Semi-global variables
    %(semiglobal)s
        for(int j = 0; j < pop%(id_pre)s.size; j++){
            rk_pre = j; // dense: ranks are indices
    %(local)s
        }
    }
}
""",
    'global': """
// Check periodicity
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    // Global variables
    %(global)s
    // Semi-global variables
    %(omp_code)s
    for(int i = 0; i < pop%(id_post)s.size; i++){
        rk_post = i;
    %(semiglobal)s
    }
}
"""
}

openmp_templates = {
    'projection_header': projection_header,
    'rng': cpp_11_rng
}
