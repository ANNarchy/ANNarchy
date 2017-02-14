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
projection_header = """#pragma once

#include "pop%(id_pre)s.hpp"
#include "pop%(id_post)s.hpp"
%(include_additional)s
%(include_profile)s

extern PopStruct%(id_pre)s pop%(id_pre)s;
extern PopStruct%(id_post)s pop%(id_post)s;
%(struct_additional)s

/////////////////////////////////////////////////////////////////////////////
// proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct%(id_proj)s{
    // Number of dendrites
    int size;

    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;

%(declare_connectivity_matrix)s
%(declare_inverse_connectivity_matrix)s
%(declare_delay)s
%(declare_event_driven)s
%(declare_rng)s
%(declare_parameters_variables)s
%(declare_additional)s
%(declare_profile)s

    // Method called to initialize the projection
    void init_projection() {
        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;

%(init_connectivity_matrix)s

        // Inverse the connectivity matrix if spiking neurons
        inverse_connectivity_matrix();

%(init_event_driven)s
%(init_parameters_variables)s
%(init_rng)s
%(init_additional)s
%(init_profile)s
    }

    // Spiking networks: inverse the connectivity matrix
    void inverse_connectivity_matrix() {
%(init_inverse_connectivity_matrix)s
    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
%(psp_prefix)s
%(psp_code)s
    }

    // Draws random numbers
    void update_rng() {
%(update_rng)s
    }

    // Updates synaptic variables
    void update_synapse() {
%(update_prefix)s
%(update_variables)s
    }

    // Post-synaptic events
    void post_event() {
%(post_event_prefix)s
%(post_event)s
    }

    // Accessors for default attributes
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }

    // Additional access methods
%(access_connectivity_matrix)s
%(access_parameters_variables)s
%(access_additional)s

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
### Structural plasticity
######################################
# All code templates needed for structural plasticity.
structural_plasticity = {
    'header_struct': {
        'header': """
    // Structural plasticity
    int dendrite_index(int post, int pre){
        int idx = 0;
        for(int i=0; i<pre_rank[post].size(); i++){
            if(pre_rank[post][i] == pre){
                idx = i;
                break;
            }
        }
        return idx;
    }
    void addSynapse(int post, int pre, double weight, int _delay=0%(extra_args)s){
        // Find where to put the synapse
        int idx = pre_rank[post].size();
        for(int i=0; i<pre_rank[post].size(); i++){
            if(pre_rank[post][i] > pre){
                idx = i;
                break;
            }
        }
        pre_rank[post].insert(pre_rank[post].begin() + idx, pre);
        w[post].insert(w[post].begin() + idx, weight);
        %(delay_code)s
%(add_code)s
%(spike_add)s
%(rd_add)s
    };
    void removeSynapse(int post, int idx){
        pre_rank[post].erase(pre_rank[post].begin() + idx);
        w[post].erase(w[post].begin() + idx);
%(remove_code)s
%(spike_remove)s
%(rd_remove)s
    };
""",
        'pruning': """
    // Pruning
    bool _pruning;
    int _pruning_period;
    long int _pruning_offset;
""",
        'creating': """
    // Creating
    bool _creating;
    int _creating_period;
    long int _creating_offset;
""",
        'spiking_addcode': """
        // Add the corresponding pair in inv_pre_rank
        int idx_post = 0;
        for(int i=0; i<post_rank.size(); i++){
            if(post_rank[i] == post){
                idx_post = i;
                break;
            }
        }
        inv_pre_rank[pre].push_back(std::pair<int, int>(idx_post, idx));
""",
        'spiking_removecode': """
        // Remove the corresponding pair in inv_pre_rank
        int pre = pre_rank[post][idx];
        for(int i=0; i<inv_pre_rank[pre].size(); i++){
            if(inv_pre_rank[pre][i].second == idx){
                inv_pre_rank[pre].erase(inv_pre_rank[pre].begin() + i);
                break;
            }
        }
"""
    },
    'pyx_struct': {
        'pruning':
"""
        # Pruning
        bool _pruning
        int _pruning_period
        long _pruning_offset
""",
        'creating':
"""
        # Creating
        bool _creating
        int _creating_period
        long _creating_offset
""",
        'func':
"""
        # Structural plasticity
        int dendrite_index(int post, int pre)
        void addSynapse(int post, int pre, double weight, int _delay%(extra_args)s)
        void removeSynapse(int post, int pre)
"""
    },
    'pyx_wrapper': {
        'pruning':
"""
    # Pruning
    def start_pruning(self, int period, long offset):
        proj%(id)s._pruning = True
        proj%(id)s._pruning_period = period
        proj%(id)s._pruning_offset = offset
    def stop_pruning(self):
        proj%(id)s._pruning = False
""",
        'creating':
"""
    # Creating
    def start_creating(self, int period, long offset):
        proj%(id)s._creating = True
        proj%(id)s._creating_period = period
        proj%(id)s._creating_offset = offset
    def stop_creating(self):
        proj%(id)s._creating = False
""",
        'func':
"""
    # Structural plasticity
    def add_synapse(self, int post_rank, int pre_rank, double weight, int delay%(extra_args)s):
        proj%(id)s.addSynapse(post_rank, pre_rank, weight, delay%(extra_values)s)
    def remove_synapse(self, int post_rank, int pre_rank):
        proj%(id)s.removeSynapse(post_rank, proj%(id)s.dendrite_index(post_rank, pre_rank))
"""
    }
}


######################################
### Rate-coded summation OMP
######################################
# Default LiL
lil_summation_operation = {
    'sum' : """
%(pre_copy)s
nb_post = post_rank.size();
%(omp_code)s
for(int i = 0; i < nb_post; i++) {
    sum = 0.0;
    for(int j = 0; j < pre_rank[i].size(); j++) {
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
}
""",
    'max': """
%(pre_copy)s
nb_post = post_rank.size();
%(omp_code)s
for(int i = 0; i < nb_post; i++){
    int j = 0;
    sum = %(psp)s ;
    for(int j = 1; j < pre_rank[i].size(); j++){
        if(%(psp)s > sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
}
""",
    'min': """
%(pre_copy)s
nb_post = post_rank.size();
%(omp_code)s
for(int i = 0; i < nb_post; i++){
    int j= 0;
    sum = %(psp)s ;
    for(int j = 1; j < pre_rank[i].size(); j++){
        if(%(psp)s < sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
}
""",
    'mean': """
%(pre_copy)s
nb_post = post_rank.size();
%(omp_code)s
for(int i = 0; i < nb_post; i++){
    sum = 0.0 ;
    for(int j = 0; j < pre_rank[i].size(); j++){
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum / (double)(pre_rank[i].size());
}
"""
}

# Dense matrix
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

######################################
### Spiking summation
######################################
spiking_summation_fixed_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){
    std::vector< std::pair<int, int> > inv_post;
    // Iterate over all incoming spikes
    for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++){
        rk_j = %(pre_array)s[_idx_j];
        inv_post = inv_pre_rank[rk_j];
        nb_post = inv_post.size();
        // Iterate over connected post neurons
        %(omp_code)s
        for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
            // Retrieve the correct indices
            i = inv_post[_idx_i].first;
            j = inv_post[_idx_i].second;
            %(event_driven)s
            %(g_target)s
            %(pre_event)s
        }
    }
} // active
"""

spiking_summation_fixed_delay_csr = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){

    // Iterate over all spiking neurons
    %(omp_code)s
    for( int _idx = 0; _idx < %(pre_array)s.size(); _idx++) {
        int _pre = %(pre_array)s[_idx];

        // Iterate over connected post neurons
        for(int syn = _row_ptr[_pre]; syn < _row_ptr[_pre + 1]; syn++) {
            %(event_driven)s
            %(g_target)s
            %(pre_event)s
        }
    }
} // active
"""

spiking_summation_fixed_delay_dense_matrix = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){

} // active
"""

spiking_summation_variable_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){
    // Iterate over all post neurons
    %(omp_code)s
    for (i=0; i<post_rank.size(); i++){
        for (j=0; j<pre_rank[i].size(); j++){
            for(int _idx_j = 0; _idx_j < pop%(id_pre)s._delayed_spike[delay[i][j]-1].size(); _idx_j++){
                if(pop%(id_pre)s._delayed_spike[delay[i][j]-1][_idx_j] == pre_rank[i][j]){
                    %(event_driven)s
                    %(g_target)s
                    %(pre_event)s
                    break;
                }
            }
        }
    }
} // active
"""

######################################
### Update synaptic variables
######################################
lil_update_variables = {
    'local': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L) ){
    %(global)s
    %(omp_code)s
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i];
    %(semiglobal)s
        for(int j = 0; j < pre_rank[i].size(); j++){
            rk_pre = pre_rank[i][j];
    %(local)s
        }
    }
}
""",
    'global': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(global)s
    %(omp_code)s
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i];
    %(semiglobal)s
    }
}
"""
}

dense_update_variables = {
    'local': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(global)s
    %(omp_code)s
    for(int i = 0; i < pop%(id_post)s.size; i++){
        rk_post = i;
    %(semiglobal)s
        for(int j = 0; j < pop%(id_pre)s.size; j++){
            rk_pre = j;
    %(local)s
        }
    }
}
""",
    'global': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(global)s
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
