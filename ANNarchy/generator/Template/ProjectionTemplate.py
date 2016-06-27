"""

    ProjectionTemplate.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

######################################
### Main template
######################################
header_struct = """#pragma once

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
%(declare_cuda_stream)s
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

%(cuda_flattening)s
};
"""


######################################
### Connectivity matrix OMP
######################################
lil_connectivity_matrix_omp = {
    'declare': """
    // Connectivity
    std::vector<int> post_rank;
    std::vector< std::vector< int > > pre_rank;
""",
    'accessor': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }
""",
    'init': """
""",
    'pyx_struct': """
        # LIL Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        void inverse_connectivity_matrix()
""",
    'pyx_wrapper_args': "synapses",
    'pyx_wrapper_init': """
        cdef CSR syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()
        proj%(id_proj)s.set_size( size )
        proj%(id_proj)s.set_post_rank( syn.post_rank )
        proj%(id_proj)s.set_pre_rank( syn.pre_rank )
""",
    'pyx_wrapper_accessor': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def set_post_rank(self, val):
        proj%(id_proj)s.set_post_rank(val)
        proj%(id_proj)s.inverse_connectivity_matrix()
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj%(id_proj)s.get_pre_rank()
    def set_pre_rank(self, val):
        proj%(id_proj)s.set_pre_rank(val)
        proj%(id_proj)s.inverse_connectivity_matrix()
"""
}

lil_weight_matrix_omp = {
    'declare': """
    // LIL weights
    std::vector< std::vector< double > > w;
""",
    'accessor': """
    // Local parameter w
    std::vector<std::vector< double > > get_w() { return w; }
    std::vector<double> get_dendrite_w(int rk) { return w[rk]; }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) { w = value; }
    void set_dendrite_w(int rk, std::vector<double> value) { w[rk] = value; }
    void set_synapse_w(int rk_post, int rk_pre, double value) { w[rk_post][rk_pre] = value; }
""",
    'init': """
""",
    'pyx_struct': """
        # Local variable w
        vector[vector[double]] get_w()
        vector[double] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[vector[double]])
        void set_dendrite_w(int, vector[double])
        void set_synapse_w(int, int, double)
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        proj%(id_proj)s.set_w(syn.w)
""",
    'pyx_wrapper_accessor': """
    # Local variable w
    def get_w(self):
        return proj%(id_proj)s.get_w()
    def set_w(self, value):
        proj%(id_proj)s.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, vector[double] value):
        proj%(id_proj)s.set_dendrite_w(rank, value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj%(id_proj)s.set_synapse_w(rank_post, rank_pre, value)
"""
}

single_weight_matrix_omp = {
    'declare': """
    // Single weight in the projection
    double w;
""",
    'accessor': "",
    'init': "",
    'pyx_struct': """
        # Local variable w
        double w
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        # Use only the first weight
        proj%(id_proj)s.w = syn.w[0][0]
""",
    'pyx_wrapper_accessor': """
    # Local variable w
    def get_w(self):
        return proj%(id_proj)s.w
    def set_w(self, value):
        proj%(id_proj)s.w = value
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.w
    def set_dendrite_w(self, int rank, double value):
        proj%(id_proj)s.w = value
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.w
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj%(id_proj)s.w = value
"""
}


inverse_connectivity_matrix = {
    'declare': """
    std::map< int, std::vector< std::pair<int, int> > > inv_pre_rank ;
    std::vector< int > inv_post_rank ;
""",
    'init': """
        inv_pre_rank =  std::map< int, std::vector< std::pair<int, int> > > ();
        for(int i=0; i<pre_rank.size(); i++){
            for(int j=0; j<pre_rank[i].size(); j++){
                inv_pre_rank[pre_rank[i][j]].push_back(std::pair<int, int>(i,j));
            }
        }
        inv_post_rank =  std::vector< int > (pop%(id_post)s.size, -1);
        for(int i=0; i<post_rank.size(); i++){
            inv_post_rank[post_rank[i]] = i;
        }
"""
}


######################################
### Connectivity matrix CUDA
######################################
csr_connectivity_matrix_cuda = {
    'declare': """
    // Connectivity
    std::vector<int> post_rank ;
    int* gpu_post_rank;
    std::vector< std::vector< int > > pre_rank ;
    int* gpu_pre_rank;
    int* gpu_nb_synapses;
    int* gpu_off_synapses;

    // flat connectivity parameters
    int overallSynapses;
    std::vector<int> flat_idx;
    std::vector<int> flat_off;
""",
   'accessor': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }
""",
   'init': """
        // post_rank
        cudaMalloc((void**)&gpu_post_rank, post_rank.size() * sizeof(int));
        cudaMemcpy(gpu_post_rank, post_rank.data(), post_rank.size() * sizeof(int), cudaMemcpyHostToDevice);

        // nb_synapses
        flat_idx = flattenIdx<int>(pre_rank);
        cudaMalloc((void**)&gpu_nb_synapses, flat_idx.size() * sizeof(int));
        cudaMemcpy(gpu_nb_synapses, flat_idx.data(), flat_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
        overallSynapses = 0;
        std::vector<int>::iterator it;
        for ( it = flat_idx.begin(); it != flat_idx.end(); it++)
            overallSynapses += *it;

        // off_synapses
        flat_off = flattenOff<int>(pre_rank);
        cudaMalloc((void**)&gpu_off_synapses, flat_off.size() * sizeof(int));
        cudaMemcpy(gpu_off_synapses, flat_off.data(), flat_off.size() * sizeof(int), cudaMemcpyHostToDevice);

        // pre_rank
        std::vector<int> flat_pre_rank = flattenArray<int>(pre_rank);
        cudaMalloc((void**)&gpu_pre_rank, flat_pre_rank.size() * sizeof(int));
        cudaMemcpy(gpu_pre_rank, flat_pre_rank.data(), flat_pre_rank.size() * sizeof(int), cudaMemcpyHostToDevice);
        flat_pre_rank.clear();
""",
    'pyx_struct': """
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        # void inverse_connectivity_matrix()
""",
    'pyx_wrapper_init': """
        proj%(id_proj)s.set_post_rank( syn.post_rank )
        proj%(id_proj)s.set_pre_rank( syn.pre_rank )
""",
    'pyx_wrapper_accessor': """
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def set_post_rank(self, val):
        proj%(id_proj)s.set_post_rank(val)
        # proj%(id_proj)s.inverse_connectivity_matrix() # TODO: spike only
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj%(id_proj)s.get_pre_rank()
    def set_pre_rank(self, val):
        proj%(id_proj)s.set_pre_rank(val)
        # proj%(id_proj)s.inverse_connectivity_matrix() ' TODO: spike only'
""",
    'pyx_wrapper_args': " syn",
}

csr_weight_matrix_cuda = {
    'declare': """
    // Local variable w
    std::vector<std::vector<double> > w;
    double *gpu_w;
    bool w_dirty;
    """,
    'accessor': """
    // Local variable w
    std::vector<std::vector< double > > get_w() { return w; }
    std::vector<double> get_dendrite_w(int rk) { return w[rk]; }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) { w = value; w_dirty = true; }
    void set_dendrite_w(int rk, std::vector<double> value) { w[rk] = value; w_dirty = true; }
    void set_synapse_w(int rk_post, int rk_pre, double value) { w[rk_post][rk_pre] = value; w_dirty = true; }
    """,
    'init': """
        // weights
        cudaMalloc((void**)&gpu_w, overallSynapses * sizeof(double));
        w_dirty = true; // enforce update
        cudaError_t err_w = cudaGetLastError();
        if ( err_w != cudaSuccess )
            std::cout << cudaGetErrorString(err_w) << std::endl;
""",
    'pyx_struct': """
        vector[ vector[ double] ] get_w()
        vector[ double ] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[ vector[ double] ])
        void set_dendrite_w( int, vector[double])
        void set_synapse_w(int, int, double)
    """,
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        proj%(id_proj)s.set_w(syn.w)
    """,
    'pyx_wrapper_accessor': """
    # Local variable w
    def get_w(self):
        return proj%(id_proj)s.get_w()
    def set_w(self, value):
        proj%(id_proj)s.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, vector[double] value):
        proj%(id_proj)s.set_dendrite_w(rank, value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj%(id_proj)s.set_synapse_w(rank_post, rank_pre, value)
"""
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
spiking_summation_variable_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){
    // Iterate over all post neurons
    %(omp_code)s
    for (i=0; i<post_rank.size(); i++){
        for (j=0; j<pre_rank[i].size(); j++){
            for(int _idx_j = 0; _idx_j < pop%(id_pre)s._delayed_spike[delay[i][j]-1].size(); _idx_j++){
                if(pop0._delayed_spike[delay[i][j]-1][_idx_j] == pre_rank[i][j]){
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
    %(omp_code)s
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i];
    %(global)s
        for(int j = 0; j < pre_rank[i].size(); j++){
            rk_pre = pre_rank[i][j];
    %(local)s
        }
    }
}
""",
    'global': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(omp_code)s
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i];
    %(global)s
    }
}
"""
}

dense_update_variables = {
    'local': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(omp_code)s
    for(int i = 0; i < pop%(id_post)s.size; i++){
        rk_post = i;
    %(global)s
        for(int j = 0; j < pop%(id_pre)s.size; j++){
            rk_pre = j;
    %(local)s
        }
    }
}
""",
    'global': """
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L)){
    %(omp_code)s
    for(int i = 0; i < pop%(id_post)s.size; i++){
        rk_post = i;
    %(global)s
    }
}
"""
}

######################################
### Delay
######################################
delay = {
    'header_struct': """
    // Non-uniform delay
    std::vector< std::vector< int > > delay ;""",
    'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] delay""",
    'pyx_wrapper_init':
"""
        proj%(id_proj)s.delay = syn.delay""",
    'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay[idx]
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
"""
}


######################################
### Event-driven
######################################
event_driven = {
    'header_struct': """
    std::vector<std::vector<long> > _last_event;
""",
    'cpp_init': """
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
    'pyx_wrapper_init':
"""
        proj%(id_proj)s._last_event = vector[vector[long]](nb_post, vector[long]())
        for n in range(nb_post):
            proj%(id_proj)s._last_event[n] = vector[long](proj%(id_proj)s.nb_synapses(n), -10000)
"""
}


######################################
### Attributes
######################################
# c like definition of synaptic attributes, whereas 'local' is used if values can vary across
# synapses, consequently 'global' is used if values are common to all synapses within a dendrite.
# Please note, that one projection represents a collection of dendrites.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
attribute_decl = {
    'openmp': {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector<%(type)s > > %(name)s;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
"""
    },
    'cuda': {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector<%(type)s > > %(name)s;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
    %(type)s* gpu_%(name)s;
    bool %(name)s_dirty;
"""
    }
}

# c like definition of accessors for synaptic attributes, whereas 'local' is used if values can vary
# across synapses, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
attribute_acc = {
    'openmp': {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector<std::vector< %(type)s > > get_%(name)s() { return %(name)s; }
    std::vector<%(type)s> get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    %(type)s get_synapse_%(name)s(int rk_post, int rk_pre) { return %(name)s[rk_post][rk_pre]; }
    void set_%(name)s(std::vector<std::vector< %(type)s > >value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) { %(name)s[rk] = value; }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) { %(name)s[rk_post][rk_pre] = value; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; }
"""
    },
    'cuda': {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector<std::vector< %(type)s > > get_%(name)s() { return %(name)s; }
    std::vector<%(type)s> get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    %(type)s get_synapse_%(name)s(int rk_post, int rk_pre) { return %(name)s[rk_post][rk_pre]; }
    void set_%(name)s(std::vector<std::vector< %(type)s > >value) { %(name)s = value; %(name)s_dirty = true; }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) { %(name)s[rk] = value; %(name)s_dirty = true; }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) { %(name)s[rk_post][rk_pre] = value; %(name)s_dirty = true; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; }
"""
    }
}

# Initialization of parameters due to the init_projection method.
#
# Parameters:
#
#    name: name of the variable
#    init: initial value
attribute_cpp_init = {
    'openmp':
    {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector< std::vector<%(type)s> >(post_rank.size(), std::vector<%(type)s>());
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(post_rank.size(), %(init)s);
"""
    },
    'cuda':
    {
    'local':
"""
        // Local %(attr_type)s %(name)s
        cudaMalloc((void**)&gpu_%(name)s, overallSynapses * sizeof(%(type)s));
        %(name)s_dirty = true;
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        cudaMalloc((void**)&gpu_%(name)s, post_rank.size() * sizeof(%(type)s));
        %(name)s_dirty = true;
"""
    }
}

# export of accessors for synaptic attributes towards python, whereas 'local' is used if values can vary
# across synapses within a dendrite, consequently 'global' is used if values are common to all synapses within
# a single dendrite.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...). One should check if cython can understand the
#          used types ( e. g. vector[bool] would not work properly... )
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
attribute_cpp_export = {
    'local':
"""
        # Local %(attr_type)s %(name)s
        vector[vector[%(type)s]] get_%(name)s()
        vector[%(type)s] get_dendrite_%(name)s(int)
        %(type)s get_synapse_%(name)s(int, int)
        void set_%(name)s(vector[vector[%(type)s]])
        void set_dendrite_%(name)s(int, vector[%(type)s])
        void set_synapse_%(name)s(int, int, %(type)s)
""",
    'global':
"""
        # Global %(attr_type)s %(name)s
        vector[%(type)s] get_%(name)s()
        %(type)s get_dendrite_%(name)s(int)
        void set_%(name)s(vector[%(type)s])
        void set_dendrite_%(name)s(int, %(type)s)
"""
}

attribute_pyx_wrapper = {
    'local':
"""
    # Local %(attr_type)s %(name)s
    def get_%(name)s(self):
        return proj%(id)s.get_%(name)s()
    def set_%(name)s(self, value):
        proj%(id)s.set_%(name)s( value )
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.get_dendrite_%(name)s(rank)
    def set_dendrite_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.set_dendrite_%(name)s(rank, value)
    def get_synapse_%(name)s(self, int rank_post, int rank_pre):
        return proj%(id)s.get_synapse_%(name)s(rank_post, rank_pre)
    def set_synapse_%(name)s(self, int rank_post, int rank_pre, %(type)s value):
        proj%(id)s.set_synapse_%(name)s(rank_post, rank_pre, value)
""",
    'global':
"""
    # Global %(attr_type)s %(name)s
    def get_%(name)s(self):
        return proj%(id)s.get_%(name)s()
    def set_%(name)s(self, value):
        proj%(id)s.set_%(name)s(value)
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.get_dendrite_%(name)s(rank)
    def set_dendrite_%(name)s(self, int rank, %(type)s value):
        proj%(id)s.set_dendrite_%(name)s(rank, value)
"""
}


######################################
### Monitoring
######################################
# Code templates for recorder class.
#
#    struct: base template
#    local: codes for struct member, init and recording for local variables
#    global: codes for struct member, init and recording for global variables
record = {
    'struct': """
class ProjRecorder%(id)s : public Monitor
{
public:
    ProjRecorder%(id)s(std::vector<int> ranks, int period, long int offset)
        : Monitor(ranks, period, offset)
    {
%(init_code)s
    };
    virtual void record() {
%(recording_code)s
    };
%(struct_code)s
};
""",
    'local': {
        'struct': """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ;
""",
        'init' : """
        this->%(name)s = std::vector< std::vector< %(type)s > >();
        this->record_%(name)s = false;
""",
        'recording': """
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            this->%(name)s.push_back(proj%(id)s.%(name)s[this->ranks[0]]);
        }
"""
    },
    'global': {
        'struct': """
    // Global variable %(name)s
    std::vector< %(type)s > %(name)s ;
    bool record_%(name)s ;
""",
        'init' : """
        this->%(name)s = std::vector< %(type)s >();
        this->record_%(name)s = false;
""",
        'recording': """
        if(this->record_%(name)s && ( (t - this->offset) %% this->period == 0 )){
            this->%(name)s.push_back(proj%(id)s.%(name)s[this->ranks[0]]);
        }
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
### CUDA stuff
######################################
cuda_stream = """
    // stream
    cudaStream_t stream;
"""
#
# Set of functions for convertion of LIL in CSR
cuda_flattening = """
    /*
     * (De-)Flattening of LIL structures
     */
    template<typename T>
    std::vector<int> flattenIdx(std::vector<std::vector<T> > in)
    {
        std::vector<T> flatIdx = std::vector<T>();
        typename std::vector<std::vector<T> >::iterator it;

        for ( it = in.begin(); it != in.end(); it++)
        {
            flatIdx.push_back(it->size());
        }

        return flatIdx;
    }

    template<typename T>
    std::vector<int> flattenOff(std::vector<std::vector<T> > in)
    {
        std::vector<T> flatOff = std::vector<T>();
        typename std::vector<std::vector<T> >::iterator it;

        int currOffset = 0;
        for ( it = in.begin(); it != in.end(); it++)
        {
            flatOff.push_back(currOffset);
            currOffset += it->size();
        }

        return flatOff;
    }

    template<typename T>
    std::vector<T> flattenArray(std::vector<std::vector<T> > in)
    {
        std::vector<T> flatVec = std::vector<T>();
        typename std::vector<std::vector<T> >::iterator it;

        for ( it = in.begin(); it != in.end(); it++)
        {
            flatVec.insert(flatVec.end(), it->begin(), it->end());
        }

        return flatVec;
    }

    template<typename T>
    std::vector<std::vector<T> > deFlattenArray(std::vector<T> in, std::vector<int> idx)
    {
        std::vector<std::vector<T> > deFlatVec = std::vector<std::vector<T> >();
        std::vector<int>::iterator it;

        int t=0;
        for ( it = idx.begin(); it != idx.end(); it++)
        {
            std::vector<T> tmp = std::vector<T>(in.begin()+t, in.begin()+t+*it);
            t += *it;

            deFlatVec.push_back(tmp);
        }

        return deFlatVec;
    }
"""



######################################
### Summation CUDA
######################################
# Comment to if (tid < 32) block:
#
# now that we are using warp-synchronous programming (below)
# we need to declare our shared memory volatile so that the compiler
# doesn't reorder stores to it and induce incorrect behavior.
cuda_psp_kernel=\
"""
__global__ void cu_proj%(id)s_psp( int* rank_pre, int *nb_synapses, int* offsets, double *pre_r, double* w, double *sum_%(target)s ) {
    unsigned int tid = threadIdx.x;
    unsigned int j = tid+offsets[blockIdx.x];

    extern double __shared__ sdata[];
    double localSum = 0.0;

    while(j < nb_synapses[blockIdx.x]+offsets[blockIdx.x])
    {
        localSum += %(psp)s

        j+= blockDim.x;
    }

    sdata[tid] = localSum;
    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = localSum = localSum + sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = localSum = localSum + sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = localSum = localSum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        volatile double* smem = sdata;

        if (blockDim.x >=  64) { smem[tid] = localSum = localSum + smem[tid + 32]; }
        if (blockDim.x >=  32) { smem[tid] = localSum = localSum + smem[tid + 16]; }
        if (blockDim.x >=  16) { smem[tid] = localSum = localSum + smem[tid +  8]; }
        if (blockDim.x >=   8) { smem[tid] = localSum = localSum + smem[tid +  4]; }
        if (blockDim.x >=   4) { smem[tid] = localSum = localSum + smem[tid +  2]; }
        if (blockDim.x >=   2) { smem[tid] = localSum = localSum + smem[tid +  1]; }

    }

    // write result for this block to global mem
    if (tid == 0)
    {
        sum_%(target)s[blockIdx.x] = sdata[0];
    }

}
"""

cuda_psp_kernel_call =\
"""
    // proj%(id)s: pop%(pre)s -> pop%(post)s
    if ( pop%(post)s._active ) {
        int sharedMemSize = __pop%(pre)s_pop%(post)s_%(target)s__ * 64;

        cu_proj%(id)s_psp<<<pop%(post)s.size, __pop%(pre)s_pop%(post)s_%(target)s__, sharedMemSize>>>(
                       /* ranks and offsets */
                       proj%(id)s.gpu_pre_rank, proj%(id)s.gpu_nb_synapses, proj%(id)s.gpu_off_synapses,
                       /* computation data */
                       pop%(pre)s.gpu_r, proj%(id)s.gpu_w,
                       /* result */
                       pop%(post)s.gpu_sum_%(target)s );
    }
"""

######################################
### Update synaptic variables CUDA
######################################
cuda_synapse_kernel=\
"""
// gpu device kernel for projection %(id)s
__global__ void cuProj%(id)s_step( /* default params */
                              int *post_rank, int *pre_rank, int* nb_synapses, int* offsets, double dt
                              /* additional params */
                              %(var)s%(par)s,
                              /* plasticity enabled */
                              bool plasticity )
{
    int i = blockIdx.x;
    int j = offsets[i] + threadIdx.x;
    int C = offsets[i]+ nb_synapses[i];
    int rk_post = post_rank[i];

    // Updating global variables of projection %(id)s
    if ( threadIdx.x == 0)
    {
%(global_eqs)s
    }

    // Updating local variables of projection %(id)s
    while ( j < C )
    {
        int rk_pre = pre_rank[j];

%(local_eqs)s

        j += blockDim.x;
    }
}
"""

cuda_synapse_kernel_call =\
"""
    // proj%(id_proj)s: pop%(pre)s -> pop%(post)s
    if ( proj%(id_proj)s._transmission && proj%(id_proj)s._update && proj%(id_proj)s._plasticity ) {
        cuProj%(id_proj)s_step<<< pop%(post)s.size, __pop%(pre)s_pop%(post)s_%(target)s__, 0, proj%(id_proj)s.stream>>>(
            proj%(id_proj)s.gpu_post_rank,
            proj%(id_proj)s.gpu_pre_rank,
            proj%(id_proj)s.gpu_nb_synapses,
            proj%(id_proj)s.gpu_off_synapses,
            dt
            %(local)s
            %(global)s
            , proj%(id_proj)s._plasticity
        );

    #ifdef _DEBUG
        cudaError_t proj%(id_proj)s_step = cudaGetLastError();
        if (proj%(id_proj)s_step != cudaSuccess) {
            std::cout << "proj%(id_proj)s_step: " << cudaGetErrorString(proj%(id_proj)s_step) << std::endl;
        }
    #endif
    }
"""
