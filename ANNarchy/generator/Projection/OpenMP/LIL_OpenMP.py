#===============================================================================
#
#     LIL_OpenmMP.py
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
attribute_single_matrix_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector< %(type)s > > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s ;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s  %(name)s ;
"""
}

attribute_sliced_matrix_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector< std::vector<%(type)s > > > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s  %(name)s ;
"""
}

attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = init_matrix_variable<%(type)s>(%(init)s);
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = init_vector_variable<%(type)s>(%(init)s);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
"""
}

cpp_11_rng = {
    'template': """#pragma omp single
{
    %(global_rng)s
    for(int i = 0; i < post_rank.size(); i++) {
    %(semiglobal_rng)s
        for(int j = 0; j < pre_rank[i].size(); j++) {
    %(local_rng)s
        }
    }
}
""",
    'global': """
%(rd_name)s = dist_%(rd_name)s(rng[0]);
""",
    'semiglobal': """
    %(rd_name)s[i] = dist_%(rd_name)s(rng[0]);
""",
    'local': """
        %(rd_name)s[i][j] = dist_%(rd_name)s(rng[0]);
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
"""},
    'nonuniform_rate_coded': {
        'declare': """
    std::vector<std::vector<int>> delay;
    int max_delay;
""",
        'init': """
    delay = init_matrix_variable<int>(1);
    update_matrix_variable_all<int>(delay, delays);

    max_delay = pop%(id_pre)s.max_delay;
""",
        'reset': "",
        'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()
""",
        'pyx_wrapper_init': "",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.delay
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.delay[idx]
    def set_delay(self, value):
        proj%(id_proj)s.delay = value
    def get_max_delay(self):
        return proj%(id_proj)s.max_delay
    def set_max_delay(self, value):
        proj%(id_proj)s.max_delay = value
    def update_max_delay(self, value):
        proj%(id_proj)s.update_max_delay(value)
    def reset_ring_buffer(self):
        proj%(id_proj)s.reset_ring_buffer()
"""
    },
    'nonuniform_spiking': {
        'declare': """
    // Nonuniform spiking delays
    std::vector<std::vector<std::vector<int>>> delay;
    int max_delay;
    int idx_delay;
    std::vector< std::vector< std::vector< std::vector< int > > > > _delayed_spikes;

    std::vector<std::vector<int>> get_delay() {
        return get_matrix_variable_all<int>(delay);
    }
    void set_delay(std::vector<std::vector<int>> value) {
        update_matrix_variable_all<int>(delay, value);
    }
""",
        'init': """
        delay = init_matrix_variable<int>(1);
        update_matrix_variable_all<int>(delay, delays);

        idx_delay = 0;
        max_delay = pop%(id_pre)s.max_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< std::vector< int > > > >(omp_get_max_threads(), std::vector< std::vector< std::vector< int > > >());
        for (int tid = 0; tid < omp_get_max_threads(); tid++) {
            _delayed_spikes[tid] = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(sub_matrices_[tid]->post_rank.size(), std::vector< int >() ) );
        }
    #ifdef _DEBUG
        std::cout << "Inited _delayed_spikes[" << omp_get_max_threads() << "][" << max_delay << "] and " << std::endl;
        for (int tid = 0; tid < omp_get_max_threads(); tid++) {
            std::cout << "   _delayed_spikes[" << tid << "][:] with vectors of size = " << _delayed_spikes[tid][0].size() << " element(s) " << std::endl;
        }
    #endif
""",
        'reset': """
        while(!_delayed_spikes.empty()) {
            auto elem = _delayed_spikes.back();
            elem.clear();
            _delayed_spikes.pop_back();
        }

        idx_delay = 0;
        max_delay = pop%(id_pre)s.max_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< std::vector< int > > > >(omp_get_max_threads(), std::vector< std::vector< std::vector< int > > >());
        for (int tid = 0; tid < omp_get_max_threads(); tid++) {
            _delayed_spikes[tid] = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(sub_matrices_[tid]->post_rank.size(), std::vector< int >() ) );
        }

""",
        'pyx_struct':
"""
        # Non-uniform delay
        int max_delay
        void update_max_delay(int)
        void reset_ring_buffer()
        vector[vector[int]] get_delay()
        void set_delay(vector[vector[int]])
""",
        'pyx_wrapper_init': "",
        'pyx_wrapper_accessor':
"""
    # Access to non-uniform delay
    def get_delay(self):
        return proj%(id_proj)s.get_delay()
    def get_dendrite_delay(self, idx):
        return proj%(id_proj)s.get_delay()[idx]
    def set_delay(self, value):
        proj%(id_proj)s.set_delay(value)
    def get_max_delay(self):
        return proj%(id_proj)s.max_delay
    def set_max_delay(self, value):
        proj%(id_proj)s.max_delay = value
    def update_max_delay(self, value):
        proj%(id_proj)s.update_max_delay(value)
    def reset_ring_buffer(self):
        proj%(id_proj)s.reset_ring_buffer()
"""
    }
}

event_driven = {
    'declare': """
    std::vector<std::vector<std::vector<long> > > _last_event;
""",
    'cpp_init': """
    _last_event = init_matrix_variable<long>(-10000);
""",
    'pyx_struct': """
        vector[vector[long]] _last_event
""",
}

###############################################################
# Rate-coded continuous transmission
###############################################################
lil_summation_operation_single_matrix = {
    'sum' : """
%(pre_copy)s
nb_post = post_rank.size();

#pragma omp for private(sum) %(schedule)s
for(int i = 0; i < nb_post; i++) {
    sum = 0.0;
    for(int j = 0; j < pre_rank[i].size(); j++) {
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'max': """
%(pre_copy)s
nb_post = post_rank.size();

#pragma omp for %(schedule)s
for(int i = 0; i < nb_post; i++){
    int j = 0;
    sum = %(psp)s ;
    for(int j = 1; j < pre_rank[i].size(); j++){
        if(%(psp)s > sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'min': """
%(pre_copy)s
nb_post = post_rank.size();

#pragma omp for %(schedule)s
for(int i = 0; i < nb_post; i++){
    int j= 0;
    sum = %(psp)s ;
    for(int j = 1; j < pre_rank[i].size(); j++){
        if(%(psp)s < sum){
            sum = %(psp)s ;
        }
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
}
""",
    'mean': """
%(pre_copy)s
nb_post = post_rank.size();

#pragma omp for %(schedule)s
for(int i = 0; i < nb_post; i++){
    sum = 0.0 ;
    for(int j = 0; j < pre_rank[i].size(); j++){
        sum += %(psp)s ;
    }
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum / (double)(pre_rank[i].size());
}
"""
}

lil_summation_operation_sliced_matrix = {
    'sum' : """
%(pre_copy)s

#pragma omp private(sum)
{
    int tid = omp_get_thread_num();
    nb_post = sub_matrices_[tid]->post_rank.size();

    for(int i = 0; i < nb_post; i++) {
        sum = 0.0;
        for(int j = 0; j < sub_matrices_[tid]->pre_rank[i].size(); j++) {
            sum += %(psp)s ;
        }
        pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
    }
}
"""
}

###############################################################
# Rate-coded synaptic plasticity
###############################################################
update_variables_single_matrix = {
    'local': """
// Check periodicity
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L) ){
    // Global variables
    %(global)s
    // Local variables

    #pragma omp for private(rk_post, rk_pre) firstprivate(dt)
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i]; // Get postsynaptic rank
        // Semi-global variables
        %(semiglobal)s
        // Local variables
        for(int j = 0; j < pre_rank[i].size(); j++){
            rk_pre = pre_rank[i][j]; // Get presynaptic rank
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

    // Local variables
    #pragma omp parallel for
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i]; // Get postsynaptic rank
    %(semiglobal)s
    }
}
"""
}

update_variables_sliced_matrix = {
    'local': """
// Check periodicity
if(_transmission && _update && pop%(id_post)s._active && ( (t - _update_offset)%%_update_period == 0L) ){
    // Global variables
    %(global)s

    for(int i = 0; i < sub_matrices_[tid]->post_rank.size(); i++){
        rk_post = sub_matrices_[tid]->post_rank[i]; // Get postsynaptic rank

        // Semi-global variables
        %(semiglobal)s

        // Local variables
        for(int j = 0; j < sub_matrices_[tid]->pre_rank[i].size(); j++){
            rk_pre = sub_matrices_[tid]->pre_rank[i][j]; // Get presynaptic rank
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

    auto post_rank = get_post_rank();
    // Local variables
    #pragma omp parallel for
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i]; // Get postsynaptic rank
    %(semiglobal)s
    }

    post_rank.clear();
}
"""
}

###############################################################
# Spiking event-driven transmission
###############################################################
spiking_summation_fixed_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){

    int tid = omp_get_thread_num();

    // Iterate over all incoming spikes (possibly delayed constantly)
    for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++) {
        // Rank of the presynaptic neuron
        int rk_j = %(pre_array)s[_idx_j];

        // Find the presynaptic neuron in the inverse connectivity matrix
        auto inv_post_ptr = sub_matrices_[tid]->inv_pre_rank.find(rk_j);
        if (inv_post_ptr == sub_matrices_[tid]->inv_pre_rank.end())
            continue;

        // List of postsynaptic neurons receiving spikes from that neuron
        std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
        // Number of post neurons
        int nb_post = inv_post.size();

        // Iterate over connected post neurons
        for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
            // Retrieve the correct indices
            int i = inv_post[_idx_i].first;
            int j = inv_post[_idx_i].second;

            // Event-driven integration
            %(event_driven)s
            // Update conductance
            %(g_target)s
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
    }
} // active
"""


# Uses a ring buffer to process non-uniform delays in spiking networks
spiking_summation_variable_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){

    int tid = omp_get_thread_num();

    // Iterate over the spikes emitted during the last step in the pre population
    for(int idx_spike=0; idx_spike<pop%(id_pre)s.spiked.size(); idx_spike++){

        // Get the rank of the pre-synaptic neuron which spiked
        int rk_pre = pop%(id_pre)s.spiked[idx_spike];
        // List of post neurons receiving connections
        auto rks_post_beg = (sub_matrices_[tid]->inv_pre_rank[rk_pre]).begin();
        auto rks_post_end = (sub_matrices_[tid]->inv_pre_rank[rk_pre]).end();

        // Iterate over the post neurons
        for (auto rks_post_it = rks_post_beg; rks_post_it != rks_post_end; rks_post_it++){
            // Index of the post neuron in the connectivity matrix
            int i = rks_post_it->first ;
            // Index of the pre neuron in the connecivity matrix
            int j = rks_post_it->second ;
            // Delay of that connection
            int d = delay[tid][i][j]-1;
            // Index in the ring buffer
            int modulo_delay = (idx_delay + d) %% max_delay;
            // Add the spike in the ring buffer
            _delayed_spikes[tid][modulo_delay][i].push_back(j);
        }
    }

    #pragma omp barrier

    // Iterate over all post neurons having received spikes in the previous steps
    for (int i=0; i<_delayed_spikes[tid][idx_delay].size(); i++){

        for (int _idx_j=0; _idx_j<_delayed_spikes[tid][idx_delay][i].size(); _idx_j++){
            // Pre-synaptic index in the connectivity matrix
            int j = _delayed_spikes[tid][idx_delay][i][_idx_j];

            // Event-driven integration
            %(event_driven)s
            // Update conductance
            %(g_target)s
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
        // Empty the current list of the ring buffer
        _delayed_spikes[tid][idx_delay][i].clear();
    }
    #pragma omp barrier

    // Increment the index of the ring buffer
    #pragma omp single
    {
        idx_delay = (idx_delay + 1) %% max_delay;
    }
} // active
"""

spiking_post_event = """
if(_transmission && pop%(id_post)s._active){

    int tid = omp_get_thread_num();

    for(int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++){
        // In which sub matrix the neuron take place
        int rk_post = pop%(id_post)s.spiked[_idx_i];

        // Find its index in the projection
        auto it = find(sub_matrices_[tid]->post_rank.begin(), sub_matrices_[tid]->post_rank.end(), rk_post);

        // Leave if the neuron is not part of the projection
        if (it==sub_matrices_[tid]->post_rank.end()) continue;

        // which position
        int i = std::distance(sub_matrices_[tid]->post_rank.begin(), it);

        // Iterate over all synapse to this neuron
        int nb_pre = sub_matrices_[tid]->pre_rank[i].size();
        for(int j = 0; j < nb_pre; j++){
%(event_driven)s
%(post_event)s
        }
    }
}
"""

conn_templates = {
    # accessors
    'attribute_decl': attribute_single_matrix_decl,
    'attribute_sliced_matrix_decl': attribute_sliced_matrix_decl,
    'attribute_cpp_init': attribute_cpp_init,
    'delay': delay,
    'event_driven': event_driven,
    'rng_update': cpp_11_rng,

    # operations
    'rate_coded_sum_single_matrix': lil_summation_operation_single_matrix,
    'rate_coded_sum_sliced_matrix': lil_summation_operation_sliced_matrix,
    'update_variables_single_matrix': update_variables_single_matrix,
    'update_variables_sliced_matrix': update_variables_sliced_matrix,
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay,
    'spiking_sum_variable_delay': spiking_summation_variable_delay,
    'post_event': spiking_post_event
}
