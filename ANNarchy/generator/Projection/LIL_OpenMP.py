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
connectivity_matrix = {
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
""",
    'pyx_wrapper_args': "synapses",
    'pyx_wrapper_init': """
        cdef LIL syn = synapses
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
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj%(id_proj)s.get_pre_rank()
    def set_pre_rank(self, val):
        proj%(id_proj)s.set_pre_rank(val)
"""
}

weight_matrix = {
    'declare': """
    // LIL weights
    std::vector< std::vector< %(float_prec)s > > w;
""",
    'accessor': """
    // Local parameter w
    std::vector<std::vector< double > > get_w() {
        std::vector< std::vector< double > > w_new(w.size(), std::vector<double>());
        for(int i = 0; i < w.size(); i++) {
            w_new[i] = std::vector<double>(w[i].begin(), w[i].end());
        }
        return w_new;
    }
    std::vector< double > get_dendrite_w(int rk) { return std::vector<double>(w[rk].begin(), w[rk].end()); }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) {
        w = std::vector< std::vector<%(float_prec)s> >( value.size(), std::vector<%(float_prec)s>() );
        for(int i = 0; i < value.size(); i++) {
            w[i] = std::vector<%(float_prec)s>(value[i].begin(), value[i].end());
        }
    }
    void set_dendrite_w(int rk, std::vector< double > value) { w[rk] = std::vector<%(float_prec)s>(value.begin(), value.end()); }
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

single_weight_matrix = {
    'declare': """
    // Single weight in the projection
    %(float_prec)s w;
""",
    'accessor': "",
    'init': "",
    'pyx_struct': """
        # Local variable w
        %(float_prec)s w
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
    def set_dendrite_w(self, int rank, %(float_prec)s value):
        proj%(id_proj)s.w = value
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.w
    def set_synapse_w(self, int rank_post, int rank_pre, %(float_prec)s value):
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

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< std::vector<%(type)s > > %(name)s;
""",
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector< %(type)s >  %(name)s ;
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s  %(name)s ;
"""
}

attribute_acc = {
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
    'semiglobal':
"""
    // Semiglobal %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; }
""",
    'global':
"""
    // Global %(attr_type)s %(name)s
    %(type)s get_%(name)s() { return %(name)s; }
    void set_%(name)s(%(type)s value) { %(name)s = value; }
"""
}

attribute_cpp_init = {
    'local':
"""
        // Local %(attr_type)s %(name)s
        %(name)s = std::vector< std::vector<%(type)s> >(post_rank.size(), std::vector<%(type)s>());
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>(post_rank.size(), %(init)s);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
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
        'init': "",
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
    'nonuniform': {
        'declare': """
    // Non-uniform delay
    std::vector< std::vector< int > > delay ;
    int idx_delay;
    int max_delay;
    std::vector< std::vector< std::vector< int > > > _delayed_spikes ;""",
        'init': """
        idx_delay = 0;
        max_delay =  pop%(id_pre)s.max_delay ;
        _delayed_spikes = std::vector< std::vector< std::vector< int > > >(max_delay, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >()) );
""",
        'pyx_struct':
"""
        # Non-uniform delay
        vector[vector[int]] delay
        int max_delay
        void update_max_delay(int)""",
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
    def get_max_delay(self):
        return proj%(id_proj)s.delay
    def set_max_delay(self, value):
        proj%(id_proj)s.max_delay = value
    def update_max_delay(self, value):
        proj%(id_proj)s.update_max_delay(value)
"""
    }
}

event_driven = {
    'declare': """
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
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
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
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
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
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum;
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
    pop%(id_post)s._sum_%(target)s%(post_index)s += sum / (double)(pre_rank[i].size());
}
"""
}

spiking_summation_fixed_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){
    %(spiked_array_fusion)s

    // Iterate over all incoming spikes (possibly delayed constantly)
    %(omp_outer_loop)s
    for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++){
        // Rank of the presynaptic neuron
        int rk_j = %(pre_array)s[_idx_j];
        // Find the presynaptic neuron in the inverse connectivity matrix
        auto inv_post_ptr = inv_pre_rank.find(rk_j);
        if (inv_post_ptr == inv_pre_rank.end())
            continue;
        // List of postsynaptic neurons receiving spikes from that neuron
        std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
        // Number of post neurons
        int nb_post = inv_post.size();

        %(omp_inner_loop)s
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

    %(omp_reduce_code)s
} // active
"""

# Uses a ring buffer to process non-uniform delays in spiking networks
spiking_summation_variable_delay = """
// Event-based summation
if (_transmission && pop%(id_post)s._active){

    // Iterate over the spikes emitted during the last step in the pre population
    for(int idx_spike=0; idx_spike<pop%(id_pre)s.spiked.size(); idx_spike++){

        // Get the rank of the pre-synaptic neuron which spiked
        int rk_pre = pop%(id_pre)s.spiked[idx_spike];
        // List of post neurons receiving connections
        std::vector< std::pair<int, int> > rks_post = inv_pre_rank[rk_pre];

        // Iterate over the post neurons
        for(int x=0; x<rks_post.size(); x++){
            // Index of the post neuron in the connectivity matrix
            int i = rks_post[x].first ;
            // Index of the pre neuron in the connecivity matrix
            int j = rks_post[x].second ;
            // Delay of that connection
            int d = delay[i][j]-1;
            // Index in the ring buffer
            int modulo_delay = (idx_delay + d) %% max_delay;
            // Add the spike in the ring buffer
            _delayed_spikes[modulo_delay][i].push_back(j);
        }
    }

    // Iterate over all post neurons having received spikes in the previous steps
    for (int i=0; i<_delayed_spikes[idx_delay].size(); i++){
        for (int _idx_j=0; _idx_j<_delayed_spikes[idx_delay][i].size(); _idx_j++){
            // Pre-synaptic index in the connectivity matrix
            int j = _delayed_spikes[idx_delay][i][_idx_j];

            // Event-driven integration
            %(event_driven)s
            // Update conductance
            %(g_target)s
            // Synaptic plasticity: pre-events
            %(pre_event)s
        }
        // Empty the current list of the ring buffer
        _delayed_spikes[idx_delay][i].clear();
    }

    // Increment the index of the ring buffer
    idx_delay = (idx_delay + 1) %% max_delay;

} // active
"""

"""
    // Old stuff just in case
    // Iterate over all post neurons
    //%(omp_code)s
    for (int i=0; i<post_rank.size(); i++){
        for (int j=0; j<pre_rank[i].size(); j++){
            int d = delay[i][j]-1;
            if(std::find(pop%(id_pre)s._delayed_spike[d].begin(), pop%(id_pre)s._delayed_spike[d].end(), pre_rank[i][j]) != pop%(id_pre)s._delayed_spike[d].end()){

                %(event_driven)s
                %(g_target)s
                %(pre_event)s
            }
        }
    }
"""

conn_templates = {
    # connectivity
    'connectivity_matrix': connectivity_matrix,
    'inverse_connectivity_matrix': inverse_connectivity_matrix,
    'weight_matrix': weight_matrix,
    'single_weight_matrix': single_weight_matrix,

    # accessors
    'attribute_decl': attribute_decl,
    'attribute_acc':attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'delay': delay,
    'event_driven': event_driven,

    # operations
    'rate_coded_sum': lil_summation_operation,
    'spiking_sum_fixed_delay': spiking_summation_fixed_delay,
    'spiking_sum_variable_delay': spiking_summation_variable_delay
}
