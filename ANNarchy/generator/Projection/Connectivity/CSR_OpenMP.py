#===============================================================================
#
#     CSR_OpenmMP.py
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
    std::vector<int> _row_ptr;
    std::vector<int> _post_ranks;
""",
    'accessor': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() {
        // TODO: maybe throw an exception here
        if ( _row_ptr.empty() )
            return std::vector<int>();

        auto post_ranks = std::vector<int>();
        for( int i = 0; i < _row_ptr.size()-1; i++ )
            if ( _row_ptr[i] != _row_ptr[i+1] )
                post_ranks.push_back(i);
        return post_ranks;
    }
    
    int nb_synapses(int n) { return _post_ranks.size(); }
    
    // CSR specific
    void set_row_ptr(std::vector<int> row_ptr) { _row_ptr = row_ptr; }
    void set_post_ranks(std::vector<int> post_ranks) { _post_ranks = post_ranks; }
""",
    'init': """
""",
    'pyx_struct': """
        # LIL Connectivity
        vector[int] get_post_rank()
        
        # CSR Connectivity
        void set_row_ptr(vector[int])
        void set_post_ranks(vector[int])
""",
    'pyx_wrapper_args': "synapses",
    'pyx_wrapper_init': """
        cdef CSR syn = synapses

        proj%(id_proj)s.set_row_ptr(syn._matrix.row_begin())
        proj%(id_proj)s.set_post_ranks(syn._matrix.column_indices())
""",
    'pyx_wrapper_accessor': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def set_post_rank(self, val):
        # makes no sense, as the row_idx cannot be initialized sensefully 
        pass
"""
}

weight_matrix = {
    'declare': """
    std::vector<double> w;
""",
    'accessor': """
    void set_w_csr(std::vector<double> w) { this->w = w; }
""",
    'init': """
""",
    'pyx_struct': """
        void set_w_csr(vector[double])
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        proj%(id_proj)s.set_w_csr(syn._matrix.values())
""",
    'pyx_wrapper_accessor': """
"""
}

single_weight_matrix = {
    'declare': """
    // Single weight in the projection
    // TODO:
""",
    'accessor': "",
    'init': "",
    'pyx_struct': """
        # Local variable w
        # TODO:
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        # Use only the first weight
        # TODO:
""",
    'pyx_wrapper_accessor': """
    # Local variable w
    # TODO:
"""
}
# These fields can remain empty as the CSRC already
# contains the backward view
inverse_connectivity_matrix = {
    'declare': """
""",
    'init': """
"""
}

attribute_decl = {
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
    'global':
"""
    // Global %(attr_type)s %(name)s
    std::vector<%(type)s> get_%(name)s() { return %(name)s; }
    %(type)s get_dendrite_%(name)s(int rk) { return %(name)s[rk]; }
    void set_%(name)s(std::vector<%(type)s> value) { %(name)s = value; }
    void set_dendrite_%(name)s(int rk, %(type)s value) { %(name)s[rk] = value; }
"""
}

attribute_cpp_init = {
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
}

delay = {
    'declare': """
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

conn_templates = {
    # connectivity
    'connectivity_matrix': connectivity_matrix,
    'inverse_connectivity_matrix': inverse_connectivity_matrix,
    'weight_matrix': weight_matrix,
    'single_weight_matrix': single_weight_matrix,
    
    # accessors
    'attribute_decl': attribute_decl,
    'attribute_acc': attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'delay': delay,
    'event_driven': event_driven
}