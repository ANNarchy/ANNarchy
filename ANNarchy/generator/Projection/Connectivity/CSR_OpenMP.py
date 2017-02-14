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
    // LIL connectivity, just for interface
    std::vector<int> post_ranks;

    // CSR connectivity
    std::vector<int> _row_ptr;
    std::vector<int> _col_idx;
    int _nb_synapses;
""",
    'accessor': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_ranks; }
    int nb_synapses(int n) { return _nb_synapses; }

    // CSR specific
    void set_row_ptr(std::vector<int> row_ptr) { _row_ptr = row_ptr; }
    void set_col_idx(std::vector<int> col_idx) { _col_idx = col_idx; _nb_synapses = _col_idx.size(); }
""",
    'init': """
""",
    'pyx_struct': """
        # LIL Connectivity
        vector[int] get_post_rank()
        
        # CSR Connectivity
        void set_row_ptr(vector[int])
        void set_col_idx(vector[int])
        void inverse_connectivity_matrix()
""",
    'pyx_wrapper_args': "synapses",
    'pyx_wrapper_init': """
        cdef %(csr_type)s syn = synapses

        proj%(id_proj)s.set_row_ptr(syn._matrix.row_begin())
        proj%(id_proj)s.set_col_idx(syn._matrix.column_indices())
        proj%(id_proj)s.inverse_connectivity_matrix()
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
    std::vector< double > get_dendrite_w(int rk) {
        std::vector<double> res;
        for(int j = _col_ptr[rk]; j < _col_ptr[rk+1]; j++)
            res.push_back(w[_inv_idx[j]]);
        return res;
    }
    std::vector< std::vector<double> > get_w() {
        std::vector< std::vector<double> > res;
        for(auto it =post_ranks.begin(); it != post_ranks.end(); it++ ) {
            res.push_back(std::move(get_dendrite_w(*it)));
        }
        return res;
    }
""",
    'init': """
""",
    'pyx_struct': """
        # Initialization
        void set_w_csr(vector[double])

        # Interface access
        vector[double] get_dendrite_w(int)
        vector[vector[double]] get_w()
""",
    'pyx_wrapper_args': "",
    'pyx_wrapper_init': """
        proj%(id_proj)s.set_w_csr(syn._matrix.values())
""",
    'pyx_wrapper_accessor': """
    def get_w(self):
        return proj%(id_proj)s.get_w()
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.get_dendrite_w(rank)
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

inverse_connectivity_matrix = {
    'declare': """
    // CSR inverse
    std::vector<int> _col_ptr;
    std::vector<int> _row_idx;
    std::vector<int> _inv_idx;
    bool _inv_computed = false;
""",
    'init': """
        if ( _inv_computed )
            return;

        if (_row_ptr.empty()) {
            std::cerr << "no row_ptr data ..." << std::endl;
            return;
        }

        //
        // 2-pass algorithm: 1st we compute the inverse connectivity as LIL, 2ndly transform it to CSR
        //
        std::map< int, std::vector< int > > inv_post_rank = std::map< int, std::vector< int > >();
        std::map< int, std::vector< int > > inv_idx = std::map< int, std::vector< int > >();

        // iterate over post neurons, post_rank_it encodes the current rank
        for( int i = 0; i < (_row_ptr.size()-1); i++ ) {
            int row_begin = _row_ptr[i];
            int row_end = _row_ptr[i+1];

            // iterate over synapses, update both result containers
            for( int syn_idx = row_begin; syn_idx < row_end; syn_idx++ ) {
                inv_post_rank[_col_idx[syn_idx]].push_back(i);
                inv_idx[_col_idx[syn_idx]].push_back(syn_idx);
            }
        }

        //
        // store as CSR
        //
        _col_ptr.clear();
        _row_idx.clear();
        _inv_idx.clear();
        int curr_off = 0;

        // iterate over pre-neurons
        for ( int i = 0; i < %(post_size)s; i++) {
            if ( !inv_post_rank[i].empty() ) {
                post_ranks.push_back(i);
                _col_ptr.push_back( curr_off );
                _row_idx.insert(_row_idx.end(), inv_post_rank[i].begin(), inv_post_rank[i].end());
                _inv_idx.insert(_inv_idx.end(), inv_idx[i].begin(), inv_idx[i].end());

                curr_off += inv_post_rank[i].size();
            }
        }
        _col_ptr.push_back(curr_off);

        if ( _nb_synapses != curr_off )
            std::cerr << "Something went wrong: (nb_synapes = " << _nb_synapses << ") != (curr_off = " << curr_off << ")" << std::endl;
    #ifdef _DEBUG_CONN
        std::cout << "Pre to Post:" << std::endl;
        for ( int i = 0; i < pop%(id_pre)s.size; i++ ) {
            std::cout << i << ": " << col_ptr[i] << " -> " << col_ptr[i+1] << std::endl;
        }
    #endif

        _inv_computed = true;
"""
}

attribute_decl = {
    'local':
"""
    // Local %(attr_type)s %(name)s
    std::vector< %(type)s > %(name)s;
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
    std::vector<std::vector< %(type)s > > get_%(name)s() {
        std::vector< std::vector< %(type)s > > res;
        for(auto it = post_ranks.begin(); it != post_ranks.end(); it++ ) {
            res.push_back(std::move(get_dendrite_%(name)s(*it)));
        }
        return res;
    }
    std::vector<%(type)s> get_dendrite_%(name)s(int rk) {
        std::vector<%(type)s> res;
        for(int j = _col_ptr[rk]; j < _col_ptr[rk+1]; j++)
            res.push_back(%(name)s[_inv_idx[j]]);
        return res;
    }
    %(type)s get_synapse_%(name)s(int rk_post, int rk_pre) {
        for(int j = _col_ptr[rk_post]; j < _col_ptr[rk_post+1]; j++)
            if ( _row_idx[j] == rk_pre )
                return %(name)s[_inv_idx[j]];
    }
    void set_%(name)s(std::vector<std::vector< %(type)s > >value) { }
    void set_dendrite_%(name)s(int rk, std::vector<%(type)s> value) { }
    void set_synapse_%(name)s(int rk_post, int rk_pre, %(type)s value) { }
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
        %(name)s = std::vector< %(type)s >( _nb_synapses, %(init)s);
""",
    'semiglobal':
"""
        // Semiglobal %(attr_type)s %(name)s
        %(name)s = std::vector<%(type)s>( post_ranks.size(), %(init)s);
""",
    'global':
"""
        // Global %(attr_type)s %(name)s
        %(name)s = %(init)s;
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
    std::vector< long > _last_event;
""",
    'cpp_init': """
""",
    'pyx_struct': """
        vector[long] _last_event
""",
    'pyx_wrapper_init':
"""
        proj%(id_proj)s._last_event = vector[long]( syn._matrix.num_elements(), -10000)
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
