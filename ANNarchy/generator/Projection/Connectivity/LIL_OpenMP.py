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
        void inverse_connectivity_matrix()
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

weight_matrix = {
    'declare': """
    // LIL weights
    std::vector< std::vector< %(float_prec)s > > w;
""",
    'accessor': """
    // Local parameter w
    std::vector<std::vector< %(float_prec)s > > get_w() { return w; }
    std::vector< %(float_prec)s > get_dendrite_w(int rk) { return w[rk]; }
    %(float_prec)s get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< %(float_prec)s > >value) { w = value; }
    void set_dendrite_w(int rk, std::vector< %(float_prec)s > value) { w[rk] = value; }
    void set_synapse_w(int rk_post, int rk_pre, %(float_prec)s value) { w[rk_post][rk_pre] = value; }
""",
    'init': """
""",
    'pyx_struct': """
        # Local variable w
        vector[vector[%(float_prec)s]] get_w()
        vector[%(float_prec)s] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[vector[%(float_prec)s]])
        void set_dendrite_w(int, vector[%(float_prec)s])
        void set_synapse_w(int, int, %(float_prec)s)
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
    def set_dendrite_w(self, int rank, vector[%(float_prec)s] value):
        proj%(id_proj)s.set_dendrite_w(rank, value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, %(float_prec)s value):
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
    'attribute_acc':attribute_acc,
    'attribute_cpp_init': attribute_cpp_init,
    'delay': delay,
    'event_driven': event_driven
}