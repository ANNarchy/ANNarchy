from ANNarchy.generator.OMP.PopulationTemplate import attribute_pyx_wrapper
header_struct_template = """#pragma once

#include "%(pre_name)s.hpp"
#include "%(post_name)s.hpp"

extern PopStruct%(pre_id)s pop%(pre_id)s;
extern PopStruct%(post_id)s pop%(post_id)s;

/////////////////////////////////////////
// proj%(id_proj)s: %(pre_name)s -> %(post_name)s with target %(target)s
/////////////////////////////////////////
struct ProjStruct%(id_proj)s{
    // number of dendrites
    int size;

    // Learning flag
    bool _learning;

    // Connectivity
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;

%(code)s

    void init_projection() {
%(init)s
    }

    void compute_psp() {
%(psp_prefix)s

%(psp)s
    }
    
    void update_synapse() {
        int rk_pre, rk_post;

%(update)s
    }

    // Accessors for c-wrapper
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }

%(accessor)s
};
"""

pyx_wrapper = """
cdef class proj%(id)s_wrapper :

    def __cinit__(self, synapses):

        cdef CSR syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()

        proj%(id)s.set_size( size )
        proj%(id)s.set_post_rank( syn.post_rank )
        proj%(id)s.set_pre_rank( syn.pre_rank )
        proj%(id)s.set_w(syn.w)
%(delay_init)s
%(exact_init)s

    property size:
        def __get__(self):
            return proj%(id)s.get_size()

    def nb_synapses(self, int n):
        return proj%(id)s.nb_synapses(n)

    def _set_learning(self, bool l):
        proj%(id)s._learning = l

    def post_rank(self):
        return proj%(id)s.get_post_rank()
    def pre_rank(self, int n):
        return proj%(id)s.get_pre_rank()[n]

%(accessor)s
%(structural_plasticity)s
"""

pyx_struct = """
    cdef struct ProjStruct%(id_proj)s :
        bool _learning

        int get_size()
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        int nb_synapses(int)
        void set_size(int)
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])

%(delay)s
%(exact)s
%(export)s
%(structural_plasticity)s
"""

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

# c like definition of accessors for synaptic attributes, whereas 'local' is used if values can vary
# across synapses, consequently 'global' is used if values are common to all neurons.
#
# Parameters:
#
#    type: data type of the variable (double, float, int ...)
#    name: name of the variable
#    attr_type: either 'variable' or 'parameter'
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

delay = {
    'decl':
"""
        vector[vector[int]] delay
""",
    'cinit':
"""
        proj%(id)s.delay = syn.delay
""",
    'pyx_wrapper_acc':
"""
    def get_delay(self):
        return proj%(id)s.delay
    def set_delay(self, value):
        proj%(id)s.delay = value
"""
}

exact_integ = {
    'decl':
"""
        vector[vector[long]] _last_event
""",
    'cinit':
"""
        proj%(id)s._last_event = vector[vector[long]](nb_post, vector[long]())
        for n in range(nb_post):
            proj%(id)s._last_event[n] = vector[long](proj%(id)s.nb_synapses(n), -10000)
"""
}

structural_plasticity = {
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