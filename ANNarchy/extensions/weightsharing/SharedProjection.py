from ANNarchy.core.Projection import Projection
from ANNarchy.core.Synapse import Synapse
import ANNarchy.core.Global as Global

import numpy
from math import floor

class SharedProjection(Projection):
    """
    """
    def __init__(self, pre, post, target, psp="w * pre.r"):
        """
        *Parameters*:
                
            * **pre**: pre-synaptic population (either its name or a ``Population`` object).
            * **post**: post-synaptic population (either its name or a ``Population`` object).
            * **target**: type of the connection.
            * **psp**: function to be summed. By default: ``w * pre.r``
        """
        # Create the description, but it will not be used for generation
        Projection.__init__(
            self, 
            pre,
            post,
            target,
            synapse = Synapse(psp=psp)
        )

        self._psp_code = psp


    def connect(self, weights, delay = None):
        self.weights = weights

        try:
            from ANNarchy.core.cython_ext.Connector import CSR
        except:
            _error('ANNarchy was not successfully installed.')
        csr = CSR()

        # Generate the code
        self._generate_code()
        
        # Store the synapses
        self.connector_name = "Shared weights"
        self.connector_description = "Shared weights"
        self._store_csr(csr)
        return self


    def _generate_code(self):

        # C++ header
        self.generator['omp']['header_proj_struct'] = """
// Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
struct ProjStruct%(id_proj)s{
    int size;
    // Connectivity
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;

    // Local parameter w
    std::vector< std::vector< double > > w ;

};  
"""

        # PYX header
        self.generator['omp']['pyx_proj_struct'] = """
    # Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
    cdef struct ProjStruct%(id_proj)s :
        int size
        vector[int] post_rank
        vector[vector[int]] pre_rank

        # Local parameter w
        vector[vector[double]] w 
"""

        # Pyx class
        self.generator['omp']['pyx_proj_class'] = """
# Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
cdef class proj%(id_proj)s_wrapper :
    def __cinit__(self, synapses):
        cdef CSR syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()
        proj%(id_proj)s.size = size
        proj%(id_proj)s.post_rank = syn.post_rank
        proj%(id_proj)s.pre_rank = syn.pre_rank
        proj%(id_proj)s.w = syn.w

    property size:
        def __get__(self):
            return proj%(id_proj)s.size

    def nb_synapses(self, int n):
        return proj%(id_proj)s.pre_rank[n].size()

    def post_rank(self):
        return proj%(id_proj)s.post_rank
    def pre_rank(self, int n):
        return proj%(id_proj)s.pre_rank[n]

    # Local parameter w
    def get_w(self):
        return proj%(id_proj)s.w
    def set_w(self, value):
        proj%(id_proj)s.w = value
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.w[rank]
    def set_dendrite_w(self, int rank, vector[double] value):
        proj%(id_proj)s.w[rank] = value
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_proj)s.w[rank_post][rank_pre]
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj%(id_proj)s.w[rank_post][rank_pre] = value
"""

        # No need to initialize anything (no recordable variable, no learning)
        self.generator['omp']['body_proj_init'] = " "

        # Compute sum
        self.generator['omp']['body_compute_psp'] = """
    // Shared proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s. 
    #pragma omp parallel for private(sum)
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        sum = 0.0;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            sum += pop%(id_pre)s.r[proj%(id_proj)s.pre_rank[i][j]]*proj%(id_proj)s.w[i][j];
        }
        pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
    }
"""
