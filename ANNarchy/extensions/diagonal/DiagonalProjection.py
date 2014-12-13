from ANNarchy.core.Projection import Projection
from ANNarchy.core.Synapse import Synapse
import ANNarchy.core.Global as Global

import numpy as np


class DiagonalProjection(Projection):
    """
    Diagonal projection based on shared weights.
    """
    def __init__(self, pre, post, target):
        """
        *Parameters*:
                
            * **pre**: pre-synaptic population (either its name or a ``Population`` object).
            * **post**: post-synaptic population (either its name or a ``Population`` object).
            * **target**: type of the connection.
        """
        # Create the description, but it will not be used for generation
        Projection.__init__(
            self, 
            pre,
            post,
            target
        )


    def connect(self, weights, delays = Global.config['dt'], offset=0, slope=1):
        """
        Creates the diagonal connection pattern.

        *Parameters*:
                
            * **weights**: filter to be applied on each column (list or 1D Numpy array).

            * **delays**: transmission delays in ms (default: dt)

            * **offset**: start position for the diagonal for the post-neuron of first coordinate 0 (default: 0).

            * **slope**: slope of the diagonal (default: 1).
        """
        self.weights = weights
        self.delays = delays
        self.offset = offset
        self.slope = slope

        # Check conditions

        # Generate the code
        self._generate_omp()

        # create a fake CSR object
        self._create()
        return self

    def _create(self):
        # create fake CSR object, just for compilation.
        try:
            from ANNarchy.core.cython_ext.Connector import CSR
        except:
            _error('ANNarchy was not successfully installed.')
        csr = CSR()
        csr.max_delay = self.delays
        csr.uniform_delay = self.delays
        self.connector_name = "Diagonal Projection"
        self.connector_description = "Diagonal Projection"
        self._store_csr(csr)


    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """        
        if not self._synapses:
            Global._error('The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not instantiated.')
            exit(0)

        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Delete the _synapses array, not needed anymore
        del self._synapses
        self._synapses = None


    ################################
    ### Code generation
    ################################

    def _generate_omp(self):

        # C++ header
        self.generator['omp']['header_proj_struct'] = """
// Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
struct ProjStruct%(id_proj)s{

    std::vector<int> post_rank ;
    std::vector<double> w ;

};  
""" % { 'id_proj': self.id, 'name_pre': self.pre.name, 'name_post': self.post.name, 'target': self.target}

        # PYX header
        self.generator['omp']['pyx_proj_struct'] = """
    # Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
    cdef struct ProjStruct%(id_proj)s :
        vector[int] post_rank
        vector[double] w
""" % {'id_proj': self.id, 'name_pre': self.pre.name, 'name_post': self.post.name, 'target': self.target}

        # Pyx class
        proj_class = """
# Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
cdef class proj%(id_proj)s_wrapper :
    def __cinit__(self, weights):
        proj%(id_proj)s.post_rank = list(range(%(size_post)s))
        proj%(id_proj)s.w = weights

    property size:
        def __get__(self):
            return %(size_post)s
    def nb_synapses(self, int n):
        return 0

    def post_rank(self):
        return proj%(id_proj)s.post_rank
    def pre_rank(self, int n):
        return 0

    # Kernel
    def get_w(self):
        return proj%(id_proj)s.w
    def set_w(self, value):
        proj%(id_proj)s.w = value

"""
        self.generator['omp']['pyx_proj_class'] = proj_class % { 
            'id_proj': self.id, 'target': self.target, 
            'name_pre': self.pre.name, 
            'name_post': self.post.name, 
            'size_post': self.post.size,
        }

        # No need to initialize anything (no recordable variable, no learning)
        self.generator['omp']['body_proj_init'] = " "

        # Compute sum
        dim_post_0 = self.post.geometry[0]
        dim_post_1 = self.post.geometry[1]
        dim_pre_0 = self.pre.geometry[0]
        dim_pre_1 = self.pre.geometry[1]

        wsum =  """
    // Diagonal proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s. 
    if(pop%(id_post)s._active){
    int proj%(id_proj)s_idx_0, proj%(id_proj)s_idx_1, proj%(id_proj)s_idx_f;
    //#pragma omp parallel for private(proj%(id_proj)s_idx_0, proj%(id_proj)s_idx_1, proj%(id_proj)s_idx_f)
    for(int idx_0 = 0; idx_0 < %(dim_post_0)s; idx_0++){
        sum = 0.0;
        for(int idx_1 = 0; idx_1 < %(dim_pre_1)s; idx_1++){
            proj%(id_proj)s_idx_0 = (idx_0 + idx_1 + %(offset)s) %% %(dim_pre_0)s;
            proj%(id_proj)s_idx_1 = (%(inc1)s idx_1) %% %(dim_pre_1)s;
            for(int idx_f=0; idx_f < proj%(id_proj)s.w.size(); idx_f++){
                proj%(id_proj)s_idx_f = (proj%(id_proj)s_idx_1 + (idx_f - %(center_filter)s) ) %% %(dim_pre_1)s;
                sum += proj%(id_proj)s.w[idx_f] * pop%(id_pre)s.r[proj%(id_proj)s_idx_f + %(dim_pre_1)s* proj%(id_proj)s_idx_0];
            }
        }
        for(int idx_1 = 0; idx_1 < %(dim_post_1)s; idx_1++){
            pop%(id_post)s._sum_%(target)s[idx_1 + %(dim_post_1)s*idx_0] += sum;
        }
    }
    }// active
""" 

        if self.slope == 1 :
            inc1 = ""             
        elif self.slope > 1 :
            inc1 = str(self.slope) + '*'
        elif self.slope == 0 :
            inc1 = '0*'
        elif self.slope == -1 :
            inc1 = str(dim_pre_1 -1)  + '-' 
        else:
            inc1 = str(dim_pre_1 -1) + ' - ' + str(-self.slope)

        self.generator['omp']['body_compute_psp'] = wsum % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'offset': self.offset,
            'dim_post_0': dim_post_0, 'dim_post_1': dim_post_1,
            'dim_pre_0': dim_pre_0, 'dim_pre_1': dim_pre_1,
            'center_filter': int(len(self.weights)/2),
            'inc1': inc1
          }