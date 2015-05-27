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

        # Generate the code
        if self.pre.dimension == 2 and self.post.dimension == 2:
            self._generate_omp_1d()
        # elif self.pre.dimension == 4 and self.post.dimension == 4:
        #     self._generate_omp_2d()
        else:
            Global._error('The diagonal projection only works when both populations have 2 or 4 dimensions.')
            exit(0)

        # create a fake CSR object
        self._create()
        return self


    def connect_gaussian(self, amp, sigma, min_val, max_distance=0.0):
        """
        Creates the diagonal connection pattern for 4D populations and Gaussian filter..

        *Parameters*:
                
            * **amp**: maximal value of the Gaussian.
            * **sigma**: width of the Gaussian.
            * **min_val**: minimal value of the weight.
            * **max_distance**: maximal distance for the Gaussian.

        """
        self.amp = amp
        self.sigma = sigma
        self.min_val = min_val
        self.max_distance = max_distance
        self.weights = {}

        if not(self.pre.dimension == 4 and self.post.dimension == 4):
            Global._error('The diagonal projection only works when both populations have 4 dimensions.')
            exit(0)

        self.offset_w = (self.pre.geometry[0]-(self.pre.geometry[0]%2))/2.0
        self.offset_h = (self.pre.geometry[1]-(self.pre.geometry[1]%2))/2.0
        self.sigma_w = self.sigma * (self.post.geometry[2] - self.post.geometry[2]%2 )
        self.sigma_h = self.sigma * (self.post.geometry[3] - self.post.geometry[3]%2 )

        # for post2 in xrange(self.post.geometry[2]):
        #     for post3 in xrange(self.post.geometry[3]):
        #         for pre0 in xrange(self.pre.geometry[0]):
        #             for pre1 in xrange(self.pre.geometry[1]):
        #                 for pre2 in xrange(self.pre.geometry[2]):
        #                     for pre3 in xrange(self.pre.geometry[3]):                                    
        #                         dist_w = (post2 - (pre0+pre2) + self.offset_w)
        #                         dist_h = (post3 - (pre1+pre3) + self.offset_h)
        #                         val = self.amp * np.exp(- (dist_w*dist_w/self.sigma_w/self.sigma_w + dist_h*dist_h/self.sigma_h/self.sigma_h) )
        #                         self.weights[(dist_w, dist_h)] = val

        for dist_w in xrange(int(self.offset_w) - self.pre.geometry[0] - self.pre.geometry[2], int(self.offset_w) + self.post.geometry[2]):
            for dist_h in xrange(int(self.offset_h) - self.pre.geometry[1] - self.pre.geometry[3], int(self.offset_h) + self.post.geometry[3]):
                val = self.amp * np.exp(- (dist_w*dist_w/self.sigma_w/self.sigma_w + dist_h*dist_h/self.sigma_h/self.sigma_h) )
                self.weights[(dist_w, dist_h)] = val


        # Generate the code
        self._generate_omp_2d_gaussian()
        
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
        csr.max_delay = 0
        csr.uniform_delay = 0
        self.connector_name = "Diagonal Projection"
        self.connector_description = "Diagonal Projection"
        self._store_connectivity(self._load_from_csr, (csr, ), 0)


    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """        
        if not self._connection_method:
            Global._error('The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')
            exit(0)

        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))



    ################################
    ### Code generation
    ################################

    def _generate_omp_1d(self):

        # C++ header
        self.generator['omp']['header_proj_struct'] = """
// Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
struct ProjStruct%(id_proj)s{
    bool _learning;

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
        int proj%(id_proj)s_idx_0, proj%(id_proj)s_idx_1, proj%(id_proj)s_idx_f, proj%(id_proj)s_start;
        std::vector<double> proj%(id_proj)s_w = proj%(id_proj)s.w;
        std::vector<double> proj%(id_proj)s_pre_r = pop%(id_pre)s.r;"""
        if Global.config['num_threads'] > 1:
            wsum += """
        #pragma omp parallel for private(sum, proj%(id_proj)s_idx_0, proj%(id_proj)s_idx_1, proj%(id_proj)s_idx_f, proj%(id_proj)s_start) firstprivate(proj%(id_proj)s_w, proj%(id_proj)s_pre_r)"""
        wsum += """
        for(int idx = 0; idx < %(dim_post_1)s; idx++){
            sum = 0.0;
            proj%(id_proj)s_start = (idx %(inc0)s %(offset)s ) ;
            //std::cout << "Neuron: " << idx << " : " << proj%(id_proj)s_start << std::endl;
            for(int idx_1 = 0; idx_1 < %(dim_pre_1)s; idx_1++){
                proj%(id_proj)s_idx_0 = idx_1;
                proj%(id_proj)s_idx_1 = proj%(id_proj)s_start + %(inc1)s idx_1;
                if ((proj%(id_proj)s_idx_1 < 0) || (proj%(id_proj)s_idx_1 > %(dim_pre_1)s-1))
                    continue;
                //std::cout << proj%(id_proj)s_idx_0 << " " << proj%(id_proj)s_idx_1 << std::endl;
                for(int idx_f=0; idx_f < %(size_filter)s; idx_f++){
                    proj%(id_proj)s_idx_f = (proj%(id_proj)s_idx_1 + (idx_f - %(center_filter)s) );
                    if ((proj%(id_proj)s_idx_f < 0) || (proj%(id_proj)s_idx_f > %(dim_pre_1)s-1))
                        continue;
                    sum += proj%(id_proj)s_w[idx_f] * proj%(id_proj)s_pre_r[proj%(id_proj)s_idx_f + %(dim_pre_1)s * proj%(id_proj)s_idx_0];
                }
            }
            for(int idx_1 = 0; idx_1 < %(dim_post_0)s; idx_1++){
                pop%(id_post)s._sum_%(target)s[idx + %(dim_post_1)s*idx_1] += sum;
            }
        }
    }// active
""" 

        if self.slope == 1 :
            inc0 = "-"
            inc1 = ""             
        elif self.slope > 1 :
            inc0 = " - "
            inc1 = str(self.slope) + '*'
        elif self.slope == 0 :
            inc0 = "-"
            inc1 = '0*'
        elif self.slope == -1 :
            inc0 = "+"
            inc1 = '-' 
        else:
            inc0 = "+"
            inc1 = ' - ' + str(-self.slope) + '*'

        self.generator['omp']['body_compute_psp'] = wsum % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'offset': self.offset,
            'dim_post_0': dim_post_0, 'dim_post_1': dim_post_1,
            'dim_pre_0': dim_pre_0, 'dim_pre_1': dim_pre_1,
            'size_filter': len(self.weights),
            'center_filter': int(len(self.weights)/2),
            'inc0': inc0,
            'inc1': inc1
          }

    def _generate_omp_2d_gaussian(self):

        # C++ header
        self.generator['omp']['header_proj_struct'] = """
// Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
struct ProjStruct%(id_proj)s{

    std::vector<int> post_rank ;
    std::map<std::pair<int, int>, double > w ;

};  
""" % { 'id_proj': self.id, 'name_pre': self.pre.name, 'name_post': self.post.name, 'target': self.target}

        # PYX header
        self.generator['omp']['pyx_proj_struct'] = """
    # Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
    cdef struct ProjStruct%(id_proj)s :
        vector[int] post_rank
        map[pair[int, int], double] w
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
        wsum =  """
    // Diagonal proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s. 
    if(pop%(id_post)s._active){
        std::vector<double> result(%(postdim2)s*%(postdim3)s, 0.0);"""

        if Global.config['num_threads'] > 1:
            wsum += """
        #pragma omp parallel for"""
    
        wsum += """
        for(int post2 = 0; post2 < %(postdim2)s; post2++){
            for(int post3 = 0; post3 < %(postdim3)s; post3++){
                double sum = 0.0;
                for(int pre0 = 0; pre0 < %(predim0)s; pre0++){
                    for(int pre1 = 0; pre1 < %(predim1)s; pre1++){
                        for(int pre2 = 0; pre2 < %(predim2)s; pre2++){
                            for(int pre3 = 0; pre3 < %(predim3)s; pre3++){
                                int dist_w = post2 - (pre0+pre2) + %(offset_w)s;
                                int dist_h = post3 - (pre1+pre3) + %(offset_h)s;
                                double val = proj%(id_proj)s.w[std::pair<int, int>(dist_w, dist_h)];
                                if(val > %(min_val)s%(wgd)s){
                                    sum += val * pop%(id_pre)s.r[pre3 + %(predim3)s * (pre2 + %(predim2)s*(pre1 + %(predim1)s * pre0))];
                                }
                            }
                        }
                    }
                }
                result[post3 + %(postdim3)s * post2] = sum;
            }
        }
        // Copy the result multiple times
        for(int i=0; i<%(postdim0)s*%(postdim1)s; i++){
            for(int j=0; j<%(postdim2)s*%(postdim3)s; j++){
                pop%(id_post)s._sum_%(target)s[j + i*(%(postdim2)s*%(postdim3)s)] += result[j];
            }
        }
    }// active
""" 


        if self.max_distance != 0.0:
            wgd = "&& abs(dist_w) < %(mgd)s && abs(dist_h) < %(mgd)s" % {'mgd': self.max_distance}
        else:
            wgd=""

        self.generator['omp']['body_compute_psp'] = wsum % {
            'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'predim0': self.pre.geometry[0], 'predim1': self.pre.geometry[1], 'predim2': self.pre.geometry[2], 'predim3': self.pre.geometry[3], 
            'postdim0': self.post.geometry[0], 'postdim1': self.post.geometry[1], 'postdim2': self.post.geometry[2], 'postdim3': self.post.geometry[3], 
            'offset_w': self.offset_w, 'offset_h': self.offset_h,
            'amp': self.amp, 'sigma_w': self.sigma_w, 'sigma_h': self.sigma_h, 'min_val': self.min_val, 'wgd': wgd
          }