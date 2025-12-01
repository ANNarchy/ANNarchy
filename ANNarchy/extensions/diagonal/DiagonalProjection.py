"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core.Projection import Projection
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages

import numpy as np


class DiagonalProjection(Projection):
    """
    Diagonal projection based on shared weights.
    """
    def __init__(self, pre, post, target, name=None, copied=False):
        """
        :param pre: pre-synaptic population (either its name or a ``Population`` object).
        :param post: post-synaptic population (either its name or a ``Population`` object).
        :param target: type of the connection.
        """
        # Create the description, but it will not be used for generation
        Projection.__init__(
            self, 
            pre,
            post,
            target,
            name=name,
            copied=copied
        )

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks.  Internal use only."
        return DiagonalProjection(pre=pre, post=post, target=self.target, name=self.name, copied=True)

    def connect(self, weights, delays=0, offset=0, slope=1):
        """
        Creates the diagonal connection pattern.

        :param weights: filter to be applied on each column (list or 1D Numpy array).
        :param delays: transmission delays in ms (default: dt)
        :param offset: start position for the diagonal for the post-neuron of first coordinate 0 (default: 0).
        :param slope: slope of the diagonal (default: 1).
        """
        self.weights = weights
        self.delays = delays
        self.offset = offset
        self.slope = slope

        # create a fake CSR object
        self._create()
        return self

    def _generate(self):
        # Generate the code
        if self.pre.dimension == 2 and self.post.dimension == 2:
            self._generate_omp_1d()
        elif self.pre.dimension == 4 and self.post.dimension == 4:
            self._generate_omp_2d_gaussian()
        else:
            Messages._error('The diagonal projection only works when both populations have 2 or 4 dimensions.')
            

    def connect_gaussian(self, amp, sigma, min_val, max_distance=0.0):
        """
        Creates the diagonal connection pattern for 4D populations and Gaussian filter..

        :param amp: maximal value of the Gaussian.
        :param sigma: width of the Gaussian.
        :param min_val: minimal value of the weight.
        :param max_distance: maximal distance for the Gaussian.

        """
        self.amp = amp
        self.sigma = sigma
        self.min_val = min_val
        self.max_distance = max_distance
        self.weights = {}

        if not(self.pre.dimension == 4 and self.post.dimension == 4):
            Messages._error('The diagonal projection only works when both populations have 4 dimensions.')
            

        self.offset_w = (self.pre.geometry[0]-(self.pre.geometry[0]%2))/2.0
        self.offset_h = (self.pre.geometry[1]-(self.pre.geometry[1]%2))/2.0
        self.sigma_w = self.sigma * (self.post.geometry[2] - self.post.geometry[2]%2 )
        self.sigma_h = self.sigma * (self.post.geometry[3] - self.post.geometry[3]%2 )

        # for post2 in range(self.post.geometry[2]):
        #     for post3 in range(self.post.geometry[3]):
        #         for pre0 in range(self.pre.geometry[0]):
        #             for pre1 in range(self.pre.geometry[1]):
        #                 for pre2 in range(self.pre.geometry[2]):
        #                     for pre3 in range(self.pre.geometry[3]):
        #                         dist_w = (post2 - (pre0+pre2) + self.offset_w)
        #                         dist_h = (post3 - (pre1+pre3) + self.offset_h)
        #                         val = self.amp * np.exp(- (dist_w*dist_w/self.sigma_w/self.sigma_w + dist_h*dist_h/self.sigma_h/self.sigma_h) )
        #                         self.weights[(dist_w, dist_h)] = val

        for dist_w in range(int(self.offset_w) - self.pre.geometry[0] - self.pre.geometry[2], int(self.offset_w) + self.post.geometry[2]):
            for dist_h in range(int(self.offset_h) - self.pre.geometry[1] - self.pre.geometry[3], int(self.offset_h) + self.post.geometry[3]):
                val = self.amp * np.exp(- (dist_w*dist_w/self.sigma_w/self.sigma_w + dist_h*dist_h/self.sigma_h/self.sigma_h) )
                self.weights[(dist_w, dist_h)] = val
        
        # create a fake CSR object
        self._create()
        return self



    def _create(self):
        # create fake CSR object, just for compilation.
        try:
            from ANNarchy.cython_ext.Connector import LILConnectivity
        except:
            Messages._error('ANNarchy was not successfully installed.')
        lil = LILConnectivity(dt=ConfigManager().get('dt', self.net_id))
        lil.max_delay = 0
        lil.uniform_delay = 0
        self.connector_name = "Diagonal Projection"
        self.connector_description = "Diagonal Projection"
        self._store_connectivity(self._load_from_lil, (lil, ), 0)


    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """        
        if not self._connection_method:
            Messages._error('The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')
            
        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        return True

    ################################
    ### Code generation
    ################################

    def _generate_omp_1d(self):
        """
        Generate openMP template code.
        """
        # Specific template for generation
        self._specific_template = {
            # Declare the connectivity matrix
            'declare_connectivity_matrix': """
    std::vector<int> post_rank;
    std::vector< %(float_prec)s > w;
""" % {'float_prec': ConfigManager().get('precision', self.net_id)},

            # Accessors for the connectivity matrix
            'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    int dendrite_size(int n) { return w.size(); }
    // Weights w
    std::vector< %(float_prec)s > get_w() { return w; }
    void set_w(std::vector< %(float_prec)s > _w) { w=_w; }
""" % {'float_prec': ConfigManager().get('precision', self.net_id)},

            # Export the connectivity matrix
            'export_connectivity': """
        # Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        vector[%(float_prec)s] get_w()
        void set_w(vector[%(float_prec)s])
""" % {'float_prec': ConfigManager().get('precision', self.net_id)},

            # Arguments to the wrapper constructor
            'wrapper_args': "weights",

            # Initialize the wrapper connectivity matrix
            'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_w(weights)
""" % {'id_proj': self.id, 'size_post': self.post.size},

            # Wrapper access to connectivity matrix
            'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_rank(self, int n):
        return 0
""" % {'id_proj': self.id},

            # Wrapper access to variables
            'wrapper_access_parameters_variables' : "",

            # Variables for the psp code
            'psp_prefix': """
        %(float_prec)s sum=0.0;"""
        } % {'float_prec': ConfigManager().get('precision', self.net_id)}

        # Compute sum
        dim_post_0 = self.post.geometry[0]
        dim_post_1 = self.post.geometry[1]
        dim_pre_0 = self.pre.geometry[0]
        dim_pre_1 = self.pre.geometry[1]

        # Pre-defined variables
        wsum =  """
        int _idx_0, _idx_1, _idx_f, _start;
        std::vector<%(float_prec)s> _w = w;
        std::vector<%(float_prec)s> _pre_r = pop%(id_pre)s.r;
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        # OpenMP statement
        if ConfigManager().get('num_threads', self.net_id) > 1:
            wsum += """
        #pragma omp for private(sum, _idx_0, _idx_1, _idx_f, _start) firstprivate(_w, _pre_r)"""

        # Computation Kernel
        wsum += """
        for(int idx = 0; idx < %(dim_post_1)s; idx++){
            sum = 0.0;
            _start = (idx %(inc0)s %(offset)s ) ;
            //std::cout << "Neuron: " << idx << " : " << _start << std::endl;
            for(int idx_1 = 0; idx_1 < %(dim_pre_1)s; idx_1++){
                _idx_0 = idx_1;
                _idx_1 = _start + %(inc1)s idx_1;
                if ((_idx_1 < 0) || (_idx_1 > %(dim_pre_1)s-1))
                    continue;
                //std::cout << _idx_0 << " " << _idx_1 << std::endl;
                for(int idx_f=0; idx_f < %(size_filter)s; idx_f++){
                    _idx_f = (_idx_1 + (idx_f - %(center_filter)s) );
                    if ((_idx_f < 0) || (_idx_f > %(dim_pre_1)s-1))
                        continue;
                    sum += _w[idx_f] * _pre_r[_idx_f + %(dim_pre_1)s * _idx_0];
                }
            }
            for(int idx_1 = 0; idx_1 < %(dim_post_0)s; idx_1++){
                pop%(id_post)s._sum_%(target)s[idx + %(dim_post_1)s*idx_1] += sum;
            }
        }
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

        self._specific_template['psp_code'] = wsum % {'id_proj': self.id, 
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
        # Specific template for generation
        self._specific_template = {
            # Declare the connectivity matrix
            'declare_connectivity_matrix': """
    std::vector<int> post_rank;
    std::map<std::pair<int, int>, %(float_prec)s > w ;
""" % {'float_prec': ConfigManager().get('precision', self.net_id)},

            # Accessors for the connectivity matrix
            'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    int dendrite_size(int n) { return w.size(); }
    // Weights w
    std::map<std::pair<int, int>, %(float_prec)s > get_w() { return w; }
    void set_w(std::map<std::pair<int, int>, %(float_prec)s > _w) { w=_w; }
""" % {'float_prec': ConfigManager().get('precision', self.net_id)},

            # Export the connectivity matrix
            'export_connectivity': """
        # Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        map[pair[int, int], %(float_prec)s] get_w()
        void set_w(map[pair[int, int], %(float_prec)s])
""" % {'float_prec': ConfigManager().get('precision', self.net_id)},

            # Arguments to the wrapper constructor
            'wrapper_args': "weights",

            # Initialize the wrapper connectivity matrix
            'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_w(weights)
""" % {'id_proj': self.id, 'size_post': self.post.size},

            # Wrapper access to connectivity matrix
            'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_rank(self, int n):
        return 0
            """ % {'id_proj': self.id},

            # Wrapper access to variables
            'wrapper_access_parameters_variables' : "",

            # Variables for the psp code
            'psp_prefix': """
        %(float_prec)s sum=0.0;"""
        } % {'float_prec': ConfigManager().get('precision', self.net_id)}

        # Compute sum
        wsum =  """
        std::vector<%(float_prec)s> result(%(postdim2)s*%(postdim3)s, 0.0);""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        if ConfigManager().get('num_threads', self.net_id) > 1:
            wsum += """
        #pragma omp for"""
    
        wsum += """
        for(int post2 = 0; post2 < %(postdim2)s; post2++){
            for(int post3 = 0; post3 < %(postdim3)s; post3++){
                %(float_prec)s sum = 0.0;
                for(int pre0 = 0; pre0 < %(predim0)s; pre0++){
                    for(int pre1 = 0; pre1 < %(predim1)s; pre1++){
                        for(int pre2 = 0; pre2 < %(predim2)s; pre2++){
                            for(int pre3 = 0; pre3 < %(predim3)s; pre3++){
                                int dist_w = post2 - (pre0+pre2) + %(offset_w)s;
                                int dist_h = post3 - (pre1+pre3) + %(offset_h)s;
                                %(float_prec)s val = proj%(id_proj)s.w[std::pair<int, int>(dist_w, dist_h)];
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
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        if self.max_distance != 0.0:
            wgd = "&& abs(dist_w) < %(mgd)s && abs(dist_h) < %(mgd)s" % {'mgd': self.max_distance}
        else:
            wgd=""

        self._specific_template['psp_code'] = wsum % {
            'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'predim0': self.pre.geometry[0], 'predim1': self.pre.geometry[1], 'predim2': self.pre.geometry[2], 'predim3': self.pre.geometry[3], 
            'postdim0': self.post.geometry[0], 'postdim1': self.post.geometry[1], 'postdim2': self.post.geometry[2], 'postdim3': self.post.geometry[3], 
            'offset_w': self.offset_w, 'offset_h': self.offset_h,
            'amp': self.amp, 'sigma_w': self.sigma_w, 'sigma_h': self.sigma_h, 'min_val': self.min_val, 'wgd': wgd
          }