from ANNarchy.core.Projection import Projection
from ANNarchy.core.Synapse import Synapse
import ANNarchy.core.Global as Global

import numpy as np
from math import floor

# Code snippets
rank_to_coord = {
1 : """        int i_%(name)s = %(name)s;
""",
2 : """        int i_%(name)s = int(%(name)s / %(second)s);
        int j_%(name)s = %(name)s %(modulo)s %(second)s;
""",
3 : """        int i_%(name)s = int(%(name)s / (%(second)s*%(third)s));
        int j_%(name)s = int(%(name)s/%(third)s) %(modulo)s %(second)s;
        int k_%(name)s = %(name)s %(modulo)s %(third)s;
"""
}

coordinates_to_rank = {
1 : """i_%(name)s""",  
2 : """%(second)s*i_%(name)s + j_%(name)s""",  
3 : """%(third)s*(%(second)s*i_%(name)s + j_%(name)s) + k_%(name)s"""
}

class SharedProjection(Projection):
    """
    """
    def __init__(self, pre, post, target, psp="w * pre.r"):
        """
        Projection based on shared weights: each post-synaptic neuron uses the same weights, so they need to be instantiated only once to save memory.

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


    def convolve(self, weights, filter_or_kernel=True, padding=0.0):
        """
        Builds the shared connection pattern that will perform a convolution of the weights kernel on the pre-synaptic population.

            * **weights**: Numpy array or list of lists representing the matrix of weights for the filter/kernel.
            * **filter_or_kernel**: defines if the given weights are filter-based (dot-product) or kernel-based (convolution). TODO: explain. Default is kernel-based.
            * **padding**: value to be used for the rates outside the pre-synaptic population. If it is a floating value, the pre-synaptic population is virtually extended with this value. If it is equal to 'border', the values on the boundaries are repeated.
        """
        self.filter_or_kernel = filter_or_kernel
        self.padding = padding

        # Process the weights
        if isinstance(weights, list):
            self.weights = np.array(weights)
        else:
            self.weights = weights


        # Check dimensions of populations and weight matrix
        self.dim_kernel = self.weights.ndim
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension

        if self.dim_post > 3:
            Global._error('Too many dimensions for the post-synaptic population.')
            exit(0)
        if self.dim_pre > 3:
            Global._error('Too many dimensions for the pre-synaptic population.')
            exit(0)
        if self.dim_kernel > 3 or self.dim_kernel > self.dim_pre:
            Global._error('Too many dimensions for the kernel.')
            exit(0)

        if self.dim_kernel < self.dim_pre and self.dim_post < self.dim_pre: # pooling
            # must reshape the weight vector
            if self.dim_pre - self.dim_post == 1:
                self.weights = self.weights.reshape((1,) + self.weights.shape)
            else: # 2
                self.weights = self.weights.reshape((1,1,) + self.weights.shape)
            self.dim_kernel = self.weights.ndim


        # Generate the code
        self._generate_omp_code()

        # create fake CSR object, just for compilation.
        try:
            from ANNarchy.core.cython_ext.Connector import CSR
        except:
            _error('ANNarchy was not successfully installed.')
        csr = CSR() 
        self.connector_name = "Shared weights"
        self.connector_description = "Shared weights"
        self._store_csr(csr)
        return self

    
    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """        
        if not self._synapses:
            Global._error('The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not instantiated.')
            exit(0)

        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights.flatten())

        # Access the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Delete the _synapses array, not needed anymore
        del self._synapses
        self._synapses = None

    def _compute_coordinates(self):
        def center(i):
            return int(i/2) if i%2==1 else int(i/2)-1

        indices = ['i', 'j', 'k'] # WIll be easier to extend to more dimensions

        # Coordinates of the post neuron
        post = rank_to_coord[self.dim_post] % {    
            'name': 'post', 
            'first': self.post.geometry[0], 
            'second': self.post.geometry[1] if self.dim_post > 1 else '', 
            'third': self.post.geometry[2] if self.dim_post > 2 else '', 
            'modulo' : '%'}
        for dim, index in enumerate(indices):
            if dim < self.dim_post:
                post += """
        coords.push_back(%(idx)s_post);""" % {'idx': index}
        if self.dim_post < self.dim_pre : # The extra axis must be set to the middle of the filter
            for dim in range(self.dim_post, self.dim_pre):
                post += """
        int %(idx)s_post = %(idx)s_center;
        coords.push_back(%(idx)s_post);""" % {'idx': indices[dim]}


        # Coordinates in the kernel
        kernel = rank_to_coord[self.dim_kernel] % {    
            'name': 'w', 
            'first': self.weights.shape[0], 
            'second': self.weights.shape[1] if self.dim_kernel > 1 else '', 
            'third': self.weights.shape[2] if self.dim_kernel > 2 else '', 
            'modulo' : '%'}

        # Center of the kernel
        kernel_center = """
    // Center of the kernel"""
        for dim, index in enumerate(indices):
            if dim < self.dim_kernel:
                kernel_center += """
    int %(idx)s_center = %(val)s;""" % {'idx': index, 'val': center(self.weights.shape[dim]) }

        return post, kernel, kernel_center

    def _compute_index(self):

        code = """        // Compute the pre coordinates"""
        indices = ['i', 'j', 'k']
        for dim, index in enumerate(indices):
            if dim < self.dim_kernel:
                code += """
        int %(idx)s_pre = post[%(dim)s] %(filter_kernel)s (%(idx)s_w - %(idx)s_center);
        if(%(idx)s_pre < 0)
            %(border_min)s; 
        if(%(idx)s_pre >= %(size)s)
            %(border_max)s; """ % {
                'idx': index, 'dim': dim, 'size': self.pre.geometry[dim], 
                'filter_kernel': '-' if self.filter_or_kernel else '+',
                'border_min': index+"_pre = 0" if self.padding == 'border' else "return -1",
                'border_max': index+"_pre = "+ str(self.pre.geometry[dim]-1) if self.padding == 'border' else "return -1"
            }
            elif dim < self.dim_pre: # Have to continue to compute the other indices
                code += """
        int %(idx)s_pre = post[%(dim)s]; """ %{'idx': index, 'dim': dim}

        # Pre rank
        code += """
        // Return the pre rank
        return %(idx)s;
""" % {'idx': coordinates_to_rank[self.dim_pre] % {'name': 'pre', 
            'first': self.pre.geometry[0], 
            'second': self.pre.geometry[1] if self.dim_pre > 1 else '', 
            'third': self.pre.geometry[2] if self.dim_pre > 2 else ''} }
        return code

    def _generate_omp_code(self):

        # Index function
        post_coords, filter_index, filter_centers = self._compute_coordinates()

        # Convolution
        pre_index = self._compute_index()

        # C++ header
        self.generator['omp']['header_proj_struct'] = """
// Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
struct ProjStruct%(id_proj)s{
    std::vector<int> post_rank ;
    std::vector< double > w ;

%(filter_center)s

    std::vector<int> post_coords(int post){
        // Coordinates of the post neuron in its geometry
        std::vector<int> coords;
%(post_coords)s
        return coords;
    }

    int index(std::vector<int> post, int w){
        // Coordinates in the filter
%(filter_index)s
%(pre_index)s
    }
};  
""" % {'id_proj': self.id, 'name_pre': self.pre.name, 'name_post': self.post.name, 'target': self.target, 'filter_index': filter_index, 'filter_center': filter_centers, 'pre_index': pre_index, 'post_coords': post_coords}

        # PYX header
        self.generator['omp']['pyx_proj_struct'] = """
    # Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
    cdef struct ProjStruct%(id_proj)s :
        vector[int] post_rank
        vector[double] w 
""" % {'id_proj': self.id, 'name_pre': self.pre.name, 'name_post': self.post.name, 'target': self.target}

        # Pyx class
        self.generator['omp']['pyx_proj_class'] = """
# Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
cdef class proj%(id_proj)s_wrapper :
    def __cinit__(self, weights):
        proj%(id_proj)s.post_rank = list(range(%(size_post)s))
        proj%(id_proj)s.w = weights

    property size:
        def __get__(self):
            return %(size_post)s

    def nb_synapses(self, int n):
        return %(size_post)s * proj%(id_proj)s.w.size()

    def post_rank(self):
        return proj%(id_proj)s.post_rank
    def pre_rank(self, int n):
        return []

    # Local parameter w
    def get_w(self):
        return proj%(id_proj)s.w
    def set_w(self, value):
        proj%(id_proj)s.w = value
""" % { 'id_proj': self.id, 'target': self.target, 
        'name_pre': self.pre.name, 
        'name_post': self.post.name, 'size_post': self.post.size,
      }

        # No need to initialize anything (no recordable variable, no learning)
        self.generator['omp']['body_proj_init'] = " "

        # Compute sum
        if isinstance(self.padding, str):
            padding = 0.0
        else:
            padding = self.padding

        self.generator['omp']['body_compute_psp'] = """
    // Shared proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s. 
    #pragma omp parallel for private(sum)
    for(int i = 0; i < %(size_post)s; i++){
        sum = 0.0;
        std::vector<int> post_coords = proj%(id_proj)s.post_coords(i);
        for(int j = 0; j < proj%(id_proj)s.w.size(); j++){
            int rk = proj%(id_proj)s.index(post_coords, j);
            if(rk>=0)
                sum += proj%(id_proj)s.w[j] * pop%(id_pre)s.r[rk];
            else
                sum += proj%(id_proj)s.w[j] * %(padding)s;
        }
        pop%(id_post)s._sum_%(target)s[i] += sum;
    }
""" % {'id_proj': self.id, 
        'target': self.target,  
        'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
        'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
        'padding': padding
      }




