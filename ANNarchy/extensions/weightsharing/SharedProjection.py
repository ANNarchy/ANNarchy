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

filter_definition_template = {
    1 : """
    std::vector<double> w;""",
    2 : """
    std::vector< std::vector<double> > w;""",
    3 : """
    std::vector< std::vector< std::vector<double> > > w;""",
}

filter_definition_pyxtemplate = {
    1 : "vector[double] w",
    2 : "vector[vector[double]] w",
    3 : "vector[vector[vector[double]]] w",
}


""

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

    ################################
    ### Connection methods
    ################################

    def convolve(self, weights, filter_or_kernel=True, padding=0.0, subsampling=None):
        """
        Builds the shared connection pattern that will perform a convolution of the weights kernel on the pre-synaptic population.

        This is the simplest convolution form, where the pre-population and kernel have the same dimension (1, 2 or 3 dimensions). 

        If the post-population has the same dimension, the convolution is regular.

        If the post-population has one dimension less than the pre-synaptic one, the last dimension of the kernel must match the last one of the pre-synaptic population. For example, filtering a N*M*3 image with a 3D filter (3 elements in the third dimension) results into a 2D population.

        If the kernel has less dimensions than the two populations, the number of neurons in the last dimension of the populations must be the same. The convolution will be calculated for each position in the last dimension (parallel convolution, useful if the pre-synaptic population is a stack of feature maps, for example).  

        Sub-sampling will be automatically performed according to the populations' geometry. If these geometries do not match, an error will be thrown. You can force sub-sampling by providing a list ``subsampling`` as argument, defining for each post-synaptic neuron the coordinates of the pre-synaptic neuron which will be the center of the filter/kernel. 


        *Parameters*:

            * **weights**: Numpy array or list of lists representing the matrix of weights for the filter/kernel.
            * **filter_or_kernel**: defines if the given weights are filter-based (dot-product) or kernel-based (convolution). TODO: explain. Default is kernel-based.
            * **padding**: value to be used for the rates outside the pre-synaptic population. If it is a floating value, the pre-synaptic population is virtually extended with this value. If it is equal to 'border', the values on the boundaries are repeated.
        """
        self.filter_or_kernel = filter_or_kernel
        self.padding = padding
        self.subsampling = subsampling

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
            Global._error('Too many dimensions for the post-synaptic population (maximum 3).')
            exit(0)
        if self.dim_pre > 3:
            Global._error('Too many dimensions for the pre-synaptic population (maximum 3).')
            exit(0)
        if self.dim_kernel > 3:
            Global._error('Too many dimensions for the kernel (maximum 3).')
            exit(0)

        self.across_features = self.dim_kernel < self.dim_pre
        if self.across_features:
            if self.pre.shape[-1] != self.post.shape[-1]:
                Global._error('If the kernel has fewer dimensions than the two populations (parallel convolutions), these must have the same number of neurons in the last dimension(s).')
                exit(0)

        if self.dim_post < self.dim_pre: # OK, but check the last dimension of the kernel has the same size as the post-population
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                Global._error('If the post-synaptic population has less dimensions than the pre-synaptic one, the last dimension of the filter must be equal to the last of the pre-synaptic population.')
                exit(0)

        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates()

        # Filter definition
        filter_definition, filter_pyx_definition = self.filter_definition()

        # Convolve_code
        convolve_code = self.generate_convolve_code()

        # Generate the code
        self._generate_omp(filter_definition, filter_pyx_definition, convolve_code)

        # Finish building the synapses
        self._create()


    def _create(self):
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

        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights, self.pre_coordinates)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Delete the _synapses array, not needed anymore
        del self._synapses
        self._synapses = None


    ################################
    ### Utilities
    ################################

    def _center_filter(self, i):
        return int(i/2) if i%2==1 else int(i/2)-1

    def _generate_pre_coordinates(self):
        " Returns a list for each post neuron of the corresponding center coordinates."

        # Check if the list is already defined:
        if self.subsampling:
            try:
                shape = np.array(self.subsampling)
            except:
                Global._error('The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size, self.pre.dimension):
                Global._error('The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
                return
            self.pre_coordinates = self.subsampling

        # Otherwise create it, possibly with sub-sampling
        coords = [[] for i in range(self.post.size)]

        # Compute pre-indices
        idx_range= []
        for dim in range(self.dim_pre):
            if dim < self.dim_post:
                pre_size = self.pre.geometry[dim]
                post_size = self.post.geometry[dim]
                sample = int(pre_size/post_size)
                if post_size * sample != pre_size:
                    Global._error('The pre-synaptic dimensions must be a multiple of the post-synaptic ones for down-sampling to work.')
                    exit(0)
                idx_range.append([int((sample-1)/2) + sample * i for i in range(post_size)])
            else: # extra dimension
                if self.across_features:
                    idx_range.append(range(self.post.geometry[dim]))
                else:
                    idx_range.append([self._center_filter(self.weights.shape[dim])])

        # Generates coordinates
        if self.dim_pre == 1 :
            rk = 0
            for i in idx_range[0]:
                coords[rk] = [i]
                rk += 1
        elif self.dim_pre == 2 :
            rk = 0
            for i in idx_range[0]:
                for j in idx_range[1]:
                    coords[rk] = [i, j]
                    rk += 1
        elif self.dim_pre == 3 :
            rk = 0
            for i in idx_range[0]:
                for j in idx_range[1]:
                    for k in idx_range[2]:
                        coords[rk] = [i, j, k]
                        rk += 1

        # Save the result
        self.pre_coordinates = coords


    def filter_definition(self):
        cpp = filter_definition_template[self.dim_kernel] 
        pyx = filter_definition_pyxtemplate[self.dim_kernel] 
        return cpp, pyx

    def generate_convolve_code(self):

        # Padding
        if isinstance(self.padding, str):
            padding = 0.0
        else:
            padding = self.padding

        # Main code
        code = ""

        # Generate for loops
        indices = ['i', 'j', 'k']
        for dim in range(self.dim_kernel):
            code += """        for(int %(index)s_w = 0; %(index)s_w < %(size)s;%(index)s_w++){
    """ % { 'index': indices[dim], 'size': self.weights.shape[dim]}

        # Compute indices
        for dim in range(self.dim_pre):
            if dim < self.dim_kernel:
                code += """
            int %(index)s_pre = coord[%(dim)s] %(operator)s (%(index)s_w - %(center)s);""" % { 'index': indices[dim], 'dim': dim, 'operator': '-' if self.filter_or_kernel else '+', 'center': self._center_filter(self.weights.shape[dim])}
            else:
                code += """
            int %(index)s_pre = coord[%(dim)s];""" % { 'index': indices[dim], 'dim': dim}


        # Check indices
        for dim in range(self.dim_kernel):
            if isinstance(self.padding, str): # 'border'
                code += """
            if (%(index)s_pre < 0) %(index)s_pre = 0 ;
            if (%(index)s_pre > %(max_size)s) %(index)s_pre = %(max_size)s ;""" % { 'index': indices[dim], 'dim': dim, 'max_size': self.pre.geometry[dim] -1}
            else:
                code += """
            if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                sum += %(padding)s;
                break;
            }""" % { 'index': indices[dim], 'padding': self.padding, 'max_size': self.weights.shape[dim] -1}

        # Compute pre-synaptic rank
        code +="""
            int rk = %(value)s;""" % {'value': coordinates_to_rank[self.dim_pre] % {
                'name': 'pre', 
                'second': self.pre.geometry[1] if self.dim_pre > 1 else 0,
                'third': self.pre.geometry[2] if self.dim_pre > 2 else 0} 
            }

        # Perform the increment
        increment = "sum += proj%(id_proj)s.w"
        for dim in range(self.dim_kernel):
            increment += '[' + indices[dim] + '_w]'
        increment += " * pop%(id_pre)s.r[rk];"
        code += """
            %(increment)s""" % {'increment': increment}

        # Close for loops
        indices = ['i', 'j', 'k']
        for dim in range(self.dim_kernel):
            code += """
        }""" % { 'index': indices[dim]}

        return code % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size
          }

    ################################
    ### Code generation
    ################################

    def _generate_omp(self, filter_definition, filter_pyx_definition, convolve_code):

        # C++ header
        self.generator['omp']['header_proj_struct'] = """
// Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
struct ProjStruct%(id_proj)s{

    std::vector<int> post_rank ;
    std::vector< std::vector<int> > pre_coords ;

%(filter)s

};  
""" % { 'id_proj': self.id, 'name_pre': self.pre.name, 'name_post': self.post.name, 'target': self.target, 
        'filter': filter_definition}

        # PYX header
        self.generator['omp']['pyx_proj_struct'] = """
    # Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
    cdef struct ProjStruct%(id_proj)s :
        vector[int] post_rank
        vector[vector[int]] pre_coords
        %(filter)s
""" % {'id_proj': self.id, 'name_pre': self.pre.name, 'name_post': self.post.name, 'target': self.target, 
        'filter': filter_pyx_definition}

        # Pyx class
        self.generator['omp']['pyx_proj_class'] = """
# Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
cdef class proj%(id_proj)s_wrapper :
    def __cinit__(self, weights, coords):
        proj%(id_proj)s.post_rank = list(range(%(size_post)s))
        proj%(id_proj)s.pre_coords = coords
        proj%(id_proj)s.w = weights

    property size:
        def __get__(self):
            return %(size_post)s

    def nb_synapses(self, int n):
        return %(size_post)s * proj%(id_proj)s.w.size()

    def post_rank(self):
        return proj%(id_proj)s.post_rank
    def pre_rank(self, int n):
        return proj%(id_proj)s.pre_coords[n]

    # Kernel
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
        wsum =  """
    // Shared proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s. 
    #pragma omp parallel for private(sum)
    for(int i = 0; i < %(size_post)s; i++){
        sum = 0.0;
        std::vector<int> coord = proj%(id_proj)s.pre_coords[i];
""" + convolve_code + """
        pop%(id_post)s._sum_%(target)s[i] += sum;
    }
""" 

        self.generator['omp']['body_compute_psp'] = wsum % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size
          }




