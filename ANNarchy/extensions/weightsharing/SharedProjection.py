from ANNarchy.core.Projection import Projection
from ANNarchy.core.Synapse import Synapse
import ANNarchy.core.Global as Global

import numpy as np
from math import floor

# Indices used for each dimension
indices = ['i', 'j', 'k', 'l', 'm', 'n']


class SharedProjection(Projection):
    """
    """
    def __init__(self, pre, post, target, psp="w * pre.r", operation="sum"):
        """
        Projection based on shared weights: each post-synaptic neuron uses the same weights, so they need to be instantiated only once to save memory.

        Learning is not possible for now. The ``synapse`` argument is removed, replaced by a single ``psp`` argument to modified what is summed and ``operation`` to replace the summation operation by max-pooling or similar.. 

        *Parameters*:
                
            * **pre**: pre-synaptic population (either its name or a ``Population`` object).
            * **post**: post-synaptic population (either its name or a ``Population`` object).
            * **target**: type of the connection.
            * **psp**: function to be summed. By default: ``w * pre.r``
            * **operation**: function applied on ``psp`` ("sum", "max", "min", "mean"). "sum" is the default.
        """
        # Create the description, but it will not be used for generation
        Projection.__init__(
            self, 
            pre,
            post,
            target,
            synapse = Synapse(psp=psp, operation=operation)
        )

        if not Global.config["paradigm"] == "openmp":
            Global._error('weightsharing is only implemented for the OpenMP paradigm.')
            exit(0)

        if not pre.neuron_type.type == 'rate':
            Global._error('weightsharing is only implemented for rate-coded populations.')
            exit(0)

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
    ### Connection methods
    ################################

    def convolve(self, weights, filter_or_kernel=True, padding=0.0, subsampling=None):
        """
        Builds the shared connection pattern that will perform a convolution of the weights kernel on the pre-synaptic population.

        Depending on the number of dimensions of the pre- and post-synaptic populations, as well as the kernel, the convolution will be  

        If the post-population has the same dimension, the convolution is regular.

        If the post-population has one dimension less than the pre-synaptic one, the last dimension of the kernel must match the last one of the pre-synaptic population. For example, filtering a N*M*3 image with a 3D filter (3 elements in the third dimension) results into a 2D population.

        If the kernel has less dimensions than the two populations, the number of neurons in the last dimension of the populations must be the same. The convolution will be calculated for each position in the last dimension (parallel convolution, useful if the pre-synaptic population is a stack of feature maps, for example).  

        If the kernel has more dimensions than the pre-synaptic population, this means a bank of different filters will be applied on the pre-synaptic population. Attention: the first index of ``weights`` corresponds to the different filters, while the result will be accessible in the last dimension of the post-synaptic population.

        Sub-sampling will be automatically performed according to the populations' geometry. If these geometries do not match, an error will be thrown. You can force sub-sampling by providing a list ``subsampling`` as argument, defining for each post-synaptic neuron the coordinates of the pre-synaptic neuron which will be the center of the filter/kernel. 


        *Parameters*:

            * **weights**: Numpy array or list of lists representing the matrix of weights for the filter/kernel.
            * **filter_or_kernel**: defines if the given weights are filter-based (dot-product) or kernel-based (convolution). TODO: explain. Default is kernel-based.
            * **padding**: value to be used for the rates outside the pre-synaptic population. If it is a floating value, the pre-synaptic population is virtually extended with this value above its boundaries. If it is equal to 'border', the values on the boundaries are repeated.
            * **subsampling**: list for each post-synaptic neuron of coordinates in the pre-synaptic population defining the center of the kernel/filter.
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

        if self.dim_post > 4:
            Global._error('Too many dimensions for the post-synaptic population (maximum 4).')
            exit(0)
        if self.dim_pre > 4:
            Global._error('Too many dimensions for the pre-synaptic population (maximum 4).')
            exit(0)
        if self.dim_kernel > 4:
            Global._error('Too many dimensions for the kernel (maximum 4).')
            exit(0)

        # Check if it is a bank of filters
        self.bank = self.dim_kernel > self.dim_pre
        if self.bank:
            return self._generate_bank()

        # Check if the last axes match for parallel convolution (e.g. 3-2-3)
        self.across_features = self.dim_kernel < self.dim_pre
        if self.across_features:
            if self.pre.geometry[-1] != self.post.geometry[-1]:
                Global._error('If the kernel has fewer dimensions than the two populations (parallel convolutions), these must have the same number of neurons in the last dimension(s).')
                exit(0)

        if self.dim_post < self.dim_pre: # OK, but check the last dimension of the kernel has the same size as the post-population
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                Global._error('If the post-synaptic population has less dimensions than the pre-synaptic one, the last dimension of the filter must be equal to the last of the pre-synaptic population.')
                exit(0)

        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates()

        # Filter definition
        filter_definition, filter_pyx_definition = self._filter_definition()

        # Convolve_code
        convolve_code, sum_code = self._generate_convolve_code()

        # Generate the code
        self._generate_omp(filter_definition, filter_pyx_definition, convolve_code, sum_code)

        # Finish building the synapses
        self._create()

    def pooling(self, extent=None, overlap=None):
        """
        Builds the shared connection pattern that will perform a pooling operation over the pre-synaptic population.

        Each post-synaptic neuron is associated to a region of the pre-synaptic one, over which the result of the operation on firing rates will be assigned to sum(target).

        If the SharedProjection does not define an operation, the default is "sum". If you want max-pooling, you should set it to "max". 
        
        *Parameters*:

            * **extent**: Extent of the pooling area expressed in the geometry of the pre-synaptic population. In each dimension, the product of this extent with the number of neurons in the post-synaptic population must be equal to the number of pre-synaptic neurons.

            * **overlap**: TODO, not implemented yet.
        """
        self.weights = []
        if extent == None: # compute the extent automatically
            if self.pre.dimension != self.post.dimension:
                Global._error('If you do not provide the extent parameter, the two populations must have the same dimensions.')
                exit(0)
            extent = list(self.pre.geometry)
            for dim in range(self.pre.dimension):
                extent[dim] /= self.post.geometry[dim]
                if self.pre.geometry[dim] != extent[dim] * self.post.geometry[dim] :
                    Global._error('Unable to compute the extent of the pooling area: the number of neurons do not match.')
                    exit(0)
        elif not isinstance(extent, tuple):
            Global._error('You must provide a tuple for the extent of the pooling operation.')
            exit(0)

        self.extent = list(extent)
        if len(self.extent) < self.pre.dimension:
            Global._error('You must provide a tuple for the extent of the pooling operation.')
            exit(0)


        # Check dimensions of populations 
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension

        if self.dim_post > 4:
            Global._error('Too many dimensions for the post-synaptic population (maximum 4).')
            exit(0)
        if self.dim_pre > 4:
            Global._error('Too many dimensions for the pre-synaptic population (maximum 4).')
            exit(0)

        # Generate the pre-synaptic coordinates
        self._generate_extent_coordinates()

        # Filter definition
        filter_definition, filter_pyx_definition = "", ""

        # Convolve_code
        convolve_code, sum_code = self._generate_pooling_code()

        # Generate the code
        self._generate_omp(filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=False)

        # Finish building the synapses
        self._create()


    def _generate_bank(self):

        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates_bank()

        # Filter definition
        filter_definition, filter_pyx_definition = self._filter_definition()

        # Convolve_code
        convolve_code, sum_code = self._generate_bank_code()

        # Generate the code
        self._generate_omp(filter_definition, filter_pyx_definition, convolve_code, sum_code)

        # Finish building the synapses
        self._create()

    ################################
    ### Generate centers
    ################################

    def _generate_extent_coordinates(self):
        " Returns a list for each post neuron of the corresponding top-left coordinates."

        # Generates coordinates TODO: Find a more robust way!
        coords = [[] for i in range(self.post.size)]
        if self.dim_pre == 1 :
            rk = 0
            for i in range(self.post.geometry[0]):
                coords[rk] = [i * self.extent[0]]
                rk += 1
        elif self.dim_pre == 2 :
            rk = 0
            for i in range(self.post.geometry[0]):
                if self.dim_post > 1:
                    for j in range(self.post.geometry[1]):
                        coords[rk] = [i * self.extent[0], j * self.extent[1]]
                        rk += 1
                else: # over the whole second axis
                    coords[rk] = [i * self.extent[0], 0]
                    rk += 1

        elif self.dim_pre == 3 :
            rk = 0
            for i in range(self.post.geometry[0]):
                for j in range(self.post.geometry[1]):
                    if self.dim_post > 2:
                        for k in range(self.post.geometry[2]):
                            coords[rk] = [i * self.extent[0], j * self.extent[1], k * self.extent[2]]
                            rk += 1
                    else: # over the whole third axis
                        coords[rk] = [i * self.extent[0], j * self.extent[1], 0]
                        rk += 1

        elif self.dim_pre == 4 : # TODO: post has less than 4 dimensions
            rk = 0
            for i in range(self.post.geometry[0]):
                for j in range(self.post.geometry[1]):
                    for k in range(self.post.geometry[2]):
                        for l in range(self.post.geometry[3]):
                            coords[rk] = [i * self.extent[0], j * self.extent[1], k * self.extent[2], l * self.extent[3]]
                            rk += 1
        # Save the result
        self.pre_coordinates = coords

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

        # Generates coordinates TODO: Find a more robust way!
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
        elif self.dim_pre == 4 :
            rk = 0
            for i in idx_range[0]:
                for j in idx_range[1]:
                    for k in idx_range[2]:
                        for l in idx_range[3]:
                            coords[rk] = [i, j, k, l]
                            rk += 1

        # print self.dim_pre
        # print idx_range
        # print coords

        # Save the result
        self.pre_coordinates = coords


    def _generate_pre_coordinates_bank(self):
        " Returns a list for each post neuron of the corresponding center coordinates, when the filter is a bank."

        self.nb_filters = self.weights.shape[0]
        self.dim_single_filter = self.weights.shape[1:]

        self.across_features = self.dim_single_filter < self.dim_pre

        if self.nb_filters != self.post.geometry[-1]:
            Global._error('The post-synaptic population must have', self.nb_filters, 'neurons in the last dimension.')
            exit(0)

        if self.dim_post -1 != self.dim_pre:
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                Global._error('The pre-synaptic population must have', self.dim_single_filter[-1], 'neurons in the last dimension.')
                exit(0)


        # Check if the list is already defined:
        if self.subsampling:
            try:
                shape = np.array(self.subsampling)
            except:
                Global._error('The sub-sampling list must have', self.post.size / self.post.geometry[-1], 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size/ self.post.geometry[-1], self.pre.dimension):
                Global._error('The sub-sampling list must have', self.post.size/ self.post.geometry[-1], 'elements of size', self.pre.dimension)
                return
            self.pre_coordinates = self.subsampling

        # Otherwise create it, possibly with sub-sampling
        coords = [[] for i in range(self.post.size)]

        # Compute pre-indices
        idx_range= []
        for dim in range(self.dim_pre):
            if dim < self.dim_post -1:
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


        # Generates coordinates TODO: Find a more robust way!
        if self.dim_pre == 1 :
            rk = 0
            for i in idx_range[0]:
                for d in range(self.nb_filters):
                    coords[rk] = [i, d]
                    rk += 1
        elif self.dim_pre == 2 :
            rk = 0
            for i in idx_range[0]:
                for j in idx_range[1]:
                    for d in range(self.nb_filters):
                        coords[rk] = [i, j, d ]
                        rk += 1
        elif self.dim_pre == 3 :
            rk = 0
            for i in idx_range[0]:
                for j in idx_range[1]:
                    for k in idx_range[2]:
                        for d in range(self.nb_filters):
                            coords[rk] = [i, j, k, d]
                            rk += 1
        elif self.dim_pre == 4 :
            rk = 0
            for i in idx_range[0]:
                for j in idx_range[1]:
                    for k in idx_range[2]:
                        for l in idx_range[3]:
                            for d in range(self.nb_filters):
                                coords[rk] = [i, j, k, l, d]
                                rk += 1

        # Save the result
        self.pre_coordinates = coords



    ################################
    ### Utilities
    ################################


    def _center_filter(self, i):
        return int(i/2) if i%2==1 else int(i/2)-1

    def _filter_definition(self):
        dim = self.dim_kernel
        cpp = 'double'
        pyx = 'double'
        for d in range(dim):
            cpp = 'std::vector< ' + cpp + ' >'
            pyx = 'vector[' + pyx + ']'
        cpp += ' w;'
        pyx += ' w'
        return cpp, pyx


    def _coordinates_to_rank(self, name, geometry):

        dim = len(geometry)

        txt = ""

        for d in range(dim):
            if txt == "" : # first coordinate is special
                txt = indices[0] + "_" + name
            else:
                txt = str(geometry[d]) + '*(' + txt + ') + ' + indices[d]  + '_' + name

        return txt

    def _generate_convolve_code(self):

        # Padding
        if isinstance(self.padding, str):
            padding = 0.0
        else:
            padding = self.padding

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse.operation

        # Main code
        code = """
        sum = 0.0;
"""

        # Generate for loops
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
            if operation in ['sum', 'mean']:
                if isinstance(self.padding, str): # 'border'
                        code += """
            if (%(index)s_pre < 0) %(index)s_pre = 0 ;
            if (%(index)s_pre > %(max_size)s) %(index)s_pre = %(max_size)s ;""" % { 'index': indices[dim], 'dim': dim, 'max_size': self.pre.geometry[dim] -1}
                else:
                    code += """
            if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                sum += %(padding)s;
                break;
            }""" % { 'index': indices[dim], 'padding': self.padding, 'max_size': self.pre.geometry[dim] -1}
            
            else: # min, max
                code += """
            if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                break;
            }""" % { 'index': indices[dim], 'max_size': self.pre.geometry[dim] -1}

        # Compute pre-synaptic rank
        code +="""
            rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}

        # Compute the increment
        psp = self.synapse.description['psp']['cpp']
        index = ""
        for dim in range(self.dim_kernel):
            index += '[' + indices[dim] + '_w]'
        increment = psp.replace('[i][j]', index)

        # Apply the operation
        if operation == "sum":
            code += """
            sum += %(increment)s""" % {'increment': increment}
        elif operation == "max":
            code += """
            double _psp = %(increment)s
            if(_psp > sum) sum = _psp;""" % {'increment': increment}
        elif operation == "min":
            code += """
            double _psp = %(increment)s
            if(_psp < sum) sum = _psp;""" % {'increment': increment}
        elif operation == "mean":
            code += """
            sum += %(increment)s""" % {'increment': increment}
        else:
            Global._error('Operation', operation, 'is not implemented yet for shared projections.')

        # Close for loops
        for dim in range(self.dim_kernel):
            code += """
        }""" 

        impl_code = code % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size
          }

        # sum code
        self.weights.size
        if operation == "mean":
            sum_code = """sum/%(filter_size)s""" % {'filter_size': self.weights.size}
        else:
            sum_code = "sum"

        return impl_code, sum_code

    def _generate_bank_code(self):

        # Padding
        if isinstance(self.padding, str):
            padding = 0.0
        else:
            padding = self.padding

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse.operation

        # Main code
        code = """
        sum = 0.0;
"""

        # Generate for loops
        for dim in range(self.dim_kernel-1):
            code += """        for(int %(index)s_w = 0; %(index)s_w < %(size)s;%(index)s_w++){
    """ % { 'index': indices[dim], 'size': self.weights.shape[dim+1]}

        # Compute indices
        for dim in range(self.dim_pre):
            if dim < self.dim_kernel:
                code += """
            int %(index)s_pre = coord[%(dim)s] %(operator)s (%(index)s_w - %(center)s);""" % { 'index': indices[dim], 'dim': dim, 'operator': '-' if self.filter_or_kernel else '+', 'center': self._center_filter(self.weights.shape[dim+1])}
            else:
                code += """
            int %(index)s_pre = coord[%(dim)s];""" % { 'index': indices[dim], 'dim': dim}


        # Check indices
        for dim in range(self.dim_kernel-1):
            if operation in ['sum', 'mean']:
                if isinstance(self.padding, str): # 'border'
                        code += """
            if (%(index)s_pre < 0) %(index)s_pre = 0 ;
            if (%(index)s_pre > %(max_size)s) %(index)s_pre = %(max_size)s ;""" % { 'index': indices[dim], 'dim': dim, 'max_size': self.pre.geometry[dim] -1}
                else:
                    code += """
            if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                sum += %(padding)s;
                break;
            }""" % { 'index': indices[dim], 'padding': self.padding, 'max_size': self.pre.geometry[dim] -1}
            
            else: # min, max
                code += """
            if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                break;
            }""" % { 'index': indices[dim], 'max_size': self.pre.geometry[dim] -1}

        # Compute pre-synaptic rank
        code +="""
            rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}

        # Compute the increment
        psp = self.synapse.description['psp']['cpp']
        index = "[coord["+str(self.dim_pre)+"]]"
        for dim in range(self.dim_kernel-1):
            index += '[' + indices[dim] + '_w]'
        increment = psp.replace('[i][j]', index)

        # Apply the operation
        if operation == "sum":
            code += """
            sum += %(increment)s""" % {'increment': increment}
        elif operation == "max":
            code += """
            double _psp = %(increment)s
            if(_psp > sum) sum = _psp;""" % {'increment': increment}
        elif operation == "min":
            code += """
            double _psp = %(increment)s
            if(_psp < sum) sum = _psp;""" % {'increment': increment}
        elif operation == "mean":
            code += """
            sum += %(increment)s""" % {'increment': increment}
        else:
            Global._error('Operation', operation, 'is not implemented yet for shared projections.')

        # Close for loops
        for dim in range(self.dim_kernel-1):
            code += """
        }""" 

        impl_code = code % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size
          }

        # sum code
        self.weights.size
        if operation == "mean":
            sum_code = """sum/%(filter_size)s""" % {'filter_size': self.weights.size}
        else:
            sum_code = "sum"

        return impl_code, sum_code

    def _generate_pooling_code(self):

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse.operation

        # Main code
        code = """
        sum = 0.0;
"""

        # Generate for loops
        for dim in range(self.dim_pre):
            if self.extent[dim] >1:
                code += """        for(int %(index)s_w = 0; %(index)s_w < %(size)s; %(index)s_w++){
    """ % { 'index': indices[dim], 'size': self.extent[dim]}

        # Compute indices
        for dim in range(self.dim_pre):
            if self.extent[dim] >1:
                code += """
            int %(index)s_pre = coord[%(dim)s] + %(index)s_w;""" % { 'index': indices[dim], 'dim': dim}
            else:
                code += """
            int %(index)s_pre = coord[%(dim)s];""" % { 'index': indices[dim], 'dim': dim}

        # Check indices
        for dim in range(self.dim_pre):
            code += """
            if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                break;
            }""" % { 'index': indices[dim], 'max_size': self.pre.geometry[dim] -1}

        # Compute pre-synaptic rank
        code += """
            rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}

        # Apply the operation
        if operation == "sum":
            code += """
            sum += pop%(id_pre)s.r[rk_pre];"""
        elif operation == "max":
            code += """
            double _psp = pop%(id_pre)s.r[rk_pre];
            if(_psp > sum) sum = _psp;"""
        elif operation == "min":
            code += """
            double _psp = pop%(id_pre)s.r[rk_pre];
            if(_psp < sum) sum = _psp;"""
        elif operation == "mean":
            code += """
            sum += pop%(id_pre)s.r[rk_pre];"""
        else:
            Global._error('Operation', operation, 'is not implemented yet for shared projections with pooling.')

        # Close for loops
        for dim in range(self.dim_pre):
            if self.extent[dim] >1:
                code += """
        }""" 

        impl_code = code % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size
          }  

        if operation == "mean":
            size = 1
            for dim in range(self.pre.dimension):
                size *= self.extent[dim]
            sum_code = "sum/"+ str(size)
        else:
            sum_code = "sum"

        return impl_code, sum_code

    ################################
    ### Code generation
    ################################

    def _generate_omp(self, filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=True):

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
        proj_class = """
# Shared projection %(name_pre)s -> %(name_post)s, target %(target)s
cdef class proj%(id_proj)s_wrapper :
    def __cinit__(self, weights, coords):
        proj%(id_proj)s.post_rank = list(range(%(size_post)s))
        proj%(id_proj)s.pre_coords = coords""" + ("""
        proj%(id_proj)s.w = weights""" if kernel else "" )+ """

    property size:
        def __get__(self):
            return %(size_post)s
""" + ( """
    def nb_synapses(self, int n):
        return %(size_post)s * proj%(id_proj)s.w.size()
""" if kernel else """
    def nb_synapses(self, int n):
        return 0
""") + """

    def post_rank(self):
        return proj%(id_proj)s.post_rank
    def pre_rank(self, int n):
        return proj%(id_proj)s.pre_coords[n]

""" 
        if kernel: # not needed for max pooling
            proj_class += """
    # Kernel
    def get_w(self):
        return proj%(id_proj)s.w
    def set_w(self, value):
        proj%(id_proj)s.w = value

"""
        self.generator['omp']['pyx_proj_class'] = proj_class % { 'id_proj': self.id, 'target': self.target, 
            'name_pre': self.pre.name, 
            'name_post': self.post.name, 'size_post': self.post.size,
        }

        # No need to initialize anything (no recordable variable, no learning)
        self.generator['omp']['body_proj_init'] = " "

        # Compute sum
        wsum =  """
    // Shared proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s. 
    #pragma omp parallel for private(sum, rk_pre)
    for(int i = 0; i < %(size_post)s; i++){
        std::vector<int> coord = proj%(id_proj)s.pre_coords[i];
""" + convolve_code + """
        pop%(id_post)s._sum_%(target)s[i] += """ + sum_code + """;
    }
""" 

        self.generator['omp']['body_compute_psp'] = wsum % {'id_proj': self.id, 
            'target': self.target,  
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size, 
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size
          }




