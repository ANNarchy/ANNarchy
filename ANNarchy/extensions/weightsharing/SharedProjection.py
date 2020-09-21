from ANNarchy.core.Projection import Projection
from ANNarchy.core.Synapse import Synapse
import ANNarchy.core.Global as Global

import numpy as np
from ANNarchy.generator.Utils import tabify # _generate_bank_code, _generate_convolve_code

# Indices used for each dimension
indices = ['i', 'j', 'k', 'l', 'm', 'n']


###############################
### Shared synapse for report()
###############################
class SharedSynapse(Synapse):
    # For reporting
    _instantiated = []
    def __init__(self, psp, operation):
        Synapse.__init__(self,
            psp=psp, operation=operation,
            name="Shared Weight",
            description="Weight shared over all synapses of the projection."
        )
        # For reporting
        self._instantiated.append(True)


###############################
### Shared projection
###############################
class SharedProjection(Projection):
    """
    """
    def __init__(self, pre, post, target, psp="w * pre.r", operation="sum", name=None, copied=False):
        """
        Projection based on shared weights: each post-synaptic neuron uses the same weights, so they need to be instantiated only once to save memory.

        Learning is not possible for now. The ``synapse`` argument is removed, replaced by a single ``psp`` argument to modified what is summed and ``operation`` to replace the summation operation by max-pooling or similar..

        :param pre: pre-synaptic population (either its name or a ``Population`` object).
        :param post: post-synaptic population (either its name or a ``Population`` object).
        :param target: type of the connection.
        :param psp: function to be summed. By default: ``w * pre.r``
        :param operation: function applied on ``psp`` ("sum", "max", "min", "mean"). "sum" is the default.
        """
        self.psp_init = psp
        self.operation = operation
        # Create the description, but it will not be used for generation
        Projection.__init__(
            self,
            pre=pre,
            post=post,
            target=target,
            synapse = SharedSynapse(psp=psp, operation=operation),
            name=name,
            copied=copied
        )

        self._omp_config['psp_schedule'] = 'schedule(dynamic)'
        if not Global.config["paradigm"] == "openmp":
            Global._error('SharedProjection: Weight sharing is only implemented for the OpenMP paradigm.')

        if not pre.neuron_type.type == 'rate':
            Global._error('SharedProjection: Weight sharing is only implemented for rate-coded populations.')

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks.  Internal use only."
        return SharedProjection(pre=self.pre, post=self.post, target=self.target, psp=self.psp_init, operation=self.operation, name=self.name, copied=True)

    def _create(self):
        # create fake LIL object, just for compilation.
        try:
            from ANNarchy.core.cython_ext.Connector import LILConnectivity
        except Exception as e:
            Global._print(e)
            Global._error('ANNarchy was not successfully installed.')

        lil = LILConnectivity()
        lil.max_delay = self.delays
        lil.uniform_delay = self.delays
        self.connector_name = "Shared weights"
        self.connector_description = "Shared weights"
        self._store_connectivity(self._load_from_lil, (lil, ), self.delays)

    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """
        if not self._connection_method:
            Global._error('SharedProjection: The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')


        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights, self.pre_coordinates)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Set delays after instantiation
        if self.delays > 0.0:
            self.cyInstance.set_delay(self.delays/Global.config['dt'])

    def center(self, *args, **kwds):
        """
        Returns the coordinates in the pre-synaptic population of the center of the kernel corresponding to the post-synaptic with the given rank or coordinates.

        :param rank: rank or coordinates of the post-synaptic neuron. If only one argument is given, it is a rank. If it is a tuple, it is coordinates.
        """
        if len(args) == 1:
            rank =  args[0]
        else:
            rank = self.post.rank_from_coordinates(args)


        if self.initialized:
            return tuple(self.cyInstance.pre_rank(rank))
        else:
            return tuple(self.pre_coordinates[rank])

    def _data(self):
        "Disable saving."
        desc = {}
        desc['post_ranks'] = self.post_ranks
        desc['attributes'] = self.attributes
        desc['parameters'] = self.parameters
        desc['variables'] = self.variables

        desc['dendrites'] = []
        desc['number_of_synapses'] = 0
        return desc

    ################################
    ### Connection methods
    ################################

    def convolve(self, weights, delays=0.0, method='convolution', keep_last_dimension=False, multiple=False, padding=0.0, subsampling=None):
        """
        Builds the shared connection pattern that will perform a convolution of the weights kernel on the pre-synaptic population.

        Depending on the number of dimensions of the pre- and post-synaptic populations, as well as the kernel, the convolution can be implemented differentely.

        * If the pre- and post-populations have the same dimension as the kernel, the convolution is regular.

        * If the post-population has one dimension less than the pre-synaptic one, the last dimension of the kernel must match the last one of the pre-synaptic population. For example, filtering a N*M*3 image with a 3D filter (3 elements in the third dimension) results into a 2D population.

        * If the kernel has less dimensions than the two populations, the number of neurons in the last dimension of the populations must be the same. The convolution will be calculated for each position in the last dimension (parallel convolution, useful if the pre-synaptic population is a stack of feature maps, for example). In this case, you must set ``keep_last_dimension`` to True.

        * If the kernel has more dimensions than the pre-synaptic population, this means a bank of different filters will be applied on the pre-synaptic population. Attention: the first index of ``weights`` corresponds to the different filters, while the result will be accessible in the last dimension of the post-synaptic population. You must set the ``multiple`` argument to True.

        Sub-sampling will be automatically performed according to the populations' geometry. If these geometries do not match, an error will be thrown. You can force sub-sampling by providing a list ``subsampling`` as argument, defining for each post-synaptic neuron the coordinates of the pre-synaptic neuron which will be the center of the filter/kernel.



        :param weights: Numpy array or list of lists representing the matrix of weights for the filter/kernel.
        :param delays: delay in synaptic transmission (default: dt). Can only be the same value for all neurons.
        :param method: defines if the given weights are filter-based (dot-product between the filter and sub-region: 'filter') or kernel-based (regular convolution: 'convolution').. Default: 'convolution'.
        :param keep_last_dimension: defines if the last dimension of the pre- and post-synaptic will be convolved in parallel. The weights matrix must have one dimension less than the pre-synaptic population, and the number of neurons in the last dimension of the pre- and post-synaptic populations must match. Default: False.
        :param multiple: defines if the weights matrix describes a bank of filters which have to applied in parallel. The weights matrix must have one dimension more than the pre-synaptic populations, and the number of neurons in the last dimension of the post-synaptic population must be equal to the number of filters.
        :param padding: value to be used for the rates outside the pre-synaptic population. If it is a floating value, the pre-synaptic population is virtually extended with this value above its boundaries. If it is equal to 'border', the values on the boundaries are repeated. Default: 0.0.
        :param subsampling: list for each post-synaptic neuron of coordinates in the pre-synaptic population defining the center of the kernel/filter. Default: None.
        """
        self._operation_type = 'convolve'
        self.method = method
        self.keep_last_dimension = keep_last_dimension
        self.multiple = multiple
        self.padding = padding
        self.subsampling = subsampling

        # Process the weights
        if isinstance(weights, list):
            self.weights = np.array(weights)
        else:
            self.weights = weights

        # Process the delays
        self.delays = delays
        if not isinstance(delays, (int, float)):
            Global._error('Shared projections can only have uniform delays.')

        # Check dimensions of populations and weight matrix
        self.dim_kernel = self.weights.ndim
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension

        if self.dim_post > 4:
            Global._error('SharedProjection: Too many dimensions for the post-synaptic population (maximum 4).')

        if self.dim_pre > 4:
            Global._error('SharedProjection: Too many dimensions for the pre-synaptic population (maximum 4).')

        if self.dim_kernel > 5  or (not self.multiple and self.dim_kernel > 4):
            Global._error('SharedProjection: Too many dimensions for the kernel (maximum 4).')

        # Check if the last axes match for parallel convolution (e.g. 3-2-3)
        if self.dim_kernel < self.dim_pre:
            if not self.keep_last_dimension:
                Global._error('SharedProjection: If the kernel has less dimensions than the pre-synaptic population, you need to set the flag keep_last_dimension to True.')

            if self.pre.geometry[-1] != self.post.geometry[-1]:
                Global._error('SharedProjection: If the kernel has fewer dimensions than the two populations (keep_last_dimension=True), these must have the same number of neurons in the last dimension.')

        # If the last dim of the kernel matches the last dim of the pre-pop, the last pop can have one dimension less.
        if self.dim_post < self.dim_pre: # OK, but check the last dimension of the kernel has the same size as the post-population
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                Global._error('SharedProjection: If the post-synaptic population has less dimensions than the pre-synaptic one, the last dimension of the filter must be equal to the last of the pre-synaptic population.')

        # Check if it is a bank of filters
        if self.dim_kernel > self.dim_pre:
            if not self.multiple:
                Global._error('SharedProjection: If the kernel has more dimensions than the pre-synaptic population, you need to set the flag multiple to True.')

            # if self.dim_kernel > self.dim_post:
            #     if not self.keep_last_dimension:
            #         Global._error('If the kernel has more dimensions than the post-synaptic population, you need to set the flag keep_last_dimension to True.')
            #
            if self.weights.shape[0] != self.post.geometry[-1]:
                Global._error('SharedProjection: For multiple filters, the last dimension of the post-synaptic population must have as many neurons as there are filters.')


        # Generate the pre-synaptic coordinates
        if not self.multiple:
            self._generate_pre_coordinates()
        else:
            self._generate_pre_coordinates_bank()

        # Finish building the synapses
        self._create()
        return self


    def pooling(self, delays=0.0, extent=None, overlap=None):
        """
        Builds the shared connection pattern that will perform a pooling operation over the pre-synaptic population.

        Each post-synaptic neuron is associated to a region of the pre-synaptic one, over which the result of the operation on firing rates will be assigned to sum(target).

        If the SharedProjection does not define an operation, the default is "sum". If you want max-pooling, you should set it to "max".

        :param delays: delays (in ms) in synaptic transmission. Must be a single value for all neurons.
        :param extent: Extent of the pooling area expressed in the geometry of the pre-synaptic population. In each dimension, the product of this extent with the number of neurons in the post-synaptic population must be equal to the number of pre-synaptic neurons.
        :param overlap: TODO, not implemented yet.
        """
        self._operation_type = 'pooling'

        self.weights = []
        if extent is None: # compute the extent automatically
            if self.pre.dimension != self.post.dimension:
                Global._error('SharedProjection: If you do not provide the extent parameter, the two populations must have the same dimensions.')

            extent = list(self.pre.geometry)
            for dim in range(self.pre.dimension):
                extent[dim] /= self.post.geometry[dim]
                if self.pre.geometry[dim] != extent[dim] * self.post.geometry[dim] :
                    Global._error('SharedProjection: Unable to compute the extent of the pooling area: the number of neurons do not match.')

        elif not isinstance(extent, tuple):
            Global._error('SharedProjection: You must provide a tuple for the extent of the pooling operation.')


        self.extent = list(extent)
        if len(self.extent) < self.pre.dimension:
            Global._error('SharedProjection: You must provide a tuple for the extent of the pooling operation.')


        # Process the delays
        self.delays = delays

        # Change the psp by default
        if self.synapse_type.description['raw_psp'] == "w * pre.r":
            self.synapse_type.description['psp']['cpp'] = "%(pre_prefix)sr%(pre_index)s"

        # Check dimensions of populations
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension

        if self.dim_post > 4:
            Global._error('SharedProjection: Too many dimensions for the post-synaptic population (maximum 4).')

        if self.dim_pre > 4:
            Global._error('SharedProjection: Too many dimensions for the pre-synaptic population (maximum 4).')


        # Generate the pre-synaptic coordinates
        self._generate_extent_coordinates()

        # Finish building the synapses
        self._create()
        return self

    def copy(self, projection):
        """
        Creates a virtual connection pattern reusing the weights and delays of an already-defined projection.

        Although the original projection can be learnable, this one can not. Changes in the original weights will be reflected in this projection. The only possible modifications are ``psp`` and ``operation``.

        The pre- and post-synaptic populations of each projection must have the same geometry.

        :param projection: the projection to reuse.
        """
        self._operation_type = 'copy'
        self.projection = projection

        if not isinstance(self.projection, Projection):
            Global._error('SharedProjection: You must provide an existing projection to copy().')


        if isinstance(self.projection, SharedProjection):
            Global._error('SharedProjection: You can only copy regular projections, not shared projections.')


        if not self.pre.geometry == self.projection.pre.geometry or not self.post.geometry == self.projection.post.geometry:
            Global._error('SharedProjection: When copying a projection, the geometries must be the same.')


        # Dummy weights
        self.weights = None
        self.pre_coordinates = []

        # Finish building the synapses
        self._create()
        return self

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
                shape = np.array(self.subsampling).shape
            except:
                Global._error('SharedProjection: The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size, self.pre.dimension):
                Global._error('SharedProjection: The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
                return
            self.pre_coordinates = self.subsampling
            return

        # Otherwise create it, possibly with sub-sampling
        coords = [[] for i in range(self.post.size)]

        # Compute pre-indices
        idx_range= []
        for dim in range(self.dim_pre):
            if dim < self.dim_post:
                pre_size = int(self.pre.geometry[dim])
                post_size = int(self.post.geometry[dim])
                sample = int(pre_size/post_size)
                if post_size * sample != pre_size:
                    Global._error('SharedProjection: The pre-synaptic dimensions must be a multiple of the post-synaptic ones for down-sampling to work.')

                idx_range.append([int((sample-1)/2) + sample * i for i in range(post_size)])
            else: # extra dimension
                if self.keep_last_dimension:
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

        # Save the result
        self.pre_coordinates = coords


    def _generate_pre_coordinates_bank(self):
        " Returns a list for each post neuron of the corresponding center coordinates, when the filter is a bank."

        self.nb_filters = self.weights.shape[0]
        self.dim_single_filter = self.weights.shape[1:]

        # Check if the list is already defined:
        if self.subsampling:
            try:
                shape = np.array(self.subsampling).shape
            except:
                Global._error('SharedProjection: The sub-sampling list must have', self.post.size / self.post.geometry[-1], 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size/ self.post.geometry[-1], self.pre.dimension):
                Global._error('SharedProjection: The sub-sampling list must have', self.post.size/ self.post.geometry[-1], 'elements of size', self.pre.dimension)
                return
            self.pre_coordinates = [c + [d] for c in self.subsampling  for d  in range(self.nb_filters)]
            return

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
                    Global._error('SharedProjection: The pre-synaptic dimensions must be a multiple of the post-synaptic ones for down-sampling to work.')

                idx_range.append([int((sample-1)/2) + sample * i for i in range(post_size)])
            else: # extra dimension
                if self.keep_last_dimension:
                    idx_range.append(range(self.post.geometry[dim]))
                else:
                    idx_range.append([self._center_filter(self.weights.shape[dim+1])])


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
        cpp = Global.config['precision']
        pyx = Global.config['precision']
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

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse_type.operation

        # Main code
        code = tabify("sum = 0.0;", 3)

        # Generate for loops
        for dim in range(self.dim_kernel):
            if dim == self.dim_kernel-1:
                inner_idx = ""
                for i in range(self.dim_kernel-1):
                    inner_idx += "["+indices[i]+"_w]"
                code += "auto inner_line = w"+inner_idx+".data();\n"

            code += tabify("""
            for(int %(index)s_w = 0; %(index)s_w < %(size)s;%(index)s_w++) {
            """ % { 'index': indices[dim], 'size': self.weights.shape[dim]}, dim)

            # Compute indices
            if dim < self.dim_kernel:
                code += tabify("""int %(index)s_pre = coord[%(dim)s] %(operator)s (%(index)s_w - %(center)s);""" % { 'id_proj': self.id, 'index': indices[dim], 'dim': dim, 'operator': '-' if self.method=='convolution' else '+', 'center': self._center_filter(self.weights.shape[dim])}, 1)
            else:
                code += tabify("""int %(index)s_pre = coord[%(dim)s];""" % { 'id_proj': self.id, 'index': indices[dim], 'dim': dim}, 1)

            # Check indices
            if operation in ['sum', 'mean']:
                if isinstance(self.padding, str): # 'border'
                        code += tabify("""
                if (%(index)s_pre < 0) %(index)s_pre = 0 ;
                if (%(index)s_pre > %(max_size)s) %(index)s_pre = %(max_size)s ;
                """ % { 'index': indices[dim], 'dim': dim, 'max_size': self.pre.geometry[dim] -1}, dim)
                else:
                    code += tabify("""
                if ((%(index)s_pre < 0) || (%(index)s_pre > %(max_size)s)){
                    sum += %(padding)s;
                    continue;
                }
                """ % { 'index': indices[dim], 'padding': self.padding, 'max_size': self.pre.geometry[dim] -1}, dim)

            else: # min, max
                code += """
                if ((%(index)s_pre < 0) || (%(index)s_pre > %(max_size)s)) {
                    continue;
                }
                """ % { 'index': indices[dim], 'max_size': self.pre.geometry[dim] -1}

        # Compute pre-synaptic rank
        code += tabify("""
                rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}, dim)

        # Compute the increment
        index = ""
        for dim in range(self.dim_kernel):
            index += '[' + indices[dim] + '_w]'

        increment = self.synapse_type.description['psp']['cpp'] % {
            'id_pre': self.pre.id,
            'id_post': self.post.id,
            'local_index': index,
            'global_index': '[i]',
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]',
            'pre_prefix': 'pop'+str(self.pre.id)+'.',
            'post_prefix': 'pop'+str(self.post.id)+'.'
        }

        # Delays
        if self.delays > Global.config['dt']:
            increment = increment.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                'delayed_r[rk_pre]'
            )

        # Apply the operation
        if operation == "sum":
            code += tabify("""
                sum += %(increment)s""" % {'increment': increment.replace('w'+inner_idx, 'inner_line')}, dim)
        elif operation == "max":
            code += tabify("""
                %(float_prec)s _psp = %(increment)s
                if(_psp > sum) sum = _psp;""" % {'increment': increment, 'float_prec': Global.config['precision']}, dim)
        elif operation == "min":
            code += tabify("""
                %(float_prec)s _psp = %(increment)s
                if(_psp < sum) sum = _psp;""" % {'increment': increment, 'float_prec': Global.config['precision']}, dim)
        elif operation == "mean":
            code += tabify("""
                sum += %(increment)s""" % {'increment': increment}, dim)
        else:
            Global._error('SharedProjection: Operation', operation, 'is not implemented yet for shared projections.')

        # Close for loops
        for dim in range(self.dim_kernel):
            code += tabify("""
            }""", self.dim_kernel-1-dim)

        impl_code = code % {'id_proj': self.id,
            'target': self.target,
            'id_pre': self.pre.id,
            'name_pre': self.pre.name,
            'size_pre': self.pre.size,
            'id_post': self.post.id,
            'name_post': self.post.name,
            'size_post': self.post.size
          }

        # sum code
        self.weights.size
        if operation == "mean":
            sum_code = """sum/%(filter_size)s""" % {'filter_size': self.weights.size}
        else:
            sum_code = "sum"

        return impl_code, sum_code

    def _generate_bank_code(self):

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse_type.operation

        # Main code
        code = tabify("sum = 0.0;", 3)

        # Generate for loops
        for dim in range(self.dim_kernel-1):
            code += tabify("""
            for(int %(index)s_w = 0; %(index)s_w < %(size)s;%(index)s_w++) {
            """ % { 'index': indices[dim], 'size': self.weights.shape[dim+1]}, dim)

            # Compute indices
            if dim < self.dim_kernel:
                code += tabify("""int %(index)s_pre = coord[%(dim)s] %(operator)s (%(index)s_w - %(center)s);""" % { 'id_proj': self.id, 'index': indices[dim], 'dim': dim, 'operator': '-' if self.method=='convolution' else '+', 'center': self._center_filter(self.weights.shape[dim+1])}, 1)
            else:
                code += tabify("""int %(index)s_pre = coord[%(dim)s];""" % { 'id_proj': self.id, 'index': indices[dim], 'dim': dim}, 1)

            # Check indices
            if operation in ['sum', 'mean']:
                if isinstance(self.padding, str): # 'border'
                    code += tabify("""
            if (%(index)s_pre < 0) %(index)s_pre = 0 ;
            if (%(index)s_pre > %(max_size)s) %(index)s_pre = %(max_size)s ;
            """ % { 'index': indices[dim], 'dim': dim, 'max_size': self.pre.geometry[dim] -1}, 1+dim)
                else:
                    code += tabify("""
            if ((%(index)s_pre < 0) || (%(index)s_pre > %(max_size)s)) {
                sum += %(padding)s;
                continue;
            }
            """ % { 'index': indices[dim], 'padding': self.padding, 'max_size': self.pre.geometry[dim] -1}, 1+dim)

            else: # min, max
                code += tabify("""
            if ((%(index)s_pre < 0) || (%(index)s_pre > %(max_size)s)){
                continue;
            }
            """ % { 'index': indices[dim], 'max_size': self.pre.geometry[dim] -1}, 1+dim)

        # Compute pre-synaptic rank
        code +=tabify("""
            rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}, 1+dim)

        # Compute the increment
        index = "[coord["+str(self.dim_pre)+"]]"
        for dim in range(self.dim_kernel-1):
            index += '[' + indices[dim] + '_w]'

        increment = self.synapse_type.description['psp']['cpp'] % {
            'id_pre': self.pre.id,
            'id_post': self.post.id,
            'local_index': index,
            'global_index': '[i]',
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]',
            'pre_prefix': 'pop'+str(self.pre.id)+'.',
            'post_prefix': 'pop'+str(self.post.id)+'.'}

        # Delays
        if self.delays > Global.config['dt']:
            increment = increment.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                'delayed_r[rk_pre]'
            )

        # Apply the operation
        if operation == "sum":
            code += tabify("""
            sum += %(increment)s""" % {'increment': increment}, 1+dim)
        elif operation == "max":
            code += tabify("""
            %(float_prec)s _psp = %(increment)s
            if(_psp > sum) sum = _psp;""" % {'increment': increment, 'float_prec': Global.config['precision']}, 1+dim)
        elif operation == "min":
            code += tabify("""
            %(float_prec)s _psp = %(increment)s
            if(_psp < sum) sum = _psp;""" % {'increment': increment, 'float_prec': Global.config['precision']}, 1+dim)
        elif operation == "mean":
            code += tabify("""
            sum += %(increment)s""" % {'increment': increment}, 1+dim)
        else:
            Global._error('SharedProjection: Operation', operation, 'is not implemented yet for shared projections.')

        # Close for loops
        for dim in range(self.dim_kernel-1):
            code += tabify("""
        }""", self.dim_kernel-1-dim)

        impl_code = code % {'id_proj': self.id,
            'target': self.target,
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size,
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size
        }

        # sum code
        if operation == "mean":
            sum_code = """sum/%(filter_size)s""" % {'filter_size': self.weights.size}
        else:
            sum_code = "sum"

        return impl_code, sum_code

    def _generate_pooling_code(self):

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse_type.operation

        # Main code
        code = """
            sum = 0.0;
"""

        # Generate for loops
        for dim in range(self.dim_pre):
            if self.extent[dim] >1:
                code += """
            for(int %(index)s_w = 0; %(index)s_w < %(size)s; %(index)s_w++){
    """ % { 'index': indices[dim], 'size': self.extent[dim]}

        # Compute indices
        for dim in range(self.dim_pre):
            if self.extent[dim] >1:
                code += """
                int %(index)s_pre = coord[%(dim)s] + %(index)s_w;""" % { 'id_proj': self.id, 'index': indices[dim], 'dim': dim}
            else:
                code += """
                int %(index)s_pre = coord[%(dim)s];""" % { 'id_proj': self.id, 'index': indices[dim], 'dim': dim}

        # Check indices
        for dim in range(self.dim_pre):
            code += """
                if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                    continue;
                }""" % { 'index': indices[dim], 'max_size': self.pre.geometry[dim] -1}

        # Compute pre-synaptic rank
        code += """
                rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}

        # Compute the value to pool
        psp = self.synapse_type.description['psp']['cpp'] % {
            'id_pre': self.pre.id,
            'id_post': self.post.id,
            'local_index':'[i][j]',
            'global_index': '[i]',
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]',
            'pre_prefix': 'pop'+str(self.pre.id)+'.',
            'post_prefix': 'pop'+str(self.post.id)+'.'
            }

        # Delays
        if self.delays > Global.config['dt']:
            psp = psp.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                # TODO HD: wouldn't it be much better to reduce delay globaly, instead of the substraction here???
                'pop%(id_pre)s._delayed_r[delay-1][rk_pre]' % {'id_pre': self.pre.id}
            )

        # Apply the operation
        if operation == "sum":
            code += """
                sum += %(psp)s;"""
        elif operation == "max":
            code += """
                %(float_prec)s _psp = %(psp)s;
                if(_psp > sum) sum = _psp;"""
        elif operation == "min":
            code += """
                %(float_prec)s _psp = %(psp)s;
                if(_psp < sum) sum = _psp;"""
        elif operation == "mean":
            code += """
                sum += %(psp)s;"""
        else:
            Global._error('SharedProjection: Operation', operation, 'is not implemented yet for shared projections with pooling.')

        # Close for loops
        for dim in range(self.dim_pre):
            if self.extent[dim] >1:
                code += """
            }"""

        impl_code = code % {'id_proj': self.id,
            'target': self.target,
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size,
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'psp': psp,
            'float_prec': Global.config['precision']
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

    def _generate(self):

        if self._operation_type == 'convolve':
            # Filter definition
            filter_definition, filter_pyx_definition = self._filter_definition()

            # Convolve_code
            if not self.multiple:
                convolve_code, sum_code = self._generate_convolve_code()
            else:
                convolve_code, sum_code = self._generate_bank_code()

            # Generate the code
            self._generate_omp(filter_definition, filter_pyx_definition, convolve_code, sum_code)
            #self._generate_cuda(filter_definition, filter_pyx_definition, convolve_code, sum_code)

        elif self._operation_type == 'pooling':
            # Filter definition
            filter_definition, filter_pyx_definition = "",""

            # Convolve_code
            convolve_code, sum_code = self._generate_pooling_code()

            # Generate the code
            self._generate_omp(filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=False)

        elif self._operation_type == 'copy':

            # Generate the code
            self._generate_copy()

    def _generate_omp(self, filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=True):
        # Specific template for generation
        self._specific_template = {
            # Declare the connectivity matrix
            'declare_connectivity_matrix': """
    std::vector<int> post_rank;
    std::vector< std::vector<int> > pre_rank;
    """ + filter_definition.strip(),

            # Accessors for the connectivity matrix
            'access_connectivity_matrix': """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }
""" ,

            # Export the connectivity matrix
            'export_connectivity': """
        # Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
""",

            # Arguments to the wrapper constructor
            'wrapper_args': "weights, coords",

            # Delays
            'wrapper_init_delay': "",

            # Initialize the wrapper connectivity matrix
            'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_pre_rank(coords)
""" % {'id_proj': self.id, 'size_post': self.post.size},

            # Wrapper access to connectivity matrix
            'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def pre_rank(self, int n):
        return proj%(id_proj)s.get_pre_rank()
            """ % {'id_proj': self.id},

            # Wrapper access to variables
            'wrapper_access_parameters_variables' : "",

            # Variables for the psp code
            'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=0.0;""" % {'float_prec': Global.config['precision']}
        }

        # Kernel-based method: specify w with the correct dimension
        if kernel:
            self._specific_template['access_connectivity_matrix'] += """
    // Local parameter w
    %(type_w)s get_w() { return w; }
    void set_w(%(type_w)s value) { w = value; }
""" % {'type_w': filter_definition.replace(' w;', '')}
            self._specific_template['export_connectivity'] += """
        # Local variable w
        %(type_w)s get_w()
        void set_w(%(type_w)s)
""" % {'type_w': filter_pyx_definition.replace(' w', '')}
            self._specific_template['wrapper_init_connectivity'] += """
        proj%(id_proj)s.set_w(weights)
""" % {'id_proj': self.id}

            self._specific_template['wrapper_access_connectivity'] += """
    # Local variable w
    def get_w(self):
        return proj%(id_proj)s.get_w()
    def set_w(self, value):
        proj%(id_proj)s.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj%(id_proj)s.get_w()
    def set_dendrite_w(self, int rank, value):
        proj%(id_proj)s.set_w(value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return 0.0
    def set_synapse_w(self, int rank_post, int rank_pre, %(float_prec)s value):
        pass
""" % {'id_proj': self.id, 'float_prec': Global.config['precision']}

        # Override the monitor to avoid recording the weights
        self._specific_template['monitor_class'] = ""

        self._specific_template['monitor_export'] = ""

        self._specific_template['monitor_wrapper'] = ""

        # OMP code
        omp_code = ""
        if Global.config['num_threads'] > 1:
            omp_code = """
        #pragma omp parallel for private(sum, rk_pre, coord) %(psp_schedule)s""" % {'psp_schedule': "" if not 'psp_schedule' in self._omp_config.keys() else self._omp_config['psp_schedule']}

        # HD ( 16.10.2015 ):
        # pre-load delayed firing rate in a local array, so we
        # prevent multiple accesses to pop%(id_pre)s._delayed_r[delay-1]
        # wheareas delay is set available as variable
        # TODO HD: wouldn't it be much better to reduce delay globaly, instead of the substraction here???
        if self.delays > Global.config['dt']:
            pre_load_r = """
        // pre-load delayed firing rate
        auto delayed_r = pop%(id_pre)s._delayed_r[delay-1];
        """% {'id_pre': self.pre.id}
        else:
            pre_load_r = ""

        # Compute sum
        wsum =  """
        if ( _transmission && pop%(id_pre)s._active ) {
        std::vector<int> coord;
""" + pre_load_r + """
        %(omp_code)s
        for(int i = 0; i < %(size_post)s; i++){
            coord = pre_rank[i];
""" + convolve_code + """
            pop%(id_post)s._sum_%(target)s[i] += """ + sum_code + """;
        } // for
        } // if
"""

        self._specific_template['psp_code'] = wsum % \
        {   'id_proj': self.id,
            'target': self.target,
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size,
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'omp_code': omp_code,
            'convolve_code': convolve_code
        }

        # override size in bytes calculation
        self._specific_template['size_in_bytes'] = "//TODO:\n"

    def _generate_cuda(self, filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=True):
        "TODO"

        # Template
        self._specific_template = {}

    def _generate_copy(self):

        # Specific template for generation
        self._specific_template = {
            # Declare the connectivity matrix
            'declare_connectivity_matrix': "",
            # Accessors for the connectivity matrix
            'access_connectivity_matrix': "",
            # No initiaization of the connectivity matrix
            'init_connectivity_matrix': "",
            # Export the connectivity matrix
            'export_connectivity': "",
            # Initialize the wrapper connectivity matrix
            'wrapper_init_connectivity': "",
            # Wrapper access to connectivity matrix
            'wrapper_access_connectivity': """
    # Connectivity
    def post_rank(self):
        return proj%(id_copy)s.get_post_rank()
    def pre_rank(self, int n):
        return proj%(id_copy)s.get_pre_rank()[n]
    # Local variable w
    def get_w(self):
        return proj%(id_copy)s.get_w()
    def set_w(self, value):
        print('Cannot modify weights of a copied projection.')
    def get_dendrite_w(self, int rank):
        return proj%(id_copy)s.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, value):
        print('Cannot modify weights of a copied projection.')
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj%(id_copy)s.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, %(float_prec)s value):
        print('Cannot modify weights of a copied projection.')
            """ % {'id_proj': self.id, 'id_copy': self.projection.id, 'float_prec': Global.config['precision']},
            # Wrapper access to variables
            'wrapper_access_parameters_variables' : "",
            # Variables for the psp code
            'psp_prefix': """
        int rk_pre;
        %(float_prec)s sum=0.0;"""
        } % {'float_prec': Global.config['precision']}

        # OMP code
        if Global.config['num_threads'] > 1:
            omp_code = '#pragma omp parallel for private(sum)' if self.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        # PSP
        psp = self.synapse_type.description['psp']['cpp']  % {
            'id_pre': self.pre.id,
            'id_post': self.post.id,
            'local_index':'[i][j]',
            'global_index': '[i]',
            'pre_index': '[pre_rank[i][j]]',
            'post_index': '[post_rank[i]]',
            'pre_prefix': 'pop'+str(self.pre.id)+'.',
            'post_prefix': 'pop'+str(self.post.id)+'.'}
        psp = psp.replace('rk_pre', 'pre_rank[i][j]').replace(';', '')

        # Take delays into account if any
        if self.delays > Global.config['dt']:
            psp = psp.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                'pop%(id_pre)s._delayed_r[delay-1][rk_pre]' % {'id_pre': self.pre.id}
                # TODO HD: wouldn't it be much better to reduce delay globaly, instead of the substraction here???
            )

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse_type.operation

        if operation == 'sum':
            sum_code = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = 0.0;
            for(int j = 0; j < pre_rank[i].size(); j++){
                sum += %(psp)s ;
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
        }
    }
"""
        elif operation == 'max':
            sum_code = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = %(psp)s;
            for(int j = 0; j < pre_rank[i].size(); j++){
                if(%(psp)s > sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
        }
    }
"""
        elif operation == 'min':
            sum_code = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = %(psp)s;
            for(int j = 0; j < pre_rank[i].size(); j++){
                if(%(psp)s < sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum;
        }
    }
"""
        elif operation == 'mean':
            sum_code = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s, copied from proj%(id)s
    if(pop%(id_post)s._active){
        %(omp_code)s
        for(int i = 0; i < post_rank.size(); i++){
            sum = 0.0;
            for(int j = 0; j < pre_rank[i].size(); j++){
                sum += %(psp)s ;
            }
            pop%(id_post)s._sum_%(target)s[post_rank[i]] += sum/ (%(float_prec)s)(pre_rank[i].size());
        }
    }
"""
        else:
            sum_code = ""

        self.generator['omp']['body_compute_psp'] = sum_code % {
            'id_proj': self.id, 'target': self.target,
            'id_pre': self.pre.id, 'name_pre': self.pre.name,
            'id_post': self.post.id, 'name_post': self.post.name,
            'id': self.projection.id,
            'float_prec': Global.config['precision'],
            'omp_code': omp_code,
            'psp': psp
        }

    ##############################
    ## Override useless methods
    ##############################
    def save_connectivity(self, filename):
        Global._warning('Shared projections can not be saved.')
    def save(self, filename):
        Global._warning('Shared projections can not be saved.')
    def load(self, filename):
        Global._warning('Shared projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        Global._warning('Shared projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        Global._warning('Shared projections can not display connectivity matrices.')
