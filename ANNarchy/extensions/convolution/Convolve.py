"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from __future__ import print_function
import numpy as np
from copy import deepcopy

from ANNarchy.core import Global
from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.intern import Messages

from ANNarchy.generator.Utils import tabify, remove_trailing_spaces
from .ConvolveTemplate import *
from .Utils import SharedSynapse

# Indices used for each dimension
indices = ['i', 'j', 'k', 'l', 'm', 'n']

class Convolution(SpecificProjection):
    r"""
    Performs a convolution of a weight kernel on the pre-synaptic population.

    Despite its name, the operation performed is actually a cross-correlation, as is usual in computer vision and convolutional neural networks:

    $$g(x) = \sum_{k=-n}^n h(k) \, f(x + k)$$

    The convolution operation benefits from giving a multi-dimensional geometry to the populations and filters, for example in 2D:

    ```python
    inp = ann.Population(geometry=(100, 100), neuron=ann.Neuron(parameters="r = 0.0"))
    pop = ann.Population(geometry=(100, 100), neuron=ann.Neuron(equations="r = sum(exc)"))
    
    proj = Convolution(inp, pop, 'exc')
    proj.connect_filter(
        [
            [-1., 0., 1.],
            [-1., 0., 1.],
            [-1., 0., 1.]
        ])
    ```

    The maximum number of dimensions for populations and filters is 4, an error is thrown otherwise.

    Depending on the number of dimensions of the pre- and post-synaptic populations, as well as of the kernel, the convolution is implemented differentely.

    **Method connect_filter()**

    * If the pre- and post-populations have the same dimension as the kernel, the convolution is regular. Example:

        (100, 100) * (3, 3) -> (100, 100)

    * If the post-population has one dimension less than the pre-synaptic one, the last dimension of the kernel must match the last one of the pre-synaptic population. Example:

        (100, 100, 3) * (3, 3, 3) -> (100, 100)

    * If the kernel has less dimensions than the two populations, the number of neurons in the last dimension of the populations must be the same. The convolution will be calculated for each feature map in the last dimension. In this case, you must set ``keep_last_dimension`` to ``True``. Example:

        (100, 100, 16) * (3, 3) -> (100, 100, 16)

    **Method connect_filters()**

    * If the kernel has more dimensions than the pre-synaptic population, this means a bank of different filters will be applied on the pre-synaptic population (like a convolutional layer in a CNN). Attention: the first index of ``weights`` corresponds to the different filters, while the result will be accessible in the last dimension of the post-synaptic population. You must set the ``multiple`` argument to True. Example:

        (100, 100) * (16, 3, 3) -> (100, 100, 16)

    The convolution **always** uses padding for elements that would be outside the array (no equivalent of ``valid`` in tensorflow). It is 0.0 by default, but can be changed using the ``padding`` argument. Setting ``padding`` to the string ``border`` will repeat the value of the border elements.

    Sub-sampling will be automatically performed according to the populations' geometry. If these geometries do not match, an error will be thrown. Example:

        (100, 100) * (3, 3) -> (50, 50)

    You can redefine the sub-sampling by providing a list ``subsampling`` as argument, defining for each post-synaptic neuron the coordinates of the pre-synaptic neuron which will be the center of the filter/kernel.

    :param pre: pre-synaptic population (either its name or a ``Population`` object).
    :param post: post-synaptic population (either its name or a ``Population`` object).
    :param target: type of the connection
    :param psp: continuous influence of a single synapse on the post-synaptic neuron (default for rate-coded: ``w*pre.r``).
    :param operation: operation (sum, max, min, mean) performed by the kernel (default: sum).
    """

    def __init__(self, pre, post, target, psp="pre.r * w", operation="sum", name=None, copied=False):

        # Create the description, but it will not be used for generation
        SpecificProjection.__init__(
            self,
            pre,
            post,
            target,
            synapse=SharedSynapse(
                psp=psp,
                operation=operation,
                name="Convolution with '"+operation+"' operation",
                description="Convoluted kernel over the pre-synaptic population."
            ),
            name=name,
            copied=copied
        )

        # Disable saving
        self._saveable = False

        # For copy
        self._used_single_filter = False
        self._used_bank_of_filters = False
        self.operation = operation

    @property
    def weights(self):
        if not self.initialized:
            return self.init["weights"]
        else:
            return np.array(self.cyInstance.get_w())

    @weights.setter
    def weights(self, value):
        if not self.initialized:
            self.init["weights"]=value
        else:
            if self.dim_kernel != value.ndim:
                raise AttributeError("Mismatch between filter dimensions")

            self.cyInstance.set_w(value)

    def connect_filter(self, weights, delays=0.0, keep_last_dimension=False, padding=0.0, subsampling=None):
        """
        Applies a single filter on the pre-synaptic population.

        :param weights: numpy array or list of lists representing the matrix of weights for the filter.
        :param delays: delay in synaptic transmission (default: dt). Can only be the same value for all neurons.
        :param keep_last_dimension: defines if the last dimension of the pre- and post-synaptic will be convolved in parallel. The weights matrix must have one dimension less than the pre-synaptic population, and the number of neurons in the last dimension of the pre- and post-synaptic populations must match. Default: False.
        :param padding: value to be used for the rates outside the pre-synaptic population. If it is a floating value, the pre-synaptic population is virtually extended with this value above its boundaries. If it is equal to 'border', the values on the boundaries are repeated. Default: 0.0.
        :param subsampling: list for each post-synaptic neuron of coordinates in the pre-synaptic population defining the center of the kernel/filter. Default: None.
        """

        # Process the weights
        self.weights = np.array(weights)

        # Process the delays
        self.delays = float(delays)
        if not isinstance(delays, (int, float)):
            Messages._error('Convolutions can only have constant delays.')

        self.subsampling = subsampling
        self.keep_last_dimension = keep_last_dimension
        self.padding = padding
        self.multiple = False

        # Check dimensions of populations and weight matrix
        self.dim_kernel = self.weights.ndim
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension

        if self.dim_post > 4:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: Too many dimensions for the post-synaptic population (maximum 4).')

        if self.dim_pre > 4:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: Too many dimensions for the pre-synaptic population (maximum 4).')

        if self.dim_kernel > 5  or (not self.multiple and self.dim_kernel > 4):
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: Too many dimensions for the kernel (maximum 4).')

        # Check if the last axes match for parallel convolution (e.g. 3-2-3)
        if self.dim_kernel < self.dim_pre:
            if not self.keep_last_dimension:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Messages._error('Convolution: If the kernel has less dimensions than the pre-synaptic population, you need to set the flag keep_last_dimension to True.')

            if self.pre.geometry[-1] != self.post.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Messages._error('Convolution: If the kernel has fewer dimensions than the two populations (keep_last_dimension=True), these must have the same number of neurons in the last dimension.')

        # If the last dim of the kernel matches the last dim of the pre-pop, the last pop can have one dimension less.
        if self.dim_post < self.dim_pre: # OK, but check the last dimension of the kernel has the same size as the post-population
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Messages._error('Convolution: If the post-synaptic population has less dimensions than the pre-synaptic one, the last dimension of the filter must be equal to the last of the pre-synaptic population.')

        # Check if it is a bank of filters
        if self.dim_kernel > self.dim_pre:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: If the kernel has more dimensions than the pre-synaptic population, you need to use the connect_filters() method.')


        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates()

        # Finish building the synapses
        self._create()

        # For copy
        self._used_single_filter = True

        return self


    def connect_filters(self, weights, delays=0.0, keep_last_dimension=False, padding=0.0, subsampling=None):
        """
        Applies a set of different filters on the pre-synaptic population.

        The weights matrix must have one dimension more than the pre-synaptic populations, and the number of neurons in the last dimension of the post-synaptic population must be equal to the number of filters.


        :param weights: numpy array or list of lists representing the matrix of weights for the filter.
        :param delays: delay in synaptic transmission (default: dt). Can only be the same value for all neurons.
        :param keep_last_dimension: defines if the last dimension of the pre- and post-synaptic will be convolved in parallel. The weights matrix must have one dimension less than the pre-synaptic population, and the number of neurons in the last dimension of the pre- and post-synaptic populations must match. Default: False.
        :param padding: value to be used for the rates outside the pre-synaptic population. If it is a floating value, the pre-synaptic population is virtually extended with this value above its boundaries. If it is equal to 'border', the values on the boundaries are repeated. Default: 0.0.
        :param subsampling: list for each post-synaptic neuron of coordinates in the pre-synaptic population defining the center of the kernel/filter. Default: None.
        """

        # Process the weights
        self.weights = np.array(weights)

        # Process the delays
        self.delays = float(delays)
        if not isinstance(delays, (int, float)):
            Messages._error('Convolutions can only have constant delays.')

        self.subsampling = subsampling
        self.keep_last_dimension = keep_last_dimension
        self.padding = padding
        self.multiple = True

        # Check dimensions of populations and weight matrix
        self.dim_kernel = self.weights.ndim
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension


        if self.dim_post > 4:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: Too many dimensions for the post-synaptic population (maximum 4).')

        if self.dim_pre > 4:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: Too many dimensions for the pre-synaptic population (maximum 4).')

        if self.dim_kernel > 5  or (not self.multiple and self.dim_kernel > 4):
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: Too many dimensions for the kernel (maximum 4).')

        # Check if the last axes match for parallel convolution (e.g. 3-2-3)
        if self.dim_kernel < self.dim_pre:
            if not self.keep_last_dimension:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Messages._error('Convolution: If the kernel has less dimensions than the pre-synaptic population, you need to set the flag keep_last_dimension to True.')

            if self.pre.geometry[-1] != self.post.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Messages._error('Convolution: If the kernel has fewer dimensions than the two populations (keep_last_dimension=True), these must have the same number of neurons in the last dimension.')

        # If the last dim of the kernel matches the last dim of the pre-pop, the last pop can have one dimension less.
        if self.dim_post < self.dim_pre: # OK, but check the last dimension of the kernel has the same size as the post-population
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Messages._error('Convolution: If the post-synaptic population has less dimensions than the pre-synaptic one, the last dimension of the filter must be equal to the last of the pre-synaptic population.')

        # The last dimension of the post population must correspond to the number of filters
        if self.weights.shape[0] != self.post.geometry[-1]:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Messages._error('Convolution: For multiple filters, the last dimension of the post-synaptic population must have as many neurons as there are filters.')

        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates_bank()

        # Finish building the synapses
        self._create()

        # For copy
        self._used_bank_of_filters = True

        return self

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks.  Internal use only."
        copied_proj = Convolution(pre=pre, post=post, target=self.target,
                                  psp=self.synapse_type.psp, operation=self.operation,
                                  name=self.name, copied=True)

        copied_proj.delays = self.delays
        copied_proj.weights = self.weights

        copied_proj.subsampling = self.subsampling
        copied_proj.keep_last_dimension = self.keep_last_dimension
        copied_proj.padding = self.padding
        copied_proj.multiple = self.multiple
        copied_proj.dim_kernel = self.weights.ndim
        copied_proj.dim_pre = self.pre.dimension
        copied_proj.dim_post = self.post.dimension

        if self._used_single_filter:
            copied_proj._generate_pre_coordinates()
        elif self._used_bank_of_filters:
            copied_proj._generate_pre_coordinates_bank()
        else:
            raise ValueError("Either use single filter or bank of filter must be True! (Missing connect?)")

        copied_proj._create()
        copied_proj._connection_method = self._connection_method
        copied_proj._connection_args = self._connection_args
        copied_proj._connection_delay = self._connection_delay
        copied_proj._storage_format = self._storage_format
        return copied_proj

    def _create(self):
        # create fake LIL object, just for compilation.
        try:
            from ANNarchy.cython_ext.Connector import LILConnectivity
        except Exception as e:
            Messages._print(e)
            Messages._error('ANNarchy was not successfully installed.')

        lil = LILConnectivity()
        lil.max_delay = self.delays
        lil.uniform_delay = self.delays
        self.connector_name = "Convolution"
        self.connector_description = "Convolution"
        self._store_connectivity(self._load_from_lil, (lil, ), self.delays, storage_format="lil", storage_order="post_to_pre")

    ################################
    ### Create connection pattern
    ################################
    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """
        if not self._connection_method:
            Messages._error('Convolution: The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')

        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.pre_coordinates, self.weights)

        # Set delays after instantiation
        if self.delays > 0.0:
            self.cyInstance.set_delay(self.delays/get_global_config('dt'))

        return True

    def _generate_pre_coordinates(self):
        " Returns a list for each post neuron of the corresponding center coordinates."

        # Check if the list is already defined:
        if self.subsampling:
            try:
                shape = np.array(self.subsampling).shape
            except:
                Messages._error('Convolution: The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size, self.pre.dimension):
                Messages._error('Convolution: The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
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
                    Messages._error('Convolution: The pre-synaptic dimensions must be a multiple of the post-synaptic ones for down-sampling to work.')

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
                Messages._error('Convolution: The sub-sampling list must have', self.post.size / self.post.geometry[-1], 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size/ self.post.geometry[-1], self.pre.dimension):
                Messages._error('Convolution: The sub-sampling list must have', self.post.size/ self.post.geometry[-1], 'elements of size', self.pre.dimension)
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
                    Messages._error('Convolution: The pre-synaptic dimensions must be a multiple of the post-synaptic ones for down-sampling to work.')

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
    # Code generation
    ################################
    def _generate(self):
        """
        Overrides default code generation. This function is called during the code generation procedure.
        """
        # Filter definition
        filter_definition, filter_pyx_definition = self._filter_definition()

        # On CPUs we have a pre-load on the inner-most sub-vector
        use_inner_line = _check_paradigm("openmp")

        # Convolve_code
        if not self.multiple:
            convolve_code, sum_code = self._generate_convolve_code(pre_load_inner_line=use_inner_line)
        else:
            convolve_code, sum_code = self._generate_bank_code(pre_load_inner_line=use_inner_line)

        if _check_paradigm("openmp"):
            self._generate_omp(filter_definition, filter_pyx_definition, convolve_code, sum_code)
        elif _check_paradigm("cuda"):
            self._generate_cuda(filter_definition, filter_pyx_definition, convolve_code, sum_code)
        else:
            raise NotImplementedError

    def _generate_omp(self, filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=True):
        """
        OpenMP code generation.
        """
        # Basic ids
        base_ids = {
            'id_proj': self.id,
            'size_post': self.post.size,
            'float_prec': get_global_config('precision')
        }

        # Fill the basic definitions
        conv_dict = deepcopy(convolve_template_omp)
        for key, value in conv_dict.items():
            value = value % base_ids
            conv_dict[key] = value
        self._specific_template.update(conv_dict)

        # Kernel-based method: specify w with the correct dimension
        if kernel:
            # The number of dimension influences the type
            cpp_type_w = filter_definition.replace(' w;', '')
            pyx_type_w = filter_pyx_definition.replace(' w', '')

            # Fill the code templates
            self._specific_template['declare_parameters_variables'] = tabify(filter_definition.strip(), 1)
            self._specific_template['export_parameters_variables'] = ""
            self._specific_template['access_parameters_variables'] = conv_filter_template["openmp"]["access"] % {'type_w': cpp_type_w}
            self._specific_template['export_connectivity'] += conv_filter_template["pyx_wrapper"]["export"] % {'type_w': pyx_type_w}
            self._specific_template['wrapper_args'] += conv_filter_template["pyx_wrapper"]["args"]
            self._specific_template['wrapper_init_connectivity'] += conv_filter_template["pyx_wrapper"]["init"] % {'id_proj': self.id}
            self._specific_template['wrapper_access_connectivity'] += conv_filter_template["pyx_wrapper"]["access"] % {'id_proj': self.id, 'float_prec': get_global_config('precision')}

        # Override the monitor to avoid recording the weights
        self._specific_template['monitor_class'] = ""
        self._specific_template['monitor_export'] = ""
        self._specific_template['monitor_wrapper'] = ""

        # Clean-up
        self._specific_template['clear_container'] = convolve_template_omp["clear"]

        # OMP code
        omp_code = ""
        if get_global_config('num_threads') > 1:
            omp_code = """
        #pragma omp for private(sum, rk_pre, coord) %(psp_schedule)s""" % {'psp_schedule': "" if not 'psp_schedule' in self._omp_config.keys() else self._omp_config['psp_schedule']}

        # HD ( 16.10.2015 ):
        # pre-load delayed firing rate in a local array, so we
        # prevent multiple accesses to pop%(id_pre)s._delayed_r[delay-1]
        # wheareas delay is set available as variable
        # TODO HD: wouldn't it be much better to reduce delay globaly, instead of the substraction here???
        if self.delays > get_global_config('dt'):
            pre_load_r = """
        // pre-load delayed firing rate
        auto delayed_r = pop%(id_pre)s._delayed_r[delay-1];
        """% {'id_pre': self.pre.id}
        else:
            pre_load_r = ""

        # Target variable depends on neuron type
        target_code = "_sum_%(target)s" if self.post.neuron_type.type=="rate" else "g_%(target)s"
        target_code %= {'target': self.target}

        # Compute sum
        wsum =  """
        if ( _transmission && pop%(id_pre)s._active ) {
            int* coord;
""" + pre_load_r + """
            %(omp_code)s
            for(int i = 0; i < %(size_post)s; i++){
                coord = pre_coords[i].data();

                // perform the convolution
""" + tabify(convolve_code, 1) + """

                // store result
                pop%(id_post)s.%(target)s[i] += """ + sum_code + """;
            } // for
        } // if
"""

        # Finalize the processing code
        self._specific_template['psp_code'] = wsum % \
        {   'id_proj': self.id,
            'target': target_code,
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size,
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'omp_code': omp_code,
            'convolve_code': convolve_code
        }

    def _generate_cuda(self, filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=True):
        """
        CUDA code generation.
        """
        # Basic ids
        base_ids = {
            'id_proj': self.id,
            'size_post': self.post.size,
            'float_prec': get_global_config('precision')
        }

        # Fill the basic definitions
        conv_dict = deepcopy(convolve_template_cuda)
        for key, value in conv_dict.items():
            value = value % base_ids
            conv_dict[key] = value
        self._specific_template.update(conv_dict)

        # Kernel-based method: specify w with the correct dimension
        if kernel:
            # The number of dimension influences the type
            cpp_type_w = filter_definition.replace(' w;', '')
            pyx_type_w = filter_pyx_definition.replace(' w', '')

            # Fill the code templates
            self._specific_template['declare_parameters_variables'] = conv_filter_template["cuda"]["declare"] % {'cpu_side_filter': filter_definition.strip(), 'float_prec': get_global_config('precision')}
            self._specific_template['export_parameters_variables'] = ""
            self._specific_template['access_parameters_variables'] = conv_filter_template["cuda"]["access"] % {'type_w': cpp_type_w, 'id_proj': self.id}
            self._specific_template['export_connectivity'] += conv_filter_template["pyx_wrapper"]["export"] % {'type_w': pyx_type_w}
            self._specific_template['wrapper_args'] += conv_filter_template["pyx_wrapper"]["args"]
            self._specific_template['wrapper_init_connectivity'] += conv_filter_template["pyx_wrapper"]["init"] % {'id_proj': self.id}
            self._specific_template['wrapper_access_connectivity'] += conv_filter_template["pyx_wrapper"]["access"] % {'id_proj': self.id, 'float_prec': get_global_config('precision')}

            # Memory transfer of variables
            dim_pre = self.dim_pre
            if self.multiple:
               dim_pre += 1
            self._specific_template['host_device_transfer'] += conv_filter_template["cuda"]["host_device_transfer"] % {'ctype': get_global_config('precision'), 'id_proj': self.id, 'pre_dim': dim_pre}

            # Other fields
            self._specific_template['size_in_bytes'] = ""

        # Override the monitor to avoid recording the weights
        self._specific_template['monitor_class'] = ""
        self._specific_template['monitor_export'] = ""
        self._specific_template['monitor_wrapper'] = ""

        # Clean-up
        self._specific_template['clear_container'] = convolve_template_cuda["clear"]

        # Add pre-synaptic variables to argument list
        pre_variables_header = ""
        pre_variables_invoke = ""
        pre_variables_call = ""
        for pre_dep in self.synapse_type.description['dependencies']['pre']:
            # HD (TODO): the type to float precision works for now, but one should
            #            look up the type in the pre-synaptic neuron type...
            pre_id_dict = {
                'id_pre': self.pre.id,
                'name': pre_dep,
                'type': get_global_config('precision')
            }
            pre_variables_header += ", const %(type)s* __restrict__ pre_%(name)s" % pre_id_dict
            pre_variables_invoke += ", pre_%(name)s" % pre_id_dict
            pre_variables_call += ", pop%(id_pre)s.gpu_%(name)s" % pre_id_dict

        # Finalize code templates
        code_ids = {
            'id_proj': self.id,
            'target': self.target,
            'id_post': self.post.id,
            'pre_dim': self.dim_pre,
            'convolve_code': convolve_code,
            'float_prec': get_global_config('precision'),
            'pre_variables_header': pre_variables_header,
            'pre_variables_invoke': pre_variables_invoke,
            'pre_variables_call': pre_variables_call,
            'pre_variable': "pre_%(name)s" % pre_id_dict,
            'convolve_code': convolve_code
        }

        # Finalize the processing code
        if not self.multiple:
            # Convolution
            self._specific_template['psp_body'] = cuda_convolution_single_filter["body"] % code_ids
            self._specific_template['psp_invoke'] = cuda_convolution_single_filter["invoke"] % code_ids
            self._specific_template['psp_header'] = cuda_convolution_single_filter["header"] % code_ids
            self._specific_template['psp_call'] = cuda_convolution_single_filter["call"] % code_ids

        else:
            num_elem_per_filter = 1
            for i in self.weights.shape[1:]:
                num_elem_per_filter *= i
            code_ids.update({
                'filter_dim': self.dim_kernel,
                'num_elem_filter': num_elem_per_filter
            })

            # Bank of filters
            if len(self.weights.shape)==3 and len(self.pre.geometry)==2:
                code_ids.update({
                    'post_size': self.post.size,
                    'filter_dim_i': self.weights.shape[1],
                    'filter_dim_j': self.weights.shape[2],
                    'pre_offset_i': self._center_filter(self.weights.shape[1]),
                    'pre_offset_j': self._center_filter(self.weights.shape[2]),
                    'pre_border_i': self.pre.geometry[0]-1,
                    'pre_border_j': self.pre.geometry[1]-1,
                    'pre_dim_j': self.pre.geometry[1]
                })
                self._specific_template['psp_body'] = cuda_convolution_bank_of_filter_3d["body"] % code_ids
                self._specific_template['psp_invoke'] = cuda_convolution_bank_of_filter_3d["invoke"] % code_ids
                self._specific_template['psp_header'] = cuda_convolution_bank_of_filter_3d["header"] % code_ids
                self._specific_template['psp_call'] = cuda_convolution_bank_of_filter_3d["call"] % code_ids

            elif len(self.weights.shape)==4 and len(self.pre.geometry)==3:
                code_ids.update({
                    'post_size': self.post.size,
                    'filter_dim_i': self.weights.shape[1],
                    'filter_dim_j': self.weights.shape[2],
                    'filter_dim_k': self.weights.shape[3],
                    'pre_offset_i': self._center_filter(self.weights.shape[1]),
                    'pre_offset_j': self._center_filter(self.weights.shape[2]),
                    'pre_offset_k': self._center_filter(self.weights.shape[3]),
                    'pre_border_i': self.pre.geometry[0]-1,
                    'pre_border_j': self.pre.geometry[1]-1,
                    'pre_border_k': self.pre.geometry[2]-1,
                    'pre_dim_j': self.pre.geometry[1],
                    'pre_dim_k': self.pre.geometry[2]
                })
                self._specific_template['psp_body'] = cuda_convolution_bank_of_filter_4d["body"] % code_ids
                self._specific_template['psp_invoke'] = cuda_convolution_bank_of_filter_4d["invoke"] % code_ids
                self._specific_template['psp_header'] = cuda_convolution_bank_of_filter_4d["header"] % code_ids
                self._specific_template['psp_call'] = cuda_convolution_bank_of_filter_4d["call"] % code_ids

            else:
                self._specific_template['psp_body'] = cuda_convolution_bank_of_filter["body"] % code_ids
                self._specific_template['psp_invoke'] = cuda_convolution_bank_of_filter["invoke"] % code_ids
                self._specific_template['psp_header'] = cuda_convolution_bank_of_filter["header"] % code_ids
                self._specific_template['psp_call'] = cuda_convolution_bank_of_filter["call"] % code_ids

        # Post-neuron is a spike neuron (e.g., part of ANN-to-SNN conversion)
        if self.post.neuron_type.type == "spike":
            self._specific_template['psp_call'] = self._specific_template['psp_call'].replace("gpu__sum_"+self.target, "gpu_g_"+self.target)

        # Remove trailing spaces
        self._specific_template['psp_body'] = remove_trailing_spaces(self._specific_template['psp_body'])

    ################################
    ### Utilities
    ################################
    def _center_filter(self, i):
        return int(i/2) if i%2==1 else int(i/2)-1

    def _filter_definition(self):
        dim = self.dim_kernel
        cpp = get_global_config('precision')
        pyx = get_global_config('precision')
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

    def _filter_coordinates_to_index(self, name, filter_dim):
        dim = len(filter_dim)

        txt = ""

        for d in range(dim):
            if txt == "" : # first coordinate is special
                txt = indices[0] + "_" + name
            else:
                txt = str(filter_dim[d]) + '*(' + txt + ') + ' + indices[d]  + '_' + name

        return txt

    def _generate_convolve_code(self, pre_load_inner_line=True):
        """
        Generate the loop for the convolution case.

        Parameters:

        * pre_load_inner_line: for CPU-code it's useful to have a local variable for accessing the innermost sub-vector
        """

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse_type.operation

        # Main code
        code = tabify("sum = 0.0;\n", 3)

        # Generate for loops
        for dim in range(self.dim_kernel):
            if pre_load_inner_line:
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
                code += tabify(
                    """int %(index)s_pre = coord[%(dim)s] %(operator)s (%(index)s_w - %(center)s);""" %
                        {
                            'id_proj': self.id,
                            'index': indices[dim],
                            'dim': dim,
                            'operator': '+' ,
                            'center': self._center_filter(self.weights.shape[dim])
                        }, 1)
            else:
                code += tabify(
                    """int %(index)s_pre = coord[%(dim)s];""" %
                        {
                            'id_proj': self.id,
                            'index': indices[dim],
                            'dim': dim
                        }, 1)

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

        # if True, we need to take the last dimension from coords
        if self.keep_last_dimension:
            id_dict = {
                'index': indices[self.dim_kernel],
                'dim': self.dim_kernel
            }
            code += "int %(index)s_pre = coord[%(dim)s];" % id_dict

        # Compute pre-synaptic rank
        code += tabify("""
                rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}, dim)
        if not pre_load_inner_line:
            code += tabify("""
                w_idx = %(value)s;""" % {'value': self._filter_coordinates_to_index('w', self.weights.shape)}, dim)

        # Compute the increment
        index = ""
        for dim in range(self.dim_kernel):
            index += '[' + indices[dim] + '_w]'

        # Indices etc. depends on the target platform
        inc_dict = {
            'id_pre': self.pre.id,
            'id_post': self.post.id,
        }
        if _check_paradigm("openmp"):
            inc_dict.update({
                'global_index': '[i]',
                'local_index': index,
                'pre_index': '[rk_pre]',
                'post_index': '[rk_post]',
                'pre_prefix': 'pop'+str(self.pre.id)+'.',
                'post_prefix': 'pop'+str(self.post.id)+'.'
            })
        elif _check_paradigm("cuda"):
            inc_dict.update({
                'global_index': '',
                'local_index': '[w_idx]',
                'pre_index': '[rk_pre]',
                'post_index': '',
                'pre_prefix': 'pre_',
                'post_prefix': 'post_'
            })
        else:
            raise NotImplementedError

        # Fill the code template
        increment = self.synapse_type.description['psp']['cpp'] % inc_dict

        # Delays
        if self.delays > get_global_config('dt'):
            increment = increment.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                'delayed_r[rk_pre]'
            )

        # Apply the operation
        if operation == "sum":
            if self.dim_kernel == 1:
                code += tabify("""
                sum += %(increment)s""" % {'increment': increment}, dim)
            else:
                if pre_load_inner_line:
                    code += tabify("""
                    sum += %(increment)s""" % {'increment': increment.replace('w'+inner_idx, 'inner_line')}, dim)
                else:
                    code += tabify("""
                    sum += %(increment)s""" % {'increment': increment}, dim)
        elif operation == "max":
            code += tabify("""
                %(float_prec)s _psp = %(increment)s
                if(_psp > sum) sum = _psp;""" % {'increment': increment, 'float_prec': get_global_config('precision')}, dim)
        elif operation == "min":
            code += tabify("""
                %(float_prec)s _psp = %(increment)s
                if(_psp < sum) sum = _psp;""" % {'increment': increment, 'float_prec': get_global_config('precision')}, dim)
        elif operation == "mean":
            code += tabify("""
                sum += %(increment)s""" % {'increment': increment}, dim)
        else:
            Messages._error('Convolution: Operation', operation, 'is not implemented yet for shared projections.')

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

    def _generate_bank_code(self, pre_load_inner_line=True):
        """
        Generate the loop for the bank of filters case.

        Parameters:

        * pre_load_inner_line: for CPU-code it's useful to have a local variable for accessing the innermost sub-vector
        """

        # Operation to be performed: sum, max, min, mean
        operation = self.synapse_type.operation

        # Main code
        code = tabify("sum = 0.0;\n", 3)

        # Generate for loops
        for dim in range(self.dim_kernel-1):
            if pre_load_inner_line:
                if dim == self.dim_kernel-2:
                    inner_idx = ""
                    for i in range(self.dim_kernel-2):
                        inner_idx += "["+indices[i]+"_w]"
                    code += tabify("""
                const %(float_prec)s* w_inner_line = w[coord[%(dim_pre)s]]%(inner_idx)s.data();
    """ % {'float_prec': get_global_config('precision'), 'inner_idx': inner_idx, 'dim_pre': self.dim_pre}, dim)

            code += tabify("""
            for (int %(index)s_w = 0; %(index)s_w < %(size)s;%(index)s_w++) {
            """ % { 'index': indices[dim], 'size': self.weights.shape[dim+1]}, dim)

            # Compute indices
            if dim < self.dim_kernel:
                code += tabify(
                    """int %(index)s_pre = coord[%(dim)s] %(operator)s (%(index)s_w - %(center)s);""" %
                    {
                        'id_proj': self.id,
                        'index': indices[dim],
                        'dim': dim,
                        'operator': '+',
                        'center': self._center_filter(self.weights.shape[dim+1])
                    }, 1)
            else:
                code += tabify(
                    """int %(index)s_pre = coord[%(dim)s];""" %
                    {
                        'id_proj': self.id,
                        'index': indices[dim],
                        'dim': dim
                    }, 1)

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
        if not pre_load_inner_line:
            code += tabify("""
                w_idx = %(value)s;""" % {'value': self._filter_coordinates_to_index('w', self.weights.shape[1:])}, dim)

        # Compute the increment
        if pre_load_inner_line:
            index = "_inner_line["+indices[self.dim_kernel-2]+"_w]"
        else:
            index = "[coord["+str(self.dim_pre)+"]]"
            for dim in range(self.dim_kernel-1):
                index += '[' + indices[dim] + '_w]'

        # Indices etc. depend on target platform
        inc_dict = {
            'id_pre': self.pre.id,
            'id_post': self.post.id,
        }
        if _check_paradigm("openmp"):
            inc_dict.update({
                'local_index': index,
                'global_index': '[i]',
                'pre_index': '[rk_pre]',
                'post_index': '[rk_post]',
                'pre_prefix': 'pop'+str(self.pre.id)+'.',
                'post_prefix': 'pop'+str(self.post.id)+'.'
            })
        elif _check_paradigm("cuda"):
            inc_dict.update({
                'local_index': "[w_idx]",
                'global_index': '[bIdx]',
                'pre_index': '[rk_pre]',
                'post_index': '[bIdx]',
                'pre_prefix': 'pre_',
                'post_prefix': 'post_'
            })
        else:
            raise NotImplementedError

        # Pixel-wise applied operation
        increment = self.synapse_type.description['psp']['cpp']
        if _check_paradigm("cuda"):
            increment = increment.replace("w%(local_index)s", "w_bank%(local_index)s")
        increment %= inc_dict

        # Delays
        if self.delays > get_global_config('dt'):
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
            if(_psp > sum) sum = _psp;""" % {'increment': increment, 'float_prec': get_global_config('precision')}, 1+dim)
        elif operation == "min":
            code += tabify("""
            %(float_prec)s _psp = %(increment)s
            if(_psp < sum) sum = _psp;""" % {'increment': increment, 'float_prec': get_global_config('precision')}, 1+dim)
        elif operation == "mean":
            code += tabify("""
            sum += %(increment)s""" % {'increment': increment}, 1+dim)
        else:
            Messages._error('SharedProjection: Operation', operation, 'is not implemented yet for shared projections.')

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

    ##############################
    ## Override useless methods
    ##############################
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

    def save_connectivity(self, filename):
        "Not available."
        Messages._warning('Convolutional projections can not be saved.')
    def save(self, filename):
        "Not available."
        Messages._warning('Convolutional projections can not be saved.')
    def load(self, filename):
        "Not available."
        Messages._warning('Convolutional projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        "Not available."
        Messages._warning('Convolutional projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        "Not available."
        Messages._warning('Convolutional projections can not display connectivity matrices.')

