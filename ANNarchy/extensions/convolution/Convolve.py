# =============================================================================
#
#     Convolution.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2019  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# =============================================================================
from __future__ import print_function
import numpy as np

from ANNarchy.core import Global
from ANNarchy.core.Projection import Projection

from ANNarchy.generator.Utils import tabify
from .Utils import SharedSynapse

# Indices used for each dimension
indices = ['i', 'j', 'k', 'l', 'm', 'n']

class Convolution(Projection):
    """
    Performs a convolution of a weight kernel on the pre-synaptic population.

    Despite its name, the operation performed is actually a cross-correlation, as is usual in computer vision and convolutional neural networks:

    $$g(x) = \sum_{k=-n}^n h(k) \, f(x + k)$$

    The convolution operation benefits from giving a multi-dimensional geometry to the populations and filters, for example in 2D:

    ```python
    inp = Population(geometry=(100, 100), neuron=Neuron(parameters="r = 0.0"))
    pop = Population(geometry=(100, 100), neuron=Neuron(equations="r = sum(exc)"))
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
    """

    def __init__(self, pre, post, target, psp="pre.r * w", operation="sum", name=None, copied=False):
        """
        :param pre: pre-synaptic population (either its name or a ``Population`` object).
        :param post: post-synaptic population (either its name or a ``Population`` object).
        :param target: type of the connection
        :param psp: continuous influence of a single synapse on the post-synaptic neuron (default for rate-coded: ``w*pre.r``).
        :param operation: operation (sum, max, min, mean) performed by the kernel (default: sum).
        """        

        # Create the description, but it will not be used for generation
        Projection.__init__(
            self,
            pre,
            post,
            target,
            synapse=SharedSynapse(psp=psp, operation=operation, name="Convolution operation", description="Convoluted kernel over the pre-synaptic population."),
            name=name,
            copied=copied
        )

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
            Global._error('Convolutions can only have constant delays.')

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
            Global._error('Convolution: Too many dimensions for the post-synaptic population (maximum 4).')

        if self.dim_pre > 4:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Global._error('Convolution: Too many dimensions for the pre-synaptic population (maximum 4).')

        if self.dim_kernel > 5  or (not self.multiple and self.dim_kernel > 4):
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Global._error('Convolution: Too many dimensions for the kernel (maximum 4).')

        # Check if the last axes match for parallel convolution (e.g. 3-2-3)
        if self.dim_kernel < self.dim_pre:
            if not self.keep_last_dimension:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Global._error('Convolution: If the kernel has less dimensions than the pre-synaptic population, you need to set the flag keep_last_dimension to True.')

            if self.pre.geometry[-1] != self.post.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Global._error('Convolution: If the kernel has fewer dimensions than the two populations (keep_last_dimension=True), these must have the same number of neurons in the last dimension.')

        # If the last dim of the kernel matches the last dim of the pre-pop, the last pop can have one dimension less.
        if self.dim_post < self.dim_pre: # OK, but check the last dimension of the kernel has the same size as the post-population
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Global._error('Convolution: If the post-synaptic population has less dimensions than the pre-synaptic one, the last dimension of the filter must be equal to the last of the pre-synaptic population.')

        # Check if it is a bank of filters
        if self.dim_kernel > self.dim_pre:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Global._error('Convolution: If the kernel has more dimensions than the pre-synaptic population, you need to use the connect_filters() method.')


        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates()

        # Finish building the synapses
        self._create()

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
            Global._error('Convolutions can only have constant delays.')

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
            Global._error('Convolution: Too many dimensions for the post-synaptic population (maximum 4).')

        if self.dim_pre > 4:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Global._error('Convolution: Too many dimensions for the pre-synaptic population (maximum 4).')

        if self.dim_kernel > 5  or (not self.multiple and self.dim_kernel > 4):
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Global._error('Convolution: Too many dimensions for the kernel (maximum 4).')

        # Check if the last axes match for parallel convolution (e.g. 3-2-3)
        if self.dim_kernel < self.dim_pre:
            if not self.keep_last_dimension:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Global._error('Convolution: If the kernel has less dimensions than the pre-synaptic population, you need to set the flag keep_last_dimension to True.')

            if self.pre.geometry[-1] != self.post.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Global._error('Convolution: If the kernel has fewer dimensions than the two populations (keep_last_dimension=True), these must have the same number of neurons in the last dimension.')

        # If the last dim of the kernel matches the last dim of the pre-pop, the last pop can have one dimension less.
        if self.dim_post < self.dim_pre: # OK, but check the last dimension of the kernel has the same size as the post-population
            if self.weights.shape[-1] != self.pre.geometry[-1]:
                print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
                Global._error('Convolution: If the post-synaptic population has less dimensions than the pre-synaptic one, the last dimension of the filter must be equal to the last of the pre-synaptic population.')

        # The last dimension of the post population must correspond to the number of filters
        if self.weights.shape[0] != self.post.geometry[-1]:
            print("Convolution:", self.dim_pre, '*', self.dim_kernel, '->', self.dim_post)
            Global._error('Convolution: For multiple filters, the last dimension of the post-synaptic population must have as many neurons as there are filters.')

        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates_bank()

        # Finish building the synapses
        self._create()

        return self

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks.  Internal use only."
        raise NotImplementedError

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
        self.connector_name = "Convolution"
        self.connector_description = "Convolution"
        self._store_connectivity(self._load_from_lil, (lil, ), self.delays)

    ################################
    ### Create connection pattern
    ################################
    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """
        if not self._connection_method:
            Global._error('Convolution: The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')

        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights, self.pre_coordinates)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Set delays after instantiation
        if self.delays > 0.0:
            self.cyInstance.set_delay(self.delays/Global.config['dt'])

    def _generate_pre_coordinates(self):
        " Returns a list for each post neuron of the corresponding center coordinates."

        # Check if the list is already defined:
        if self.subsampling:
            try:
                shape = np.array(self.subsampling).shape
            except:
                Global._error('Convolution: The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size, self.pre.dimension):
                Global._error('Convolution: The sub-sampling list must have', self.post.size, 'elements of size', self.pre.dimension)
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
                    Global._error('Convolution: The pre-synaptic dimensions must be a multiple of the post-synaptic ones for down-sampling to work.')

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
                Global._error('Convolution: The sub-sampling list must have', self.post.size / self.post.geometry[-1], 'elements of size', self.pre.dimension)
                return
            if shape != (self.post.size/ self.post.geometry[-1], self.pre.dimension):
                Global._error('Convolution: The sub-sampling list must have', self.post.size/ self.post.geometry[-1], 'elements of size', self.pre.dimension)
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
                    Global._error('Convolution: The pre-synaptic dimensions must be a multiple of the post-synaptic ones for down-sampling to work.')

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

        # Convolve_code
        if not self.multiple:
            convolve_code, sum_code = self._generate_convolve_code()
        else:
            convolve_code, sum_code = self._generate_bank_code()

        if Global._check_paradigm("openmp"):
            self._generate_omp(filter_definition, filter_pyx_definition, convolve_code, sum_code)
        elif Global._check_paradigm("cuda"):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _generate_omp(self, filter_definition, filter_pyx_definition, convolve_code, sum_code, kernel=True):
        """
        OpenMP code generation.
        """
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

            # Initialize the wrapper connectivity matrix
            'wrapper_init_connectivity': """
        proj%(id_proj)s.set_post_rank(list(range(%(size_post)s)))
        proj%(id_proj)s.set_pre_rank(coords)
""" % {'id_proj': self.id, 'size_post': self.post.size},

            # Delays
            'wrapper_init_delay': "",

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
        self._specific_template['size_in_bytes'] = "//TODO:\n"

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
        code = tabify("sum = 0.0;\n", 3)

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
            if self.dim_kernel == 1:
                code += tabify("""
                sum += %(increment)s""" % {'increment': increment}, dim)
            else:
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
            Global._error('Convolution: Operation', operation, 'is not implemented yet for shared projections.')

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
        code = tabify("sum = 0.0;\n", 3)

        # Generate for loops
        for dim in range(self.dim_kernel-1):
            code += tabify("""
            for(int %(index)s_w = 0; %(index)s_w < %(size)s;%(index)s_w++) {
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
        Global._warning('Convolutional projections can not be saved.')
    def save(self, filename):
        "Not available."
        Global._warning('Convolutional projections can not be saved.')
    def load(self, filename):
        "Not available."
        Global._warning('Convolutional projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        "Not available."
        Global._warning('Convolutional projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        "Not available."
        Global._warning('Convolutional projections can not display connectivity matrices.')

