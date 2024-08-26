"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.intern import Messages

from ANNarchy.generator.Utils import tabify, remove_trailing_spaces

from copy import deepcopy

from .PoolingTemplate import *
from .Utils import SharedSynapse

# Indices used for each dimension
indices = ['i', 'j', 'k', 'l', 'm', 'n']

class Pooling(SpecificProjection):
    """
    Performs a pooling operation (e.g. max.pooling) on the pre-synaptic population.

    Each post-synaptic neuron covers a specific region (``extent``) of the pre-synaptic
    population, over which the result of the operation on firing rates will be
    assigned to sum(target).

    The extent is automatically computed using the geometry of the populations, but can be specified in the `connect_pooling()`` methods.

    Example:

    ```python
    inp = ann.Population(geometry=(100, 100), neuron=ann.Neuron(parameters="r = 0.0"))
    pop = ann.Population(geometry=(50, 50), neuron=ann.Neuron(equations="r = sum(exc)"))
    
    proj = Pooling(inp, pop, 'exc', operation='max') # max-pooling
    proj.connect_pooling() # extent=(2, 2) is implicit
    ```

    :param pre: pre-synaptic population (either its name or a ``Population`` object).
    :param post: post-synaptic population (either its name or a ``Population`` object).
    :param target: type of the connection
    :param operation: pooling function to be applied ("max", "min", "mean")
    """
    def __init__(self, pre, post, target, psp="pre.r", operation="max", name=None, copied=False):

        # Sanity check
        if not operation in ["max", "mean", "min"]:
            Messages._error("Pooling: the operation must be either 'max', 'mean' or 'min'.")
        self.operation = operation

        # Store for _copy
        self.psp = psp

        SpecificProjection.__init__(
            self,
            pre,
            post,
            target,
            synapse=SharedSynapse(
                psp=psp, 
                operation=operation, 
                name="Pooling with '"+operation+"' operation",
                description=operation+"-pooling operation over the pre-synaptic population."
            ),
            name=name,
            copied=copied
        )

        # check dimensions of populations, should not exceed 4
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension
        if self.dim_post > 4:
            Messages._error('Pooling: Too many dimensions for the post-synaptic population (maximum 4).')
        if self.dim_pre > 4:
            Messages._error('Pooling: Too many dimensions for the pre-synaptic population (maximum 4).')

        # Disable saving
        self._saveable = False



    def connect_pooling(self, extent:tuple=None, delays:float=0.0):
        """
        :param extent: extent of the pooling area expressed in the geometry of the pre-synaptic population (e.g ``(2, 2)``). In each dimension, the product of this extent with the number of neurons in the post-synaptic population must be equal to the number of pre-synaptic neurons. Default: None.
        :param delays: synaptic delay in ms
        """

        # process extent
        self.extent_init = extent
        if extent is None:  # compute the extent automatically
            if self.pre.dimension != self.post.dimension:
                Messages._error(
                    'Pooling: If you do not provide the extent parameter, the two populations must have the same number of dimensions.')

            extent = list(self.pre.geometry)
            for dim in range(self.pre.dimension):
                extent[dim] /= self.post.geometry[dim]
                if self.pre.geometry[dim] != extent[dim] * self.post.geometry[dim]:
                    Messages._error(
                        'Pooling: Unable to compute the extent of the pooling area: the number of neurons do not match.')

        elif not isinstance(extent, tuple):
            Messages._error('Pooling: You must provide a tuple for the extent of the pooling operation.')

        self.extent = list(extent)
        if len(self.extent) < self.pre.dimension:
            Messages._error('Pooling: You must provide a tuple for the extent of the pooling operation.')

        # process delays
        self.delays = delays

        # Generate the pre-synaptic coordinates
        self._generate_extent_coordinates()

        # create fake LIL
        self._create()

        return self

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks.  Internal use only."
        copied_proj = Pooling(pre=pre, post=post, target=self.target, psp=self.psp,
                              operation=self.operation, name=self.name, copied=True)

        copied_proj.extent = self.extent
        copied_proj.delays = self.delays

        copied_proj._generate_extent_coordinates()
        copied_proj._create()

        copied_proj._connection_method = self._connection_method
        copied_proj._connection_args = self._connection_args
        copied_proj._connection_delay = self._connection_delay
        copied_proj._storage_format = self._storage_format
        return copied_proj

    def _create(self):
        """
        create fake LIL object, just for compilation process

        :return: no return value
        """
        try:
            from ANNarchy.cython_ext.Connector import LILConnectivity
        except Exception as e:
            Messages._print(e)
            Messages._error('ANNarchy was not successfully installed.')

        lil = LILConnectivity()
        lil.max_delay = self.delays
        lil.uniform_delay = self.delays
        self.connector_name = "Pooling"
        self.connector_description = "Pooling"
        self._store_connectivity(self._load_from_lil, (lil, ), self.delays, storage_format="lil", storage_order="post_to_pre")

    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """
        if not self._connection_method:
            Messages._error(
                'Pooling: The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')

        # Create the Cython instance
        proj = getattr(module, 'proj' + str(self.id) + '_wrapper')
        self.cyInstance = proj(self.pre_coordinates)

        return True

    def _generate_extent_coordinates(self):
        """
        Generates for each post-neuron the position of the top-left corner, where the pooling should be applied.

        :return:  a list for each post neuron of the corresponding top-left coordinates
        """
        # Generates coordinates TODO: Find a more robust way!
        coords = [[] for i in range(self.post.size)]
        if self.dim_pre == 1:
            rk = 0
            for i in range(self.post.geometry[0]):
                coords[rk] = [i * self.extent[0]]
                rk += 1
        elif self.dim_pre == 2:
            rk = 0
            for i in range(self.post.geometry[0]):
                if self.dim_post > 1:
                    for j in range(self.post.geometry[1]):
                        coords[rk] = [i * self.extent[0], j * self.extent[1]]
                        rk += 1
                else: # over the whole second axis
                    coords[rk] = [i * self.extent[0], 0]
                    rk += 1

        elif self.dim_pre == 3:
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

        elif self.dim_pre == 4: # TODO: post has less than 4 dimensions
            rk = 0
            for i in range(self.post.geometry[0]):
                for j in range(self.post.geometry[1]):
                    for k in range(self.post.geometry[2]):
                        for l in range(self.post.geometry[3]):
                            coords[rk] = [i * self.extent[0], j * self.extent[1], k * self.extent[2], l * self.extent[3]]
                            rk += 1
        # Save the result
        self.pre_coordinates = coords

    def _generate(self):
        """
        Overrides the default code generation.
        """
        # Convolve_code
        convolve_code, sum_code = self._generate_pooling_code()

        # Generate the code
        if _check_paradigm("openmp"):
            self._generate_omp(convolve_code, sum_code)
        elif _check_paradigm("cuda"):
            self._generate_cuda()
        else:
            Messages._error("Pooling: not implemented for the configured paradigm")

    def _generate_pooling_code(self):
        """
        Generate loop statements for the desired pooling operation.
        """
        # Operation to be performed: sum, max, min, mean
        operation = self.synapse_type.operation

        # Main code
        # default value for sum in code depends on operation
        sum_default = "0.0"
        if self.synapse_type.operation == "min":
            sum_default = "std::numeric_limits<%(float_prec)s>::max()" % {'float_prec': get_global_config('precision')}
        elif self.synapse_type.operation == "max":
            sum_default = "std::numeric_limits<%(float_prec)s>::min()" % {'float_prec': get_global_config('precision')}

        code = """
            sum = %(sum_default)s;
""" % {'sum_default': sum_default}

        # Generate for loops
        for dim in range(self.dim_pre):
            ind_dict = {
                'index': indices[dim],
                'size': self.extent[dim]
            }
            if self.extent[dim] > 1:
                code += """
            for(int %(index)s_w = 0; %(index)s_w < %(size)s; %(index)s_w++){
    """ % ind_dict

        # Compute indices
        for dim in range(self.dim_pre):
            ind_dict = {
                'index': indices[dim],
                'dim': dim
            }
            if self.extent[dim] > 1:
                code += """
                int %(index)s_pre = coord[%(dim)s] + %(index)s_w;""" % ind_dict
            else:
                code += """
                int %(index)s_pre = coord[%(dim)s];""" % ind_dict

        # Check indices
        for dim in range(self.dim_pre):
            ind_dict = {
                'index': indices[dim],
                'max_size': self.pre.geometry[dim] - 1
            }
            code += """
                if ((%(index)s_pre < 0) ||(%(index)s_pre > %(max_size)s)){
                    continue;
                }""" % ind_dict

        # Compute pre-synaptic rank
        code += """
                rk_pre = %(value)s;""" % {'value': self._coordinates_to_rank('pre', self.pre.geometry)}

        # Compute the value to pool
        psp = self.synapse_type.description['psp']['cpp'] % {
            'id_pre': self.pre.id,
            'id_post': self.post.id,
            'local_index': '[i][j]',
            'global_index': '[i]',
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]',
            'pre_prefix': 'pop'+str(self.pre.id)+'.',
            'post_prefix': 'pop'+str(self.post.id)+'.'
        }

        # Delays
        if self.delays > get_global_config('dt'):
            psp = psp.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                'pop%(id_pre)s._delayed_r[%(delay)s][rk_pre]' % {'id_pre': self.pre.id, 'delay': str(int(self.delays/get_global_config('dt'))-1)}
            )

        # Apply the operation
        if operation == "max":
            code += """
                %(float_prec)s _psp = %(psp)s;
                if(_psp > sum) sum = _psp;"""
        elif operation == "min":
            code += """
                %(float_prec)s _psp = %(psp)s;
                if(_psp < sum) sum = _psp;"""
        elif operation == "sum":
            code += """
                sum += %(psp)s;"""
        elif operation == "mean":
            code += """
                sum += %(psp)s;"""
        else:
            Messages._error('SharedProjection: Operation', operation, 'is not implemented yet for shared projections with pooling.')

        # Close for loops
        for dim in range(self.dim_pre):
            if self.extent[dim] > 1:
                code += """
            }"""

        impl_code = code % {
            'id_proj': self.id,
            'target': self.target,
            'id_pre': self.pre.id, 'name_pre': self.pre.name, 'size_pre': self.pre.size,
            'id_post': self.post.id, 'name_post': self.post.name, 'size_post': self.post.size,
            'psp': psp,
            'float_prec': get_global_config('precision')
        }

        if operation == "mean":
            size = 1
            for dim in range(self.pre.dimension):
                size *= self.extent[dim]
            sum_code = "sum/" + str(size)
        else:
            sum_code = "sum"

        return impl_code, sum_code

    def _generate_omp(self, convolve_code, sum_code):
        """
        Update the ProjectionGenerator._specific_template structure and bypass the standard openMP code generation.

        :param convolve_code:
        :param sum_code:
        """
        # default value for sum in code depends on operation
        sum_default = "0.0"
        if self.synapse_type.operation == "min":
            sum_default = "std::numeric_limits<%(float_prec)s>::max()" % {'float_prec': get_global_config('precision')}
        elif self.synapse_type.operation == "max":
            sum_default = "std::numeric_limits<%(float_prec)s>::min()" % {'float_prec': get_global_config('precision')}

        # Specific template for generation
        pool_dict = deepcopy(pooling_template_omp)
        for key, value in pool_dict.items():
            value = value % {
                'id_proj': self.id,
                'size_post': self.post.size,
                'sum_default': sum_default,
                'float_prec': get_global_config('precision')
            }
            pool_dict[key] = value
        self._specific_template.update(pool_dict)

        # OMP code
        omp_code = ""
        if get_global_config('num_threads') > 1:
            omp_code = """
        #pragma omp for private(sum, rk_pre, coord) %(psp_schedule)s""" % {
                'psp_schedule': "" if not 'psp_schedule' in self._omp_config.keys() else self._omp_config[
                    'psp_schedule']}

        # HD ( 16.10.2015 ):
        # pre-load delayed firing rate in a local array, so we
        # prevent multiple accesses to pop%(id_pre)s._delayed_r[%(delay)s]
        if self.delays > get_global_config('dt'):
            pre_load_r = """
        // pre-load delayed firing rate
        auto delayed_r = pop%(id_pre)s._delayed_r[%(delay)s];
        """ % {'id_pre': self.pre.id, 'delay': str(int(self.delays / get_global_config('dt')) - 1)}
        else:
            pre_load_r = ""

        # Target variable depends on neuron type
        target_code = "_sum_%(target)s" if self.post.neuron_type.type=="rate" else "g_%(target)s"
        target_code %= {'target': self.target}

        # Compute sum
        wsum = """
        if ( _transmission && pop%(id_pre)s._active ) {
        std::vector<int> coord;
""" + pre_load_r + """
        %(omp_code)s
        for(int i = 0; i < %(size_post)s; i++){
            coord = pre_coords[i];
""" + convolve_code + """
            pop%(id_post)s.%(target)s[i] += """ + sum_code + """;
        } // for
        } // if
"""

        # Delays
        self._specific_template['wrapper_init_delay'] = ""

        # Dictionary keys
        psp_dict = {
            'id_proj': self.id,
            'target': target_code,
            'id_pre': self.pre.id, 'name_pre': self.pre.name,
            'size_pre': self.pre.size,
            'id_post': self.post.id, 'name_post': self.post.name,
            'size_post': self.post.size,
            'omp_code': omp_code,
            'convolve_code': convolve_code
        }

        # Psp code
        self._specific_template['psp_code'] = wsum % psp_dict
        self._specific_template['size_in_bytes'] = """
        // connectivity
        size_in_bytes += sizeof(std::vector<int>);
        size_in_bytes += pre_coords.capacity() * sizeof(int);

        size_in_bytes += sizeof(std::vector<std::vector<int>>);
        size_in_bytes += pre_coords.capacity() * sizeof(std::vector<int>);
        for (auto it = pre_coords.begin(); it != pre_coords.end(); it++) {
            size_in_bytes += it->capacity() * sizeof(int);
        }
"""

        # Clean-up
        self._specific_template['clear_container'] = pooling_template_omp["clear"]

    def _generate_cuda(self):
        """
        Update the ProjectionGenerator._specific_template structure and bypass the standard CUDA code generation.
        """
        # Extract operation and pre-synaptic variable which should be processed
        pool_operation = self.synapse_type.operation
        pre_var = self.synapse_type.psp.split(".")[1]

        # default value for sum in code depends on operation
        sum_default = "0.0"
        if pool_operation == "min":
            sum_default = "FLT_MAX"
        elif pool_operation == "max":
            sum_default = "FLT_MIN"

        # operation to perform
        pool_op_code = cuda_op_code[pool_operation] % {'float_prec': get_global_config('precision')}

        # mean operation requires one additional computation
        if pool_operation == "mean":
            size = 1
            for dim in range(self.pre.dimension):
                size *= self.extent[dim]
            final_result = "psp[bIdx] += local_res/" + str(size) + ";"
        else:
            final_result = "psp[bIdx] += local_res;"

        # result dictionary with code for
        # body, call and header
        pool_template = {}
        base_ids = {
            'id_proj': self.id,
            'id_pre': self.pre.id,
            'id_post': self.post.id,
            'target': self.target,
            'float_prec': get_global_config('precision'),
            'size_post': self.post.size # TODO: population views?
        }

        # The correct templates depends on both
        # kernel-geometry and extent
        if len(self.pre.geometry) == 2:
            # For small extents, we compute multiple coords within one warp. If one extent can fill alone
            # a half-warp we switch to the other implementation.
            if self.extent[0] < 6:

                pool_op_reduce_code = cuda_pooling_code_2d_small_extent['reduce_code'][pool_operation] % {
                    'float_prec': get_global_config('precision'),
                    'row_extent': int(self.extent[0]),
                    'col_extent': int(self.extent[1])
                }

                pool_dict = deepcopy(base_ids)
                pool_dict.update({
                    'sum_default': sum_default,
                    'row_extent': int(self.extent[0]),
                    'col_extent': int(self.extent[1]),
                    'row_size': int(self.pre.geometry[0]),
                    'col_size': int(self.pre.geometry[1]),
                    'operation': tabify(pool_op_code, 3),
                    'operation_reduce': pool_op_reduce_code,
                    'final_result': final_result
                })

                pool_template['psp_body'] = cuda_pooling_code_2d_small_extent['psp_body'] % pool_dict
                pool_template['psp_invoke'] = cuda_pooling_code_2d_small_extent['psp_invoke'] % pool_dict
                pool_template['psp_header'] = cuda_pooling_code_2d_small_extent['psp_header'] % pool_dict
                pool_template['psp_call'] = cuda_pooling_code_2d_small_extent['psp_call'] % pool_dict

            else:
                pool_op_reduce_code = cuda_pooling_code_2d['reduce_code'][pool_operation] % {
                    'float_prec': get_global_config('precision'),
                    'row_extent': int(self.extent[0]),
                    'col_extent': int(self.extent[1])
                }

                pool_dict = deepcopy(base_ids)
                pool_dict.update({
                    'sum_default': sum_default,
                    'row_extent': int(self.extent[0]),
                    'col_extent': int(self.extent[1]),
                    'row_size': int(self.pre.geometry[0]),
                    'col_size': int(self.pre.geometry[1]),
                    'operation': tabify(pool_op_code, 3),
                    'operation_reduce': tabify(pool_op_reduce_code, 2),
                    'final_result': final_result
                })

                pool_template['psp_body'] = remove_trailing_spaces(cuda_pooling_code_2d['psp_body'] % pool_dict)
                pool_template['psp_invoke'] = remove_trailing_spaces(cuda_pooling_code_2d['psp_invoke'] % pool_dict)
                pool_template['psp_header'] = cuda_pooling_code_2d['psp_header'] % pool_dict
                pool_template['psp_call'] = cuda_pooling_code_2d['psp_call'] % pool_dict

        elif len(self.pre.geometry) == 3:

            pool_dict = deepcopy(base_ids)
            pool_dict.update({
                'sum_default': sum_default,
                'row_extent': self.extent[0],
                'col_extent': self.extent[1],
                'plane_extent': self.extent[2],
                'row_size': self.pre.geometry[0],
                'col_size': self.pre.geometry[1],
                'plane_size': self.pre.geometry[2],
                'pre_var': pre_var,
                'operation': tabify(pool_op_code, 4),
                'final_result': final_result
            })

            pool_template['psp_body'] = remove_trailing_spaces(cuda_pooling_code_3d['psp_body'] % pool_dict)
            pool_template['psp_invoke'] = remove_trailing_spaces(cuda_pooling_code_3d['psp_invoke'] % pool_dict)
            pool_template['psp_header'] = cuda_pooling_code_3d['psp_header'] % pool_dict
            pool_template['psp_call'] = cuda_pooling_code_3d['psp_call'] % pool_dict

        else:
            raise NotImplementedError

        # Post-neuron is a spike neuron (e.g., part of ANN-to-SNN conversion)
        if self.post.neuron_type.type == "spike":
            pool_template['psp_call'] = pool_template['psp_call'].replace("gpu__sum_"+self.target, "gpu_g_"+self.target)

        # Update psp fields
        self._specific_template.update(pool_template)

        # Specific template for generation (wrapper, etc)
        pool_dict = deepcopy(pooling_template_cuda)
        for key, value in pool_dict.items():
            value = value % base_ids
            pool_dict[key] = value
        self._specific_template.update(pool_dict)

        self._specific_template['wrapper_connector_call'] = ""
        self._specific_template['access_parameters_variables'] = ""

        self._specific_template['size_in_bytes'] = "//TODO:\n"

        # Clean-up
        self._specific_template['clear_container'] = pooling_template_cuda["clear"]

    @staticmethod
    def _coordinates_to_rank(name, geometry):
        """
        Generate the code for array access, for instance used
        for pre-synaptic ranks.
        """
        dim = len(geometry)

        txt = ""

        for d in range(dim):
            if txt == "":   # first coordinate is special
                txt = indices[0] + "_" + name
            else:
                txt = str(geometry[d]) + '*(' + txt + ') + ' + indices[d] + '_' + name

        return txt

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
        Messages._warning('Pooling projections can not be saved.')
    def save(self, filename):
        "Not available."
        Messages._warning('Pooling projections can not be saved.')
    def load(self, filename):
        "Not available."
        Messages._warning('Pooling projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        "Not available."
        Messages._warning('Pooling projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        "Not available."
        Messages._warning('Pooling projections can not display connectivity matrices.')
