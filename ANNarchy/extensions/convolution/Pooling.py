# =============================================================================
#
#     Pooling.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2018-2019  Julien Vitay <julien.vitay@gmail.com>,
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
from ANNarchy.core.Projection import Projection
from ANNarchy.core.Synapse import Synapse
from ANNarchy.core import Global
from ANNarchy.generator.Utils import tabify

from copy import deepcopy

from .PoolingTemplate import *
from .Utils import SharedSynapse

# Indices used for each dimension
indices = ['i', 'j', 'k', 'l', 'm', 'n']

class Pooling(Projection):
    """
    Performs a pooling operation (e.g. max.pooling) on the pre-synaptic population.

    Each post-synaptic neuron covers a specific region (``extent``) of the pre-synaptic
    population, over which the result of the operation on firing rates will be
    assigned to sum(target).

    The extent is automatically computed using the geometry of the populations, but can be specified in the `connect_pooling()`` methods.

    Example:

    ```python
    inp = Population(geometry=(100, 100), neuron=Neuron(parameters="r = 0.0"))
    pop = Population(geometry=(50, 50), neuron=Neuron(equations="r = sum(exc)"))
    proj = Pooling(inp, pop, 'exc', operation='max') # max-pooling
    proj.connect_pooling() # extent=(2, 2) is implicit
    ```
    """
    def __init__(self, pre, post, target, operation="max", name=None, copied=False):
        """
        :param pre: pre-synaptic population (either its name or a ``Population`` object).
        :param post: post-synaptic population (either its name or a ``Population`` object).
        :param target: type of the connection
        :param operation: pooling function to be applied ("max", "min", "mean")
        """
        if not operation in ["max", "mean", "min"]:
            Global._error("Pooling: the operation must be either 'max', 'mean' or 'min'.")
        self.operation = operation

        Projection.__init__(
            self,
            pre,
            post,
            target,
            synapse=SharedSynapse(psp="pre.r", operation=operation, name="Pooling operation", description=operation+"-pooling operation over the pre-synaptic population."),
            name=name,
            copied=copied
        )

        if not pre.neuron_type.type == 'rate':
            Global._error('Pooling: only implemented for rate-coded populations.')

        # check dimensions of populations, should not exceed 4
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension
        if self.dim_post > 4:
            Global._error('Pooling: Too many dimensions for the post-synaptic population (maximum 4).')
        if self.dim_pre > 4:
            Global._error('Pooling: Too many dimensions for the pre-synaptic population (maximum 4).')

        # Disable saving
        self._saveable = False



    def connect_pooling(self, extent=None, delays=0.0):
        """
        :param extent: extent of the pooling area expressed in the geometry of the pre-synaptic population (e.g ``(2, 2)``). In each dimension, the product of this extent with the number of neurons in the post-synaptic population must be equal to the number of pre-synaptic neurons. Default: None.
        :param delays: synaptic delay in ms
        """

        # process extent
        self.extent_init = extent
        if extent is None:  # compute the extent automatically
            if self.pre.dimension != self.post.dimension:
                Global._error(
                    'Pooling: If you do not provide the extent parameter, the two populations must have the same number of dimensions.')

            extent = list(self.pre.geometry)
            for dim in range(self.pre.dimension):
                extent[dim] /= self.post.geometry[dim]
                if self.pre.geometry[dim] != extent[dim] * self.post.geometry[dim]:
                    Global._error(
                        'Pooling: Unable to compute the extent of the pooling area: the number of neurons do not match.')

        elif not isinstance(extent, tuple):
            Global._error('Pooling: You must provide a tuple for the extent of the pooling operation.')

        self.extent = list(extent)
        if len(self.extent) < self.pre.dimension:
            Global._error('Pooling: You must provide a tuple for the extent of the pooling operation.')

        # process delays
        self.delays = delays

        # Generate the pre-synaptic coordinates
        self._generate_extent_coordinates()

        # create fake LIL
        self._create()

        return self

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks.  Internal use only."
        return NotImplementedError
        
    def _create(self):
        """
        create fake LIL object, just for compilation process

        :return: no return value
        """
        try:
            from ANNarchy.core.cython_ext.Connector import LILConnectivity
        except Exception as e:
            Global._print(e)
            Global._error('ANNarchy was not successfully installed.')

        lil = LILConnectivity()
        lil.max_delay = self.delays
        lil.uniform_delay = self.delays
        self.connector_name = "Pooling"
        self.connector_description = "Pooling"
        self._store_connectivity(self._load_from_lil, (lil, ), self.delays)

    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """
        if not self._connection_method:
            Global._error(
                'Pooling: The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')

        # Create the Cython instance
        proj = getattr(module, 'proj' + str(self.id) + '_wrapper')
        self.cyInstance = proj([], self.pre_coordinates)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

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
        if Global._check_paradigm("openmp"):
            self._generate_omp(convolve_code, sum_code)
        elif Global._check_paradigm("cuda"):
            self._generate_cuda(convolve_code, sum_code)
        else:
            Global._error("Pooling: not implemented for the configured paradigm")

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
            sum_default = "std::numeric_limits<%(float_prec)s>::max()" % {'float_prec': Global.config['precision']}
        elif self.synapse_type.operation == "max":
            sum_default = "std::numeric_limits<%(float_prec)s>::min()" % {'float_prec': Global.config['precision']}

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
        if self.delays > Global.config['dt']:
            psp = psp.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                'pop%(id_pre)s._delayed_r[%(delay)s][rk_pre]' % {'id_pre': self.pre.id, 'delay': str(int(self.delays/Global.config['dt'])-1)}
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
            Global._error('SharedProjection: Operation', operation, 'is not implemented yet for shared projections with pooling.')

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
            'float_prec': Global.config['precision']
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
            sum_default = "std::numeric_limits<%(float_prec)s>::max()" % {'float_prec': Global.config['precision']}
        elif self.synapse_type.operation == "max":
            sum_default = "std::numeric_limits<%(float_prec)s>::min()" % {'float_prec': Global.config['precision']}

        # Specific template for generation
        pool_dict = deepcopy(pooling_template_omp)
        for key, value in pool_dict.items():
            value = value % {
                'id_proj': self.id,
                'size_post': self.post.size,
                'sum_default': sum_default,
                'float_prec': Global.config['precision']
            }
            pool_dict[key] = value
        self._specific_template.update(pool_dict)

        # OMP code
        omp_code = ""
        if Global.config['num_threads'] > 1:
            omp_code = """
        #pragma omp parallel for private(sum, rk_pre, coord) %(psp_schedule)s""" % {
                'psp_schedule': "" if not 'psp_schedule' in self._omp_config.keys() else self._omp_config[
                    'psp_schedule']}

        # HD ( 16.10.2015 ):
        # pre-load delayed firing rate in a local array, so we
        # prevent multiple accesses to pop%(id_pre)s._delayed_r[%(delay)s]
        if self.delays > Global.config['dt']:
            pre_load_r = """
        // pre-load delayed firing rate
        auto delayed_r = pop%(id_pre)s._delayed_r[%(delay)s];
        """ % {'id_pre': self.pre.id, 'delay': str(int(self.delays / Global.config['dt']) - 1)}
        else:
            pre_load_r = ""

        # Compute sum
        wsum = """
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

        # Delays
        self._specific_template['wrapper_init_delay'] = ""

        # Psp code
        self._specific_template['psp_code'] = wsum % \
                                              {'id_proj': self.id,
                                               'target': self.target,
                                               'id_pre': self.pre.id, 'name_pre': self.pre.name,
                                               'size_pre': self.pre.size,
                                               'id_post': self.post.id, 'name_post': self.post.name,
                                               'size_post': self.post.size,
                                               'omp_code': omp_code,
                                               'convolve_code': convolve_code
                                               }
        self._specific_template['size_in_bytes'] = "//TODO:\n"

    def _generate_cuda(self, convolve_code, sum_code):
        """
        Update the ProjectionGenerator._specific_template structure and bypass the standard CUDA code generation.
        """
        pool_operation = self.synapse_type.operation

        # default value for sum in code depends on operation
        sum_default = "0.0"
        if pool_operation == "min":
            sum_default = "FLT_MAX"
        elif pool_operation == "max":
            sum_default = "FLT_MIN"

        pool_template = ""
        pool_op_code = cuda_op_code[pool_operation]
        pool_dict = {
            'sum_default': sum_default,
            'float_prec': Global.config['precision'],
        }
        if len(self.pre.geometry) == 2:
            pool_template = cuda_pooling_code_2d
            pool_dict.update({
                'row_extent': self.extent[0],
                'col_extent': self.extent[1],
                'col_size': self.pre.geometry[1],
                'operation': tabify(pool_op_code, 3)
            })
            pooling_code = pool_template % pool_dict
        elif len(self.pre.geometry) == 3:
            pool_template = cuda_pooling_code_3d
            pool_dict.update({
                'row_extent': self.extent[0],
                'col_extent': self.extent[1],
                'plane_extent': self.extent[2],
                'row_size': self.pre.geometry[0],
                'col_size': self.pre.geometry[1],
                'plane_size': self.pre.geometry[2],
                'operation': tabify(pool_op_code, 4)
            })
            pooling_code = pool_template % pool_dict
        else:
            raise NotImplementedError

        # Specific template for generation
        pool_dict = deepcopy(pooling_template_cuda)
        for key, value in pool_dict.items():
            value = value % {
                'id_proj': self.id,
                'id_pre': self.pre.id,
                'id_post': self.post.id,
                'size_post': self.post.size,
                'target': self.target,
                'float_prec': Global.config['precision'],
                'pooling_code': pooling_code
            }
            pool_dict[key] = value
        self._specific_template.update(pool_dict)
        self._specific_template['size_in_bytes'] = "//TODO:\n"

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
        Global._warning('Pooling projections can not be saved.')
    def save(self, filename):
        "Not available."
        Global._warning('Pooling projections can not be saved.')
    def load(self, filename):
        "Not available."
        Global._warning('Pooling projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        "Not available."
        Global._warning('Pooling projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        "Not available."
        Global._warning('Pooling projections can not display connectivity matrices.')
