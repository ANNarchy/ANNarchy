# =============================================================================
#
#     CopyProjection.py
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
from copy import deepcopy

from ANNarchy.core.Projection import Projection
from ANNarchy.core import Global
from ANNarchy.extensions.convolution import ConvolutionProjection, PoolingProjection

from .CopyTemplate import copy_proj_template, copy_sum_template
from .Utils import SharedSynapse

class CopyProjection(Projection):
    """
    Creates a virtual connection pattern reusing the weights and delays of an already-defined projection.

    Although the original projection can be learnable, this one can not. Changes in the original weights will be reflected in this projection. The only possible modifications are ``psp`` and ``operation``.

    The pre- and post-synaptic populations of each projection must have the same geometry.
    """
    def __init__(self, projection):
        """
        Constructor

        *Parameters*:

        * **projection**: the projection to reuse.
        """
        self._operation_type = 'copy'
        self.projection = projection

        # Create the description, but it will not be used for generation
        Projection.__init__(
            self,
            pre=projection.pre,
            post=projection.post,
            target=projection.target,
            synapse = SharedSynapse(psp=projection.psp, operation=projection.operation),
            name=projection.name,
            copied=True
        )

        # Sanity checks
        if not isinstance(self.projection, Projection):
            Global._error('CopyProjection: You must provide an existing projection to copy().')
            
        if isinstance(self.projection, (ConvolutionProjection, PoolingProjection)):
            Global._error('CopyProjection: You can only copy regular projections, not shared projections.')
            
        if not self.pre.geometry == self.projection.pre.geometry or not self.post.geometry == self.projection.post.geometry:
            Global._error('CopyProjection: When copying a projection, the geometries must be the same.')
            
        # Dummy weights
        self.weights = None
        self.pre_coordinates = []

        # Finish building the synapses
        self._create()

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks. Internal use only."
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

    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """
        if not self._connection_method:
            Global._error('CopyProjection: The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')
            
        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights, self.pre_coordinates)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Set delays after instantiation
        if self.delays > 0.0:
            self.cyInstance.set_delay(self.delays/Global.config['dt'])

    def _generate(self):
        """
        Overrides default code generation. This function is called during the code generation procedure.
        """
        if Global._check_paradigm("openmp"):
            self._generate_omp()
        elif Global._check_paradigm("cuda"):
            self._generate_cuda
        else:
            raise NotImplementedError

    def generate_omp(self):
        """
        Code generation of CopyProjection object for the openMP paradigm.
        """
        # Set the projection specific parameters
        copy_proj_dict = deepcopy(copy_proj_template)
        for key, value in copy_proj_dict.items():
            value = value % {
                'id_proj': self.id,
                'id_copy': self.projection.id,
                'float_prec': Global.config['precision']
            }
            copy_proj_dict[key] = value
        
        # Update specific template
        self._specific_template.update(copy_proj_dict)

        # OMP code if more then one thread
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

        # Select template for operation to be performed: sum, max, min, mean
        try:
            sum_code = copy_sum_template[self.synapse_type.operation]
        except KeyError:
            Global._error("CopyProjection: the operation ", self.synapse_type.operation, ' is not available.')

        # Finalize code
        self.generator['omp']['body_compute_psp'] = sum_code % {
            'id_proj': self.id, 'target': self.target,
            'id_pre': self.pre.id, 'name_pre': self.pre.name,
            'id_post': self.post.id, 'name_post': self.post.name,
            'id': self.projection.id,
            'float_prec': Global.config['precision'],
            'omp_code': omp_code,
            'psp': psp
        }

    def _generate_cuda(self):
        """
        Code generation of CopyProjection object for the CUDA paradigm.

        Note: currently not implemented (TODO HD)
        """
        raise NotImplementedError

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
        Global._warning('Copied projections can not be saved.')
    def save(self, filename):
        Global._warning('Copied projections can not be saved.')
    def load(self, filename):
        Global._warning('Copied projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        Global._warning('Copied projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        Global._warning('Copied projections can not display connectivity matrices.')
