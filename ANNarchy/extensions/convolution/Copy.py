"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from copy import deepcopy

from ANNarchy.core import Global
from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.intern import Messages
from ANNarchy.core.Projection import Projection
from ANNarchy.extensions.convolution import Convolution, Pooling

from .CopyTemplate import copy_proj_template, copy_sum_template
from .Utils import SharedSynapse

class Copy(SpecificProjection):
    """
    Creates a virtual projection reusing the weights and delays of an already-defined projection.

    Although the original projection can be learnable, this one can not. Changes in the original weights will be reflected in this projection. The only possible modifications are ``psp`` and ``operation``.

    The pre- and post-synaptic populations of both projections must have the same geometry.

    Example:

    ```python
    proj = ann.Projection(pop1, pop2, "exc")
    proj.connect_fixed_probability(0.1, 0.5)

    copy_proj = Copy(pop1, pop3, "exc")
    copy_proj.connect_copy(proj)
    ```

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
            pre=pre,
            post=post,
            target=target,
            synapse = SharedSynapse(psp=psp, operation=operation),
            name=name,
            copied=copied
        )

    def connect_copy(self, projection):
        """
        :param projection: Existing projection to copy.
        """
        self.projection = projection

        # Sanity checks
        if not isinstance(self.projection, Projection):
            Messages._error('Copy: You must provide an existing projection to copy().')

        if isinstance(self.projection, (Convolution, Pooling)):
            Messages._error('Copy: You can only copy regular projections, not shared projections.')

        if not self.pre.geometry == self.projection.pre.geometry or not self.post.geometry == self.projection.post.geometry:
            Messages._error('Copy: When copying a projection, the geometries must be the same.')

        # Dummy weights
        self.weights = None
        self.pre_coordinates = []

        # Finish building the synapses
        self._create()

        return self

    def _copy(self, pre, post):
        "Returns a copy of the projection when creating networks. Internal use only."
        raise NotImplementedError

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
        self.connector_name = "Copy"
        self.connector_description = "Copy projection"
        self._store_connectivity(self._load_from_lil, (lil, ), self.delays)

    def _connect(self, module):
        """
        Builds up dendrites either from list or dictionary. Called by instantiate().
        """
        if not self._connection_method:
            Messages._error('Copy: The projection between ' + self.pre.name + ' and ' + self.post.name + ' is declared but not connected.')

        # Create the Cython instance
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj(self.weights, self.pre_coordinates)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Set delays after instantiation
        if self.delays > 0.0:
            self.cyInstance.set_delay(self.delays/get_global_config('dt'))

        return True

    def _generate(self):
        """
        Overrides default code generation. This function is called during the code generation procedure.
        """
        if _check_paradigm("openmp"):
            self._generate_omp()
        elif _check_paradigm("cuda"):
            self._generate_cuda()
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
                'float_prec': get_global_config('precision')
            }
            copy_proj_dict[key] = value

        # Update specific template
        self._specific_template.update(copy_proj_dict)

        # OMP code if more then one thread
        if get_global_config('num_threads') > 1:
            omp_code = '#pragma omp for private(sum)' if self.post.size > Global.OMP_MIN_NB_NEURONS else ''
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
        if self.delays > get_global_config('dt'):
            psp = psp.replace(
                'pop%(id_pre)s.r[rk_pre]' % {'id_pre': self.pre.id},
                'pop%(id_pre)s._delayed_r[delay-1][rk_pre]' % {'id_pre': self.pre.id}
                # TODO HD: wouldn't it be much better to reduce delay globaly, instead of the substraction here???
            )

        # Select template for operation to be performed: sum, max, min, mean
        try:
            sum_code = copy_sum_template[self.synapse_type.operation]
        except KeyError:
            Messages._error("CopyProjection: the operation ", self.synapse_type.operation, ' is not available.')

        # Finalize code
        self.generator['omp']['body_compute_psp'] = sum_code % {
            'id_proj': self.id, 'target': self.target,
            'id_pre': self.pre.id, 'name_pre': self.pre.name,
            'id_post': self.post.id, 'name_post': self.post.name,
            'id': self.projection.id,
            'float_prec': get_global_config('precision'),
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
        "Not available."
        Messages._warning('Copied projections can not be saved.')
    def save(self, filename):
        "Not available."
        Messages._warning('Copied projections can not be saved.')
    def load(self, filename):
        "Not available."
        Messages._warning('Copied projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        "Not available."
        Messages._warning('Copied projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        "Not available."
        Messages._warning('Copied projections can not display connectivity matrices.')
