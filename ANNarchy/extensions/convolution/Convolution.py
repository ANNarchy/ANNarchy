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

class Conv1D(Projection):

    def __init__(self, pre, post, target, psp="pre.r * w", operation="sum", method='filter', padding=0.0, stride=1, subsampling=None):
        """
        Conv1D
        """
        self._operation_type = 'convolve'
        self.psp_code = psp
        self.operation = operation
        self.method = method
        self.padding = padding
        self.stride = stride
        self.subsampling = subsampling

        # Create the description, but it will not be used for generation
        Projection.__init__(
            self,
            pre,
            post,
            target,
            synapse=SharedSynapse(psp=psp, operation=operation)
        )

    def connect_kernel(self, weights, delays=0.0):
        ""
        # Process the weights
        if isinstance(weights, list):
            self.weights = np.array(weights)
        else:
            self.weights = weights

        # Process the delays
        self.delays = delays
        if not isinstance(delays, (int, float)):
            Global._error('Conv1D: Convolutions can only have uniform delays.')

        # Check dimensions of populations and weight matrix
        self.dim_kernel = self.weights.ndim
        self.dim_pre = self.pre.dimension
        self.dim_post = self.post.dimension

        # if not self.dim_kernel ==1:
        #     Global._error('Conv1D: the kernel must be a 1D array. Use Conv2D, Conv3D instead.')

        # TODO

        # Generate the pre-synaptic coordinates
        self._generate_pre_coordinates()

        # Finish building the synapses
        self._create()

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
        self.connector_name = "Convolutional projection"
        self.connector_description = "Convolution with a kernel."
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
        self.cyInstance = proj(self.weights.reshape(self.weights.size), self.pre_coordinates)

        # Define the list of postsynaptic neurons
        self.post_ranks = list(range(self.post.size))

        # Set delays after instantiation
        if self.delays > 0.0:
            self.cyInstance.set_delay(self.delays/Global.config['dt'])

    def _generate_pre_coordinates(self):
        " Returns for each post neuron a list of the corresponding pre-ranks."

        # Number of values in the filter = number_pre
        nb_pre = self.weights.size

        # Pre-synaptic ranks
        coords = - np.ones((self.post.size, nb_pre), dtype=np.int32)

        if self.weights.ndim == 1:

            # Indices between -K and K
            K = int(nb_pre/2)

            # Compute pre-indices
            for idx_post in range(self.post.size):
                coords[idx_post, :] = np.array([ (idx_post + i) for i in range(-K, K+1)], dtype=np.int32)

            # Remove coordinates outside the array
            coords[coords < 0] = -1
            coords[coords >= self.post.size] = -1

        else:

            post_coords = np.unravel_index(range(self.post.size), self.post.geometry)
            print(post_coords)

        # Save the result
        self.pre_coordinates = coords


    ################################
    # Code generation
    ################################
    def _generate(self):
        """
        Overrides default code generation. This function is called during the code generation procedure.
        """

        if Global._check_paradigm("openmp"):
            self._generate_omp()
        elif Global._check_paradigm("cuda"):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _generate_omp(self):
        """
        OpenMP code generation.
        """
        # Specific template for generation
        self._specific_template = {
            # Declare the connectivity matrix
            'declare_connectivity_matrix': """
    std::vector<int> post_rank;
    std::vector< std::vector<int> > pre_rank;
    std::vector< %(float_prec)s > w;
    """ % {'float_prec': Global.config['precision']},

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
        self._specific_template['access_connectivity_matrix'] += """
    // Local parameter w
    std::vector< %(float_prec)s > get_w() { return w; }
    void set_w(std::vector< %(float_prec)s > value) { w = value; }
""" % {'float_prec': Global.config['precision']}
        self._specific_template['export_connectivity'] += """
        # Local variable w
        vector[%(float_prec)s] get_w()
        void set_w(vector[%(float_prec)s])
""" % {'float_prec': Global.config['precision']}
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
        tpl = """
        if ( _transmission && pop%(id_pre)s._active ) {

            for(int i = 0; i < pop%(id_post)s.size; i++){
                sum = 0.0;

                for(int id_w = 0; id_w < w.size(); id_w++){
                    rk_pre = pre_rank[i][id_w];
                    if(rk_pre == -1){continue;}
                    sum += pop%(id_pre)s.r[rk_pre] * w[id_w];
                }

                pop%(id_post)s._sum_%(target)s[i] += sum;
            } // for
        } // if
"""

        self._specific_template['psp_code'] = tpl % \
        {   'id_proj': self.id,
            'target': self.target,
            'id_pre': self.pre.id,
            'id_post': self.post.id,
            'float_prec': Global.config['precision']
        }

    ################################
    ### Utilities
    ################################
    def _center_filter(self, i):
        return int(i/2) if i%2==1 else int(i/2)-1

    def _coordinates_to_rank(self, name, geometry):

        dim = len(geometry)

        txt = ""

        for d in range(dim):
            if txt == "" : # first coordinate is special
                txt = indices[0] + "_" + name
            else:
                txt = str(geometry[d]) + '*(' + txt + ') + ' + indices[d]  + '_' + name

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
        Global._warning('Convolutional projections can not be saved.')
    def save(self, filename):
        Global._warning('Convolutional projections can not be saved.')
    def load(self, filename):
        Global._warning('Convolutional projections can not be loaded.')
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        Global._warning('Convolutional projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        Global._warning('Convolutional projections can not display connectivity matrices.')
