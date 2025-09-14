"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core import Global
from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages
from ANNarchy.models.Synapses import DefaultRateCodedSynapse, DefaultSpikingSynapse

class Transpose(SpecificProjection):
    """
    Transposed projection reusing the weights of an already-defined projection. 
    
    Even though the original projection can be learnable, this one can not. The computed post-synaptic potential is the default case for rate-coded projections: "w * pre.r"

    The proposed `target` can differ from the target of the forward projection.

    Example:

    ```python
    proj_ff = net.connect input, output, target="exc" )
    proj_ff.all_to_all(weights=Uniform(0,1)

    proj_fb = net.connect(Transpose(proj_ff, target="inh"))
    proj_fb.transpose()
    ```
    
    :param projection: original projection.
    :param target: type of the connection (can differ from the original one).
    """
    def __init__(self, projection, target, name=None, copied=False, net_id=0):

        # Transpose is not intended for hybrid projections
        if projection.pre.neuron_type.type == "rate" and projection.post.neuron_type.type == "rate":
            SpecificProjection.__init__(
                self,
                pre = projection.post,
                post = projection.pre,
                target = target,
                synapse = DefaultRateCodedSynapse,
                name = name,
                copied=copied,
                net_id=net_id,
            )
        elif projection.pre.neuron_type.type == "spike" and projection.post.neuron_type.type == "spike":
            SpecificProjection.__init__(
                self,
                pre = projection.post,
                post = projection.pre,
                target = target,
                synapse = DefaultSpikingSynapse,
                name = name,
                copied=copied,
                net_id=net_id,
            )
        else:
            Messages._error('TransposeProjection are not applyable on hybrid projections ...')

        # in the code generation we directly access properties of the
        # forward projection. Therefore we store the link here to have access in
        # self._generate()
        self.projection = projection

        if (projection._connection_delay > 0.0):
            Messages._error('TransposeProjection can not be applied on delayed projections yet ...')

        # simply copy from the forward view
        self.delays = projection._connection_delay
        self.max_delay = projection.max_delay
        self.uniform_delay = projection.uniform_delay

    def _copy(self, pre, post, net_id=None):
        "Returns a copy of the projection when creating networks. Internal use only."
        return Transpose(
            projection=self.projection, 
            target=self.target,
            copied=True,
            net_id = self.net_id if not net_id else net_id
        )

    def _create(self):
        pass

    def connect_transpose(self):
        return self.transpose()
    
    def transpose(self):
        # create fake LIL object to have the forward view in C++
        try:
            from ANNarchy.cython_ext.Connector import LILConnectivity
        except Exception as e:
            Messages._print(e)
            Messages._error('ANNarchy was not successfully installed.')

        lil = LILConnectivity(dt=ConfigManager().get('dt', self.net_id))
        lil.max_delay = self.max_delay
        lil.uniform_delay = self.uniform_delay
        self.connector_name = "Transpose"
        self.connector_description = "Transpose"

    def _connect(self, module):
        self.cyInstance = getattr(module, 'proj'+str(self.id)+'_wrapper')()
        return True

    def _generate(self):
        """
        Overrides default code generation. This function is called during the code generation procedure.
        """
        if self.synapse_type.type == "rate":
            self._generate_rate_coded()
        else:
            self._generate_spiking()

    def _generate_rate_coded(self):
        """
        Generates the transpose projection for rate-coded models.
        """
        #
        # C++ definition and PYX wrapper
        self._specific_template['struct_additional'] = """
extern ProjStruct%(fwd_id_proj)s* proj%(fwd_id_proj)s;    // Forward projection
""" % { 'fwd_id_proj': self.projection.id }

        self._specific_template['declare_connectivity_matrix'] = """
    // LIL connectivity (inverse of proj%(id)s)
    std::vector< int > inv_post_rank ;
    std::vector< std::vector< std::pair< int, int > > > inv_pre_rank ;
""" % {'float_prec': ConfigManager().get('precision', self.net_id), 'id': self.projection.id}
        self._specific_template['export_connector_call'] = ""

        # TODO: error message on setter?
        self._specific_template['access_connectivity_matrix'] = """
    // Accessor to connectivity data
    std::vector<int> get_post_rank() {
        return inv_post_rank;
    }
    size_t nb_synapses() {
        size_t size = 0;
        for (auto it = inv_pre_rank.cbegin(); it != inv_pre_rank.cend(); it++) {
            size += it->size();
        }
        return size;
    }
    int nb_dendrites() {
        return inv_post_rank.size();
    }
    int dendrite_size(int lil_idx) {
        return inv_pre_rank[lil_idx].size();
    }
"""
        self._specific_template['init_additional'] = """
        // Inverse connectivity to Proj%(fwd_id_proj)s
        auto inv_conn =  std::map< int, std::vector< std::pair<int, int> > > ();

        for (int i = 0; i < proj%(fwd_id_proj)s->pre_rank.size(); i++) {
            int post_rk = proj%(fwd_id_proj)s->post_rank[i];

            for (int j = 0; j < proj%(fwd_id_proj)s->pre_rank[i].size(); j++ ) {
                int pre_rk = proj%(fwd_id_proj)s->pre_rank[i][j];

                inv_conn[pre_rk].push_back(std::pair<int, int>(i, j));
            }
        }

        // keys are automatically sorted
        for (auto it = inv_conn.begin(); it != inv_conn.end(); it++ ) {
            inv_post_rank.push_back(it->first);
            inv_pre_rank.push_back(it->second);
        }

        // sanity check
        size_t inv_size = 0;
        for (int i = 0; i < inv_post_rank.size(); i++) {
            inv_size += inv_pre_rank[i].size();
        }

        assert( (proj%(fwd_id_proj)s->nb_synapses() == inv_size) );
""" % { 'fwd_id_proj': self.projection.id }


        # No attributes
        self._specific_template['declare_parameters_variables'] = ""
        self._specific_template['export_parameters_variables'] = ""
        self._specific_template['access_parameters_variables'] = ""

        # Override the monitor to avoid recording the weights
        self._specific_template['monitor_class'] = ""
        self._specific_template['monitor_export'] = ""
        self._specific_template['monitor_wrapper'] = ""

        self._specific_template['size_in_bytes'] = ""
        self._specific_template['clear_container'] = ""

        self._specific_template['wrapper'] = """
    // Transpose ProjStruct%(id_proj)s
    nanobind::class_<ProjStruct%(id_proj)s>(m, "proj%(id_proj)s_wrapper")
        // Constructor
        .def(nanobind::init<>())

        // Flags
        .def_rw("_transmission", &ProjStruct%(id_proj)s::_transmission)
        .def_rw("_axon_transmission", &ProjStruct%(id_proj)s::_axon_transmission)
        .def_rw("_update", &ProjStruct%(id_proj)s::_update)
        .def_rw("_plasticity", &ProjStruct%(id_proj)s::_plasticity)

        // Other methods
        .def("clear", &ProjStruct%(id_proj)s::clear);    
    
    """ %  {
                'id_proj': self.id,
                'id_copy': self.projection.id,
                'float_prec': ConfigManager().get('precision', self.net_id)
            }

        # The weight index depends on the
        # weight of the forward projection
        if self.projection._has_single_weight():
            weight_index = ""
        else:
            weight_index = "[post_idx][pre_idx]"

        #
        # PSP code
        self._specific_template['psp_code'] = """
        if (pop%(id_post)s->_active && _transmission) {
            %(omp_code)s
            for (int i = 0; i < inv_post_rank.size(); i++) {
                %(float_prec)s sum = 0.0;

                for (auto it = inv_pre_rank[i].begin(); it != inv_pre_rank[i].end(); it++) {
                    auto post_idx = it->first;
                    auto pre_idx = it->second;

                    sum += pop%(id_pre)s->r[proj%(fwd_id_proj)s->post_rank[post_idx]] * proj%(fwd_id_proj)s->w%(index)s;
                }
                pop%(id_post)s->_sum_%(target)s[inv_post_rank[i]] += sum;
            }
        }
""" % { 'float_prec': ConfigManager().get('precision', self.net_id),
        'target': self.target,
        'id_pre': self.pre.id,
        'id_post': self.post.id,
        'fwd_id_proj': self.projection.id,
        'index': weight_index,
        'omp_code': "" if ConfigManager().get('num_threads', self.net_id) == 1 else "#pragma omp for"
}

    def _generate_spiking(self):
        """
        Generates the transpose projection for spiking models.

        TODO: openMP
        """
        if ConfigManager().get('num_threads', self.net_id) > 1:
            Messages._error('TransposeProjection for spiking projections is only available for single-thread yet ...')

        # Which projection is transposed
        self._specific_template['struct_additional'] = """
extern ProjStruct%(fwd_id_proj)s *proj%(fwd_id_proj)s;    // Forward projection
""" % { 'fwd_id_proj': self.projection.id }

        # Connectivity
        self._specific_template['declare_connectivity_matrix'] = "" # reuse fwd proj data
        self._specific_template['access_connectivity_matrix'] = """
    std::vector<int> get_post_rank() {
        return proj%(fwd_id_proj)s->inv_post_rank;
    }
    size_t nb_synapses() {
        size_t size = 0;
        for (auto it = proj%(fwd_id_proj)s->inv_pre_rank.cbegin(); it != proj%(fwd_id_proj)s->inv_pre_rank.cend(); it++) {
            size += (it->second).size();
        }
        return size;
    }
    int nb_dendrites() {
        return proj%(fwd_id_proj)s->inv_post_rank.size();
    }
    int dendrite_size(int lil_idx) {
        int post_rank = proj%(fwd_id_proj)s->inv_post_rank[lil_idx];
        return proj%(fwd_id_proj)s->inv_pre_rank[post_rank].size();
    }
""" % { 'fwd_id_proj': self.projection.id }

        # The weight index depends on the
        # weight of the forward projection
        if self.projection._has_single_weight():
            weight_index = ""
        else:
            weight_index = "[post_idx][syn_idx]"

        # Computation
        self._specific_template['psp_prefix'] = ""
        self._specific_template['psp_code'] = """
        for (auto it = pop%(id_pre)s->spiked.cbegin(); it != pop%(id_pre)s->spiked.cend(); it++) {
            auto pos_it = std::find(proj%(fwd_id_proj)s->post_rank.cbegin(), proj%(fwd_id_proj)s->post_rank.cend(), *it);
            if (pos_it == proj%(fwd_id_proj)s->post_rank.end())
                continue;

            auto post_idx = std::distance(proj%(fwd_id_proj)s->post_rank.cbegin(), pos_it);

            for (int syn_idx = 0; syn_idx < proj%(fwd_id_proj)s->pre_rank[post_idx].size(); syn_idx++) {
                auto pre_idx = proj%(fwd_id_proj)s->pre_rank[post_idx][syn_idx];
                pop%(id_post)s->g_%(target)s[pre_idx] += proj%(fwd_id_proj)s->w%(weight_index)s;
            }
        }
""" % {
    'id_pre': self.pre.id,
    'id_post': self.post.id,
    'target': self.target,
    'fwd_id_proj': self.projection.id,
    'weight_index': weight_index
}
        
        self._specific_template['wrapper'] = """
    // Transpose ProjStruct%(id_proj)s
    nanobind::class_<ProjStruct%(id_proj)s>(m, "proj%(id_proj)s_wrapper")
        // Constructor
        .def(nanobind::init<>())

        // Flags
        .def_rw("_transmission", &ProjStruct%(id_proj)s::_transmission)
        .def_rw("_axon_transmission", &ProjStruct%(id_proj)s::_axon_transmission)
        .def_rw("_update", &ProjStruct%(id_proj)s::_update)
        .def_rw("_plasticity", &ProjStruct%(id_proj)s::_plasticity)

        // Other methods
        .def("clear", &ProjStruct%(id_proj)s::clear);    
    
    """ %  {
                'id_proj': self.id,
                'id_copy': self.projection.id,
                'float_prec': ConfigManager().get('precision', self.net_id)
            }
        
        # No attributes
        self._specific_template['declare_parameters_variables'] = ""
        self._specific_template['export_parameters_variables'] = ""
        self._specific_template['access_parameters_variables'] = ""

        # Override the monitor to avoid recording the weights
        self._specific_template['monitor_class'] = ""
        self._specific_template['monitor_export'] = ""
        self._specific_template['monitor_wrapper'] = ""

        self._specific_template['size_in_bytes'] = ""
        self._specific_template['clear_container'] = ""

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
        Messages._warning('Transposed projections can not be saved.')
    def save(self, filename):
        "Not available."
        Messages._warning('Transposed projections can not be saved.')
    def load(self, filename):
        "Not available."
        Messages._warning('Transposed projections can not be loaded.')

    # TODO: maybe this functions would be helpful for debugging. Even though
    #       they will be time consuming as the matrix need to be constructed.
    #       (HD, 9th July 2020)
    def receptive_fields(self, variable = 'w', in_post_geometry = True):
        "Not available."
        Messages._warning('Transposed projections can not display receptive fields.')
    def connectivity_matrix(self, fill=0.0):
        "Not available."
        Messages._warning('Transposed projections can not display connectivity matrices.')
