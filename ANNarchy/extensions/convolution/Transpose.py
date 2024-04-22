"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core import Global
from ANNarchy.intern.SpecificProjection import SpecificProjection
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages
from ANNarchy.models.Synapses import DefaultRateCodedSynapse, DefaultSpikingSynapse

class Transpose(SpecificProjection):
    """
    Transposed projection reusing the weights of an already-defined rate-coded projection. 
    
    Even though the original projection can be learnable, this one can not. The computed post-synaptic potential is the default case for rate-coded projections: "w * pre.r"

    The proposed `target` can differ from the target of the forward projection.

    Example:

    ```python
    proj_ff = ann.Projection( input, output, target="exc" )
    proj_ff.connect_all_to_all(weights=Uniform(0,1)

    proj_fb = Transpose(proj_ff, target="inh")
    proj_fb.connect()
    ````
    
    :param proj: original projection.
    :param target: type of the connection (can differ from the original one).

    """
    def __init__(self, proj, target):

        # Transpose is not intended for hybrid projections
        if proj.pre.neuron_type.type == "rate" and proj.post.neuron_type.type == "rate":
            SpecificProjection.__init__(
                self,
                pre = proj.post,
                post = proj.pre,
                target = target,
                synapse = DefaultRateCodedSynapse
            )
        elif proj.pre.neuron_type.type == "spike" and proj.post.neuron_type.type == "spike":
            SpecificProjection.__init__(
                self,
                pre = proj.post,
                post = proj.pre,
                target = target,
                synapse = DefaultSpikingSynapse
            )
        else:
            Messages._error('TransposeProjection are not applyable on hybrid projections ...')

        # in the code generation we directly access properties of the
        # forward projection. Therefore we store the link here to have access in
        # self._generate()
        self.fwd_proj = proj

        if (proj._connection_delay > 0.0):
            Messages._error('TransposeProjection can not be applied on delayed projections yet ...')

        # simply copy from the forward view
        self.delays = proj._connection_delay
        self.max_delay = proj.max_delay
        self.uniform_delay = proj.uniform_delay

    def _copy(self):
        raise NotImplementedError

    def _create(self):
        pass

    def connect(self):
        # create fake LIL object to have the forward view in C++
        try:
            from ANNarchy.cython_ext.Connector import LILConnectivity
        except Exception as e:
            Messages._print(e)
            Messages._error('ANNarchy was not successfully installed.')

        lil = LILConnectivity()
        lil.max_delay = self.max_delay
        lil.uniform_delay = self.uniform_delay
        self.connector_name = "Transpose"
        self.connector_description = "Transpose"

    def _connect(self, module):
        proj = getattr(module, 'proj'+str(self.id)+'_wrapper')
        self.cyInstance = proj()

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
extern ProjStruct%(fwd_id_proj)s proj%(fwd_id_proj)s;    // Forward projection
""" % { 'fwd_id_proj': self.fwd_proj.id }

        self._specific_template['declare_connectivity_matrix'] = """
    // LIL connectivity (inverse of proj%(id)s)
    std::vector< int > inv_post_rank ;
    std::vector< std::vector< std::pair< int, int > > > inv_pre_rank ;
""" % {'float_prec': get_global_config('precision'), 'id': self.fwd_proj.id}
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

        for (int i = 0; i < proj%(fwd_id_proj)s.pre_rank.size(); i++) {
            int post_rk = proj%(fwd_id_proj)s.post_rank[i];

            for (int j = 0; j < proj%(fwd_id_proj)s.pre_rank[i].size(); j++ ) {
                int pre_rk = proj%(fwd_id_proj)s.pre_rank[i][j];

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

        assert( (proj%(fwd_id_proj)s.nb_synapses() == inv_size) );
""" % { 'fwd_id_proj': self.fwd_proj.id }

        self._specific_template['wrapper_connector_call'] = ""
        self._specific_template['wrapper_init_connectivity'] = """
        pass
"""
        self._specific_template['wrapper_access_connectivity'] = """
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def nb_synapses(self):
        return proj%(id_proj)s.nb_synapses()
    def nb_dendrites(self):
        return proj%(id_proj)s.nb_dendrites()
    def dendrite_size(self, lil_idx):
        return proj%(id_proj)s.dendrite_size(lil_idx)
""" % { 'id_proj': self.id }

        # memory management
        self._specific_template['size_in_bytes'] = ""
        self._specific_template['clear_container'] = ""

        #
        # suppress monitor
        self._specific_template['monitor_export'] = ""
        self._specific_template['monitor_wrapper'] = ""
        self._specific_template['monitor_class'] = ""
        self._specific_template['pyx_wrapper'] = ""

        # The weight index depends on the
        # weight of the forward projection
        if self.fwd_proj._has_single_weight():
            weight_index = ""
        else:
            weight_index = "[post_idx][pre_idx]"

        #
        # PSP code
        self._specific_template['psp_code'] = """
        if (pop%(id_post)s._active && _transmission) {
            %(omp_code)s
            for (int i = 0; i < inv_post_rank.size(); i++) {
                %(float_prec)s sum = 0.0;

                for (auto it = inv_pre_rank[i].begin(); it != inv_pre_rank[i].end(); it++) {
                    auto post_idx = it->first;
                    auto pre_idx = it->second;

                    sum += pop%(id_pre)s.r[proj%(fwd_id_proj)s.post_rank[post_idx]] * proj%(fwd_id_proj)s.w%(index)s;
                }
                pop%(id_post)s._sum_%(target)s[inv_post_rank[i]] += sum;
            }
        }
""" % { 'float_prec': get_global_config('precision'),
        'target': self.target,
        'id_pre': self.pre.id,
        'id_post': self.post.id,
        'fwd_id_proj': self.fwd_proj.id,
        'index': weight_index,
        'omp_code': "" if get_global_config('num_threads') == 1 else "#pragma omp for"
}

    def _generate_spiking(self):
        """
        Generates the transpose projection for spiking models.

        TODO: openMP
        """
        if get_global_config('num_threads') > 1:
            Messages._error('TransposeProjection for spiking projections is only available for single-thread yet ...')

        # Which projection is transposed
        self._specific_template['struct_additional'] = """
extern ProjStruct%(fwd_id_proj)s proj%(fwd_id_proj)s;    // Forward projection
""" % { 'fwd_id_proj': self.fwd_proj.id }

        # Connectivity
        self._specific_template['declare_connectivity_matrix'] = "" # reuse fwd proj data
        self._specific_template['access_connectivity_matrix'] = """
    std::vector<int> get_post_rank() {
        return proj%(fwd_id_proj)s.inv_post_rank;
    }
    size_t nb_synapses() {
        size_t size = 0;
        for (auto it = proj%(fwd_id_proj)s.inv_pre_rank.cbegin(); it != proj%(fwd_id_proj)s.inv_pre_rank.cend(); it++) {
            size += (it->second).size();
        }
        return size;
    }
    int nb_dendrites() {
        return proj%(fwd_id_proj)s.inv_post_rank.size();
    }
    int dendrite_size(int lil_idx) {
        int post_rank = proj%(fwd_id_proj)s.inv_post_rank[lil_idx];
        return proj%(fwd_id_proj)s.inv_pre_rank[post_rank].size();
    }
""" % { 'fwd_id_proj': self.fwd_proj.id }
        self._specific_template['export_connector_call'] = ""
        self._specific_template['export_connectivity'] = """
        size_t nb_synapses()
        int nb_dendrites()
        int dendrite_size(int)
        vector[int] get_post_rank()
"""
        self._specific_template['wrapper_init_connectivity'] = """
        pass
"""
        self._specific_template['wrapper_access_connectivity'] = """
    def post_rank(self):
        return proj%(id_proj)s.get_post_rank()
    def nb_synapses(self):
        return proj%(id_proj)s.nb_synapses()
    def nb_dendrites(self):
        return proj%(id_proj)s.nb_dendrites()
    def dendrite_size(self, lil_idx):
        return proj%(id_proj)s.dendrite_size(lil_idx)
""" % { 'id_proj': self.id }
        self._specific_template['wrapper_connector_call'] = ""

        # The weight index depends on the
        # weight of the forward projection
        if self.fwd_proj._has_single_weight():
            weight_index = ""
        else:
            weight_index = "[post_idx][syn_idx]"

        # Computation
        self._specific_template['psp_prefix'] = ""
        self._specific_template['psp_code'] = """
        for (auto it = pop%(id_pre)s.spiked.cbegin(); it != pop%(id_pre)s.spiked.cend(); it++) {
            auto pos_it = std::find(proj%(fwd_id_proj)s.post_rank.cbegin(), proj%(fwd_id_proj)s.post_rank.cend(), *it);
            if (pos_it == proj%(fwd_id_proj)s.post_rank.end())
                continue;

            auto post_idx = std::distance(proj%(fwd_id_proj)s.post_rank.cbegin(), pos_it);

            for (int syn_idx = 0; syn_idx < proj%(fwd_id_proj)s.pre_rank[post_idx].size(); syn_idx++) {
                auto pre_idx = proj%(fwd_id_proj)s.pre_rank[post_idx][syn_idx];
                pop%(id_post)s.g_%(target)s[pre_idx] += proj%(fwd_id_proj)s.w%(weight_index)s;
            }
        }
""" % {
    'id_pre': self.pre.id,
    'id_post': self.post.id,
    'target': self.target,
    'fwd_id_proj': self.fwd_proj.id,
    'weight_index': weight_index
}
        # transpose means we use the forward view of the target matrix

        
        #
        # suppress monitor
        self._specific_template['monitor_export'] = ""
        self._specific_template['monitor_wrapper'] = ""
        self._specific_template['monitor_class'] = ""
        self._specific_template['pyx_wrapper'] = ""

        # Others
        self._specific_template['size_in_bytes'] = "//TODO:"
        self._specific_template['clear'] = "//TODO:"


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
