from ANNarchy.core import Global
import CSR_CUDA
import LIL_OpenMP
import Dense_OpenMP

class Connectivity(object):
    """
    Base class to define connectivities in ANNarchy, the derived classes are
    responsible to select the correct templates and store them in the variable
    self._templates
    """
    def __init__(self):
        super(Connectivity, self).__init__()

    def configure(self, proj):
        """
        This function should be called before any other method of the
        Connectivity class is called, as the templates are adapted.
        """
        raise NotImplementedError

    def _connectivity(self, proj):
        """
        Create codes for connectivity, comprising usually of post_rank and
        pre_rank. In case of spiking models they are extended by an inv_rank
        data field. The extension SharedProjection as well as
        SpecificProjection members overwrite the _specific_template member
        variable of the Projection object, to replace directly the default
        codes.

        Returns:

            a dictionary containing the following fields: *declare*, *init*,
            *accessor*, *declare_inverse*, *init_inverse*
        
        TODO:
        
            Some templates require additional information (e. g. pre- or post-
            synaptic population id) and some not. Yet, I simply add the informa-
            tion in all cases, even if there are not absolutely necessary.
        """
        declare_inverse_connectivity_matrix = ""
        init_inverse_connectivity_matrix = ""

        # Retrieve the templates
        connectivity_matrix_tpl = self._templates['connectivity_matrix']

        # Special case if there is a constant weight across all synapses
        if proj._has_single_weight():
            weight_matrix_tpl = self._templates['single_weight_matrix']
        else:
            weight_matrix_tpl = self._templates['weight_matrix']

        # Connectivity
        declare_connectivity_matrix = connectivity_matrix_tpl['declare']
        access_connectivity_matrix = connectivity_matrix_tpl['accessor']
        init_connectivity_matrix = connectivity_matrix_tpl['init'] % {
            'id_pre': proj.pre.id, 'id_post': proj.post.id
        }

        # Weight array
        declare_connectivity_matrix += weight_matrix_tpl['declare']
        access_connectivity_matrix += weight_matrix_tpl['accessor']
        init_connectivity_matrix += weight_matrix_tpl['init']

        # Spiking model require inverted ranks
        if proj.synapse_type.type == "spike":
            inv_connectivity_matrix_tpl = self._templates['inverse_connectivity_matrix']
            declare_inverse_connectivity_matrix = inv_connectivity_matrix_tpl['declare']
            init_inverse_connectivity_matrix = inv_connectivity_matrix_tpl['init']

        # Specific projections can overwrite
        if 'declare_connectivity_matrix' in proj._specific_template.keys():
            declare_connectivity_matrix = proj._specific_template['declare_connectivity_matrix']
        if 'access_connectivity_matrix' in proj._specific_template.keys():
            access_connectivity_matrix = proj._specific_template['access_connectivity_matrix']
        if 'declare_inverse_connectivity_matrix' in proj._specific_template.keys():
            declare_inverse_connectivity_matrix = proj._specific_template['declare_inverse_connectivity_matrix']
        if 'init_connectivity_matrix' in proj._specific_template.keys():
            init_connectivity_matrix = proj._specific_template['init_connectivity_matrix']
        if 'init_inverse_connectivity_matrix' in proj._specific_template.keys():
            init_inverse_connectivity_matrix = proj._specific_template['init_inverse_connectivity_matrix']

        return {
            'declare' : declare_connectivity_matrix,
            'init' : init_connectivity_matrix,
            'accessor' : access_connectivity_matrix,
            'declare_inverse': declare_inverse_connectivity_matrix,
            'init_inverse': init_inverse_connectivity_matrix
        }

    def _declaration_accessors(self, proj):
        """
        Generate declaration and accessor code for variables/parameters of the projection.

        Returns:
            (dict, str): first return value contain declaration code and last one the accessor code.

            The declaration dictionary has the following fields:
                delay, event_driven, rng, parameters_variables, additional, cuda_stream
        """
        # create the code for non-specific projections
        declare_event_driven = ""
        declare_rng = ""
        declare_parameters_variables = ""
        declare_additional = ""

        # choose templates dependend on the paradigm
        decl_template = self._templates['attribute_decl']
        acc_template = self._templates['attribute_acc']

        # Delays
        declare_delay = self._templates['delay']['declare']

        # Code for declarations and accessors
        accessor = ""
        # Parameters
        for var in proj.synapse_type.description['parameters']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue

            ids = {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
            declare_parameters_variables += decl_template[var['locality']] % ids
            accessor += acc_template[var['locality']] % ids

        # Variables
        for var in proj.synapse_type.description['variables']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue

            ids = {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}
            declare_parameters_variables += decl_template[var['locality']] % ids
            accessor += acc_template[var['locality']] % ids

        # If no psp is defined, it's event-driven
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break
        if has_event_driven:
            declare_event_driven = self._templates['event_driven']['declare']

        # Arrays for the random numbers
        if len(proj.synapse_type.description['random_distributions']) > 0:
            declare_rng += """
    // Random numbers
"""
            for rd in proj.synapse_type.description['random_distributions']:
                declare_rng += """    std::vector< std::vector<double> > %(rd_name)s;
    %(template)s dist_%(rd_name)s;
""" % {'rd_name' : rd['name'], 'template': rd['template']}

        # Local functions
        if len(proj.synapse_type.description['functions']) > 0:
            declare_parameters_variables += """
    // Local functions
"""
            for func in proj.synapse_type.description['functions']:
                declare_parameters_variables += ' '*4 + func['cpp'] + '\n'

        # Structural plasticity
        if Global.config['structural_plasticity']:
            declare_parameters_variables += self._header_structural_plasticity(proj)

        # Specific projections can overwrite
        if 'declare_parameters_variables' in proj._specific_template.keys():
            declare_parameters_variables = proj._specific_template['declare_parameters_variables']
        if 'declare_rng' in proj._specific_template.keys():
            declare_rng = proj._specific_template['declare_rng']
        if 'declare_event_driven' in proj._specific_template.keys():
            declare_event_driven = proj._specific_template['declare_event_driven']
        if 'declare_delay' in proj._specific_template.keys():
            declare_delay = proj._specific_template['declare_delay']
        if 'declare_additional' in proj._specific_template.keys():
            declare_additional = proj._specific_template['declare_additional']
        if 'access_parameters_variables' in proj._specific_template.keys():
            accessor = proj._specific_template['access_parameters_variables']

        # Finalize the declarations
        declaration = {
            'delay': declare_delay,
            'event_driven': declare_event_driven,
            'rng': declare_rng,
            'parameters_variables': declare_parameters_variables,
            'additional': declare_additional
        }

        return declaration, accessor

class OpenMPConnectivity(Connectivity):
    """
    Implementor class to define connectivities in ANNarchy for single- and multi-core CPUs.
    """
    def __init__(self):
        super(OpenMPConnectivity, self).__init__()

    def configure(self, proj):
        self._templates.update(LIL_OpenMP.conn_templates)

class CUDAConnectivity(Connectivity):
    """
    Implementor class to define connectivities in ANNarchy for Nvidia CUDA.
    """
    def __init__(self):
        """
        Initialization of the class and updates the _templates variable of
        ProjectionGenerator with some fields.
        """
        super(CUDAConnectivity, self).__init__()

    def configure(self, proj):
        self._templates.update(CSR_CUDA.conn_templates)
