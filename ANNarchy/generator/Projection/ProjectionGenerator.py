#===============================================================================
#
#     ProjectionGenerator.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
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
#===============================================================================
from ANNarchy.core import Global

class ProjectionGenerator(object):
    """
    Abstract definition of a ProjectionGenerator.
    """
    _templates = {}
    _connectivity_class = None

    def __init__(self, profile_generator, net_id):
        """
        Initialization of the class object and store some ids.
        """
        super(ProjectionGenerator, self).__init__()

        self._prof_gen = profile_generator
        self._net_id = net_id

    def header_struct(self, proj, annarchy_dir):
        """
        Generate and store the projection code in a single header file. The
        name is defined as
        proj%(id)s.hpp.

        Parameters:

            proj: Projection object
            annarchy_dir: working directory

        Returns:

            (str, str): include directive, pointer definition

        Templates:

            header_struct: basic template

        """
        raise NotImplementedError

    def creating(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def pruning(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _computesum_rate(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _computesum_spiking(self, proj):
        "Implemented by child class"
        raise NotImplementedError

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
        if proj.uniform_delay > 1 :
            key_delay = "uniform"
        else:
            key_delay = "nonuniform"
        declare_delay = self._templates['delay'][key_delay]['declare']
        init_delay = self._templates['delay'][key_delay]['init']

        # Code for declarations and accessors
        accessor = ""

        attributes = []

        # Parameters
        for var in proj.synapse_type.description['parameters']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            # Avoid doublons
            if var['name'] in attributes:
                continue

            ids = {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
            declare_parameters_variables += decl_template[var['locality']] % ids
            accessor += acc_template[var['locality']] % ids

        # Variables
        for var in proj.synapse_type.description['variables']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            # Avoid doublons
            if var['name'] in attributes:
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
                declare_rng += self._templates['rng'][rd['locality']]['decl'] % {
                    'rd_name' : rd['name'],
                    'type': rd['ctype'],
                    'float_prec': Global.config['precision'],
                    'template': rd['template'] % {'float_prec':Global.config['precision']}
                }

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
            'declare_delay': declare_delay,
            'init_delay': init_delay,
            'event_driven': declare_event_driven,
            'rng': declare_rng,
            'parameters_variables': declare_parameters_variables,
            'additional': declare_additional
        }

        return declaration, accessor

    @staticmethod
    def _get_attr_and_type(proj, name):
        """
        Small helper function, used for instance in self.update_spike_neuron().

        For a given variable name, the data container is searched and checked,
        whether it is a local or global variable, a random variable or a
        variable related to global operations.

        **Hint**:

        Returns (None, None) by default, if none of this cases is true, indicating
        an error in code generation procedure.
        """
        desc = proj.synapse_type.description
        for attr in desc['parameters']:
            if attr['name'] == name:
                return 'var', attr

        for attr in desc['variables']:
            if attr['name'] == name:
                return 'par', attr

        for attr in desc['random_distributions']:
            if attr['name'] == name:
                return 'rand', attr

        return None, None

    def _header_structural_plasticity(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _init_parameters_variables(self, proj):
        """
        Generate initialization code for variables / parameters of the
        projection *proj*.

        Returns:

            str: the string contains initialization code.
        """
        # Is it a specific projection?
        if 'init_parameters_variables' in proj._specific_template.keys():
            return proj._specific_template['init_parameters_variables']

        # Learning by default
        code = ""

        # choose initialization templates based on chosen paradigm
        attr_init_tpl = self._templates['attribute_cpp_init']

        attributes = []

        # Initialize parameters
        for var in proj.synapse_type.description['parameters']:
            if var['name'] == 'w':
                continue
            # Avoid doublons
            if var['name'] in attributes:
                continue

            init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
            code += attr_init_tpl[var['locality']] % {
                'id': proj.id,
                'id_post': proj.post.id,
                'name': var['name'],
                'type': var['ctype'],
                'init': init,
                'attr_type': 'parameter'
            }

        # Initialize variables
        for var in proj.synapse_type.description['variables']:
            if var['name'] == 'w':
                continue
            # Avoid doublons
            if var['name'] in attributes:
                continue

            init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
            code += attr_init_tpl[var['locality']] % {
                'id': proj.id, 'name': var['name'],
                'type': var['ctype'], 'init': init,
                'attr_type': 'variable'
            }

        # Pruning
        if Global.config['structural_plasticity']:
            if 'pruning' in proj.synapse_type.description.keys():
                code += """
        // Pruning
        _pruning = false;
        _pruning_period = 1;
        _pruning_offset = 0;
"""% {'id_proj': proj.id}
            if 'creating' in proj.synapse_type.description.keys():
                code += """
        // Creating
        _creating = false;
        _creating_period = 1;
        _creating_offset = 0;
"""% {'id_proj': proj.id}

        return code

    def _init_random_distributions(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _local_functions(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _post_event(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _update_random_distributions(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _update_synapse(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _determine_size_in_bytes(self, proj):
        """
        Generate code template to determine size in bytes for the C++ object *proj*. Please note, that this contain only
        default elementes (parameters, variables). User defined elements, parallelization support data structures or similar
        are not considered.

        Consequently implementing generators should extent the resulting code template. This is done by the 'size_in_bytes'
        field in the _specific_template dictionary.
        """
        if 'size_in_bytes' in proj._specific_template.keys():
            return proj._specific_template['size_in_bytes']

        from ANNarchy.generator.Utils import tabify
        code = ""

        for attr in proj.synapse_type.description['variables']+proj.synapse_type.description['parameters']:
            ids = {'ctype': attr['ctype'], 'name': attr['name'], 'locality': attr['locality']}

            if attr in proj.synapse_type.description['parameters']:
                code += "// %(locality)s parameter %(name)s\n" % ids
            else:
                code += "// %(locality)s variable %(name)s\n" % ids

            if attr['name'] == "w" and proj._has_single_weight():
                code += "size_in_bytes += sizeof(%(ctype)s);\t// %(name)s\n" % ids
                continue

            if attr['locality'] == "global":
                code += "size_in_bytes += sizeof(%(ctype)s);\t// %(name)s\n" % ids
            elif attr['locality'] == "semiglobal":
                code += "size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();\n" % ids
            else:
                if proj._storage_format == "lil":
                    code += """size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();
for(auto it = %(name)s.begin(); it != %(name)s.end(); it++)
    size_in_bytes += (it->capacity()) * sizeof(%(ctype)s);\n""" % ids
                elif proj._storage_format == "csr":
                    code += """size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();""" % ids
                else:
                    # TODO: sanity check???
                    pass
        code = tabify(code, 2)
        return code

    def _clear_container(self, proj):
        """
        Generate code template to destroy allocated container of the C++ object *proj*.

        User defined elements, parallelization support data structures or similar are not considered. Consequently
        implementing generators should extent the resulting code template.
        """
        from ANNarchy.generator.Utils import tabify
        code = ""

        # Variables
        code += "// Variables\n"
        for attr in proj.synapse_type.description['variables']:
            # HD: clear alone does not deallocate, it only resets size.
            #     So we need to call shrink_to_fit afterwards.
            ids = {'ctype': attr['ctype'], 'name': attr['name']}
            code += "%(name)s.clear();\n" % ids
            code += "%(name)s.shrink_to_fit();\n" % ids

        code = tabify(code, 2)
        return code


######################################
### Code generation
######################################
def get_bounds(param):
    """
    Analyses the bounds of a variables used in pre- and post-spike
    statements in a synapse description and returns a code template.
    """
    code = ""
    for bound, val in param['bounds'].items():
        if bound == "init":
            continue

        # Min-Max bounds
        code += """if(%(var)s%(index)s %(operator)s %(val)s)
    %(var)s%(index)s = %(val)s;
""" % {
        'index': "%(local_index)s",
        'var' : param['name'],
        'val' : val,
        'operator': '<' if bound == 'min' else '>'
      }
    return code
