#===============================================================================
#
#     PopulationGenerator.py
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

class PopulationGenerator(object):
    """
    Base class for population generators in ANNarchy. Inherited by

    * OpenMPGenerator: single-thread and multi-core implementation
    * CUDAGenerator: gpu implementation
    """
    _templates = {}

    def __init__(self, profile_generator, net_id):
        """
        Initialize PopulationGenerator.
        """
        self._prof_gen = profile_generator
        self._net_id = net_id

    def header_struct(self, pop, annarchy_dir):
        """
        Generate the c-style struct definition for a population object.

        Parameters:

            pop: Population object
            annarchy_dir: working directory

        Returns:

            dictionary: include directive, pointer definition, call statements
                        and other informations needed by CodeGenerator to
                        create the ANNarchy.cpp/.cu code.

                        Please note, the returned dictionary may vary dependent
                        on population type and used parallelization paradigm.

        Exceptions:

            Will cause a NotImplementedError as it should implemented by the
            inheriting generator classes.
        """
        raise NotImplementedError

    def reset_computesum(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _delay_code(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _generate_decl_and_acc(self, pop):
        """
        Data exchange between Python and ANNarchyCore library is done by get-
        and set-methods. This function creates for all variables and parameters
        the corresponding methods.
        """
        # Pick basic template based on neuron type
        attr_template = self._templates['attr_decl']
        acc_template = self._templates['attr_acc']

        declaration = "" # member declarations
        accessors = "" # export member functions
        attributes = []

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])
            declaration += attr_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
            accessors += acc_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}

        # Variables
        for var in pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])
            declaration += attr_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}
            accessors += acc_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}

        # Arrays for the presynaptic sums for rate-coded neurons.
        # Important: the conductance/current variables for spiking
        # neurons are stored in pop.neuron_type.description['variables'].
        if pop.neuron_type.type == 'rate':
            declaration += """
    // Targets
"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                declaration += self._templates['rate_psp']['decl'] % {'target': target, 'float_prec': Global.config['precision']}

        else:
            # HD: the above statement is only true, if the target is used in the equations
            try:
                all_targets = set(pop.neuron_type.description['targets'] + pop.targets)
            except TypeError:
                # The projection has multiple targets
                all_targets = set(pop.neuron_type.description['targets'] + pop.targets[0])

            for target in sorted(list(all_targets)):
                attr_name = 'g_'+target
                if attr_name not in attributes:
                    id_dict = {
                        'type' : Global.config['precision'],
                        'name': attr_name,
                        'attr_type': 'variable'
                    }
                    declaration += attr_template[var['locality']] % id_dict
                    accessors += acc_template[var['locality']] % id_dict

        # Global operations
        declaration += """
    // Global operations
"""
        for op in pop.global_operations:
            op_dict = {'type': Global.config['precision'], 'op': op['function'], 'var': op['variable']}
            if Global.config['paradigm'] == "openmp":
                declaration += """    %(type)s _%(op)s_%(var)s;
""" % op_dict
            elif Global.config['paradigm'] == "cuda":
                declaration += """    %(type)s _%(op)s_%(var)s;
    %(type)s *_gpu_%(op)s_%(var)s;
""" % op_dict
            else:
                Global._error("Internal: acc/decl of global operations are not implemented for: " + Global.config['paradigm'])

        # Arrays for the random numbers
        declaration += """
    // Random numbers
"""
        for rd in pop.neuron_type.description['random_distributions']:
            declaration += self._templates['rng'][rd['locality']]['decl'] % {
                'rd_name' : rd['name'],
                'type': rd['ctype'],
                'template': rd['template'] % {'float_prec':Global.config['precision']}
            }

        return declaration, accessors

    @staticmethod
    def _get_attr(pop, name):
        """
        Small helper function, used for instance in self.update_spike_neuron()
        """
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] == name:
                return attr

        return None

    @staticmethod
    def _get_attr_and_type(pop, name):
        """
        Small helper function, used for instance in self.update_spike_neuron().

        For a given variable name, the data container is searched and checked,
        whether it is a local or global variable, a random variable or a
        variable related to global operations.
        """
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] == name:
                return 'attr', attr

        for attr in pop.neuron_type.description['random_distributions']:
            if attr['name'] == name:
                return 'rand', attr

        # the given name wasn't either an attribute nor a random distribution,
        # lets test if it was a psp
        for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
            if name == "sum("+target+")":
                return 'psp', { 'ctype': Global.config['precision'], 'name': '_sum_'+target }

        print(name, "not found")
        return None, None

    def _init_fr(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _init_globalops(self, pop):
        """
        Generate the C++ codes for the initialization of global operations
        within the Population::init_population() method.
        """
        if len(pop.global_operations) == 0:
            return ""

        code = "// Initialize global operations\n"
        for op in pop.global_operations:
            if Global.config['paradigm'] == "openmp":
                code += """    _%(op)s_%(var)s = 0.0;
""" % {'op': op['function'], 'var': op['variable']}
            elif Global.config['paradigm'] == "cuda":
                code += """    _%(op)s_%(var)s = 0.0;
    cudaMalloc((void**)&_gpu_%(op)s_%(var)s, sizeof(%(type)s));
""" % {'op': op['function'], 'var': op['variable'], 'type': Global.config['precision']}
            else:
                raise NotImplementedError

        return code

    def _init_random_dist(self, pop):
        raise NotImplementedError

    def _init_population(self, pop):
        """
        Generate the codes for the C++ function Population::init_population() method.
        """
        code = ""
        attr_tpl = self._templates['attribute_cpp_init']
        attributes = []

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
            var_ids = {'id': pop.id, 'name': var['name'], 'type': var['ctype'],
                       'init': init, 'attr_type': 'parameter'}
            code += attr_tpl[var['locality']] % var_ids

        # Variables
        for var in pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
            var_ids = {'id': pop.id, 'name': var['name'], 'type': var['ctype'],
                       'init': init, 'attr_type': 'variable'}
            code += attr_tpl[var['locality']] % var_ids

        # Random numbers
        code += self._init_random_dist(pop)[1]

        # Global operations
        code += self._init_globalops(pop)

        # Targets, only if rate-code
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                code += self._templates['rate_psp']['init'] % {'id': pop.id, 'target': target, 'float_prec': Global.config['precision']}

        return code

    def _local_functions(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _stop_condition(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_fr(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_globalops(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_random_distributions(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_rate_neuron(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_spiking_neuron(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _determine_size_in_bytes(self, pop):
        """
        Generate code template to determine size in bytes for the C++ object *pop*. Please note, that this contain only
        default elements (parameters, variables). User defined elements, parallelization support data structures or
        similar are not considered.

        Consequently implementing generators should extent the resulting code template. This is done by filling the
        'size_in_bytes' field in the _specific_template.
        """
        if 'size_in_bytes' in pop._specific_template.keys():
            return pop._specific_template['size_in_bytes']

        from ANNarchy.generator.Utils import tabify
        code = ""

        # Parameters
        code += "// Parameters\n"
        for attr in pop.neuron_type.description['parameters']:
            ids = {'ctype': attr['ctype'], 'name': attr['name']}
            if attr['locality'] == "global":
                code += "size_in_bytes += sizeof(%(ctype)s);\t// %(name)s\n" % ids
            else:
                code += "size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();\t// %(name)s\n" % ids

        # Variables
        code += "// Variables\n"
        for attr in pop.neuron_type.description['variables']:
            ids = {'ctype': attr['ctype'], 'name': attr['name']}
            if attr['locality'] == "global":
                code += "size_in_bytes += sizeof(%(ctype)s);\t// %(name)s\n" % ids
            else:
                code += "size_in_bytes += sizeof(%(ctype)s) * %(name)s.capacity();\t// %(name)s\n" % ids

        code = tabify(code, 2)
        return code

    def _clear_container(self, pop):
        """
        Generate code template to destroy allocated container of the C++ object *pop*.

        User defined elements, parallelization support data structures or similar are not considered. Consequently
        implementing generators should extent the resulting code template.
        """
        from ANNarchy.generator.Utils import tabify
        code = ""

        # Variables
        code += "// Variables\n"
        for attr in pop.neuron_type.description['variables']:
            # HD: we need to clear only local variables, the others are no vectors
            if attr['locality'] == "local":
                # HD: clear alone does not deallocate, it only resets size.
                #     So we need to call shrink_to_fit afterwards.
                ids = {'ctype': attr['ctype'], 'name': attr['name']}
                code += "%(name)s.clear();\n" % ids
                code += "%(name)s.shrink_to_fit();\n" % ids

        code = tabify(code, 2)
        return code
