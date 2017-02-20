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
        if len(pop.neuron_type.description['random_distributions']) > 0:
            code += """
        // Random numbers"""
            for rd in pop.neuron_type.description['random_distributions']:
                # in principal only important for openmp
                rng_def = {
                    'id': pop.id,
                    'float_prec': Global.config['precision']
                }
                # RNG declaration, only for openmp
                rng_ids = {
                    'id': pop.id,
                    'rd_name': rd['name'],
                    'type': rd['ctype'],
                    'rd_init': rd['definition'] % rng_def
                }
                code += self._templates['rng'][rd['locality']]['init'] % rng_ids

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
