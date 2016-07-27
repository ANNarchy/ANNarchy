"""

    PopulationGenerator.py

    This file is part of ANNarchy.

    Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from ANNarchy.core import Global

class PopulationGenerator(object):
    """
    Base class for population generators in ANNarchy.
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

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            declaration += attr_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
            accessors += acc_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}

        # Variables
        for var in pop.neuron_type.description['variables']:
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
                declaration += self._templates['rate_psp']['decl'] % {'target': target}

        # Global operations
        declaration += """
    // Global operations
"""
        for op in pop.global_operations:
            if Global.config['paradigm'] == "openmp":
                declaration += """    double _%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}
            elif Global.config['paradigm'] == "cuda":
                declaration += """    double _%(op)s_%(var)s;
    double *_gpu_%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}
            else:
                Global._error("Internal: acc/decl of global operations are not implemented for: " + Global.config['paradigm'])

        # Arrays for the random numbers
        declaration += """
    // Random numbers
"""
        for rd in pop.neuron_type.description['random_distributions']:
            declaration += self._templates['rng'][rd['locality']]['decl'] % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}

        # Local functions
        if len(pop.neuron_type.description['functions']) > 0:
            declaration += """
    // Local functions
"""
            for func in pop.neuron_type.description['functions']:
                declaration += ' '*4 + func['cpp'] + '\n'

        return declaration, accessors

    def _get_attr(self, pop, name):
        """
        Small helper function, used for instance in self.update_spike_neuron_cuda
        """
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] == name:
                return attr

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
    cudaMalloc((void**)&_gpu_%(op)s_%(var)s, sizeof(double));
""" % {'op': op['function'], 'var': op['variable']}
            else:
                raise NotImplementedError

        return code

    def _init_population(self, pop):
        """
        Generate the codes for the C++ function Population::init_population() method.
        """
        code = ""
        attr_tpl = self._templates['attribute_cpp_init']

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            init = 0.0 if var['ctype'] == 'double' else 0
            code += attr_tpl[var['locality']] % {'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init, 'attr_type': 'parameter'}

        # Variables
        for var in pop.neuron_type.description['variables']:
            init = 0.0 if var['ctype'] == 'double' else 0
            code += attr_tpl[var['locality']] % {'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init, 'attr_type': 'variable'}

        # Random numbers
        if len(pop.neuron_type.description['random_distributions']) > 0:
            code += """
        // Random numbers"""
            for rd in pop.neuron_type.description['random_distributions']:
                code += self._templates['rng'][rd['locality']]['init'] % {'id': pop.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': pop.id}}

        # Global operations
        code += self._init_globalops(pop)

        # Targets, only if rate-code
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                code += self._templates['rate_psp']['init'] % {'id': pop.id, 'target': target}

        return code

    def _stop_condition(self, pop):
        """
        Simulation can either end after a fixed point in time or
        dependent on a population related condition. The code for
        this is generated here and added to the ANNarchy.cpp/.cu
        file.
        """
        if not pop.stop_condition: # no stop condition has been defined
            return ""

        # Process the stop condition
        pop.neuron_type.description['stop_condition'] = {'eq': pop.stop_condition}
        from ANNarchy.parser.Extraction import extract_stop_condition
        extract_stop_condition(pop.neuron_type.description)

        # Retrieve the code
        condition = pop.neuron_type.description['stop_condition']['cpp']% {
            'id': pop.id,
            'local_index': "[i]",
            'global_index': ''}

        # Generate the function
        if pop.neuron_type.description['stop_condition']['type'] == 'any':
            stop_code = """
    // Stop condition (any)
    bool stop_condition(){
        for(int i=0; i<size; i++)
        {
            if(%(condition)s){
                return true;
            }
        }
        return false;
    }
    """ % {'condition': condition}
        else:
            stop_code = """
    // Stop condition (all)
    bool stop_condition(){
        for(int i=0; i<size; i++)
        {
            if(!(%(condition)s)){
                return false;
            }
        }
        return true;
    }
    """ % {'condition': condition}

        return stop_code

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
