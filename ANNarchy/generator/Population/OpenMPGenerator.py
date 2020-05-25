#===============================================================================
#
#     OpenMPGenerator.py
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
import datetime

from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_extern as global_op_extern_dict
from ANNarchy.generator.Utils import generate_equation_code, tabify, remove_trailing_spaces
from ANNarchy.core import Global
from ANNarchy import __release__

from .PopulationGenerator import PopulationGenerator
from .OpenMPTemplates import openmp_templates

class OpenMPGenerator(PopulationGenerator):
    """
    Generate the header for a Population object to run either on single core
    or multi-cores with OpenMP.
    """
    _templates = openmp_templates

    def __init__(self, profile_generator, net_id):
        super(OpenMPGenerator, self).__init__(profile_generator, net_id)

    ##################################################
    # Main method
    ##################################################
    def header_struct(self, pop, annarchy_dir):
        """
        Specialized implementation of PopulationGenerator.header_struct() for
        generation of an openMP header.

        two passes:

            * generate the codes for population header
            * fill the dictionary with call codes (return)
        """
        # Generate declaration and accessors of all parameters and variables
        declaration_parameters_variables, access_parameters_variables = self._generate_decl_and_acc(pop)

        # Additional includes and structures
        include_additional = ""
        access_additional = ""
        struct_additional = ""
        declare_additional = ""
        init_additional = ""
        reset_additional = ""

        # Declare global operations as extern at the beginning of the file
        extern_global_operations = ""
        for op in pop.global_operations:
            extern_global_operations += global_op_extern_dict[op['function']] % {'type': Global.config['precision']}

        # Initialize parameters and variables
        init_parameters_variables = self._init_population(pop)

        # Spike-specific stuff
        reset_spike = ""; declare_spike = ""; init_spike = ""
        if pop.neuron_type.description['type'] == 'spike':
            spike_specific_tpl = self._templates['spike_specific']

            # Main data for spiking pops
            declare_spike += spike_specific_tpl['spike']['declare'] % {'id': pop.id}
            init_spike += spike_specific_tpl['spike']['init'] % {'id': pop.id}
            reset_spike += spike_specific_tpl['spike']['reset'] % {'id': pop.id}

            # If there is a refractory period
            if pop.neuron_type.refractory or pop.refractory:
                declare_spike += spike_specific_tpl['refractory']['declare'] % {'id': pop.id}
                if isinstance(pop.neuron_type.description['refractory'], str): # no need to instantiate refractory
                    init_spike += spike_specific_tpl['refractory']['init_extern'] % {'id': pop.id}
                else:
                    init_spike += spike_specific_tpl['refractory']['init'] % {'id': pop.id}
                reset_spike += spike_specific_tpl['refractory']['reset'] % {'id': pop.id}

            # If axonal spike condition was defined
            if pop.neuron_type.axon_spike:
                declare_spike += spike_specific_tpl['axon_spike']['declare']
                init_spike += spike_specific_tpl['axon_spike']['init']
                reset_spike += spike_specific_tpl['axon_spike']['reset']

        # Process eventual delay
        declare_delay = ""; init_delay = ""; update_delay = ""; update_max_delay = ""; reset_delay = ""
        if pop.max_delay > 1:
            declare_delay, init_delay, update_delay, update_max_delay, reset_delay = self._delay_code(pop)

        # Process mean FR computations
        declare_FR, init_FR = self._init_fr(pop)

        # Init rng_dist
        init_rng_dist, _ = self._init_random_dist(pop)

        # Update random distributions
        update_rng = self._update_random_distributions(pop)

        # Update global operations
        update_global_ops = self._update_globalops(pop)

        # Defintion of local functions
        declaration_parameters_variables += self._local_functions(pop)

        # Update the neural variables
        if pop.neuron_type.type == 'rate':
            update_variables = self._update_rate_neuron(pop)
            test_spike_cond = ""
        else:
            update_variables, test_spike_cond = self._update_spiking_neuron(pop)

        # Stop condition
        stop_condition = self._stop_condition(pop)

        # Memory management
        determine_size_in_bytes = self._determine_size_in_bytes(pop)
        clear_container = self._clear_container(pop)

        # Profiling
        if self._prof_gen:
            include_profile = """#include "Profiling.h"\n"""
            declare_profile, init_profile = self._prof_gen.generate_init_population(pop)
        else:
            include_profile = ""
            init_profile = ""
            declare_profile = ""

        ## When everything is generated, we override the fields defined by the specific population
        if 'include_additional' in pop._specific_template.keys():
            include_additional = pop._specific_template['include_additional']
        if 'struct_additional' in pop._specific_template.keys():
            struct_additional = pop._specific_template['struct_additional']
        if 'extern_global_operations' in pop._specific_template.keys():
            extern_global_operations = pop._specific_template['extern_global_operations']
        if 'declare_spike_arrays' in pop._specific_template.keys():
            declare_spike = pop._specific_template['declare_spike_arrays']
        if 'declare_parameters_variables' in pop._specific_template.keys():
            declaration_parameters_variables = pop._specific_template['declare_parameters_variables']
        if 'declare_additional' in pop._specific_template.keys():
            declare_additional = pop._specific_template['declare_additional']
        if 'declare_FR' in pop._specific_template.keys():
            declare_FR = pop._specific_template['declare_FR']
        if 'declare_delay' in pop._specific_template.keys() and pop.max_delay > 1:
            declare_delay = pop._specific_template['declare_delay']
        if 'access_parameters_variables' in pop._specific_template.keys():
            access_parameters_variables = pop._specific_template['access_parameters_variables']
        if 'access_additional' in pop._specific_template.keys():
            access_additional = pop._specific_template['access_additional']
        if 'init_parameters_variables' in pop._specific_template.keys():
            init_parameters_variables = pop._specific_template['init_parameters_variables']
        if 'init_spike' in pop._specific_template.keys():
            init_spike = pop._specific_template['init_spike']
        if 'init_delay' in pop._specific_template.keys() and pop.max_delay > 1:
            init_delay = pop._specific_template['init_delay']
        if 'init_FR' in pop._specific_template.keys():
            init_FR = pop._specific_template['init_FR']
        if 'init_additional' in pop._specific_template.keys():
            init_additional = pop._specific_template['init_additional']
        if 'reset_spike' in pop._specific_template.keys():
            reset_spike = pop._specific_template['reset_spike']
        if 'reset_delay' in pop._specific_template.keys() and pop.max_delay > 1:
            reset_delay = pop._specific_template['reset_delay']
        if 'reset_additional' in pop._specific_template.keys():
            reset_additional = pop._specific_template['reset_additional']
        if 'update_variables' in pop._specific_template.keys():
            update_variables = pop._specific_template['update_variables']
        if 'test_spike_cond' in pop._specific_template.keys():
            test_spike_cond = pop._specific_template['test_spike_cond']
        if 'update_rng' in pop._specific_template.keys():
            update_rng = pop._specific_template['update_rng']
        if 'update_delay' in pop._specific_template.keys() and pop.max_delay > 1:
            update_delay = pop._specific_template['update_delay']
        if 'update_max_delay' in pop._specific_template.keys() and pop.max_delay > 1:
            update_max_delay = pop._specific_template['update_max_delay']
        if 'update_global_ops' in pop._specific_template.keys():
            update_global_ops = pop._specific_template['update_global_ops']

        # Fill the template
        code = self._templates['population_header'] % {
            # some information for
            'annarchy_version': __release__,
            #'time_stamp': '{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()),
            # fill code templates
            'float_prec': Global.config['precision'],
            'id': pop.id,
            'name': pop.name,
            'size': pop.size,
            'include_additional': include_additional,
            'include_profile': include_profile,
            'struct_additional': struct_additional,
            'extern_global_operations': extern_global_operations,
            'declare_spike_arrays': declare_spike,
            'declare_parameters_variables': declaration_parameters_variables,
            'declare_additional': declare_additional,
            'declare_delay': declare_delay,
            'declare_FR': declare_FR,
            'declare_profile': declare_profile,
            'access_parameters_variables': access_parameters_variables,
            'access_additional': access_additional,
            'init_parameters_variables': init_parameters_variables,
            'init_spike': init_spike,
            'init_delay': init_delay,
            'init_FR': init_FR,
            'init_additional': init_additional,
            'init_rng_dist': init_rng_dist,
            'init_profile': init_profile,
            'reset_spike': reset_spike,
            'reset_delay': reset_delay,
            'reset_additional': reset_additional,
            'update_variables': update_variables,
            'test_spike_cond': test_spike_cond,
            'update_rng': update_rng,
            'update_delay': update_delay,
            'update_max_delay': update_max_delay,
            'update_global_ops': update_global_ops,
            'stop_condition': stop_condition,
            'determine_size': determine_size_in_bytes,
            'clear_container': clear_container
        }

        # remove right-trailing spaces
        code = remove_trailing_spaces(code)

        # Store the complete header definition in a single file
        with open(annarchy_dir+'/generate/net'+str(self._net_id)+'/pop'+str(pop.id)+'.hpp', 'w') as ofile:
            ofile.write(code)

        # Basic informations common to all populations
        pop_desc = {
            'include': """#include "pop%(id)s.hpp"\n""" % {'id': pop.id},
            'extern': """extern PopStruct%(id)s pop%(id)s;\n"""% {'id': pop.id},
            'instance': """PopStruct%(id)s pop%(id)s;\n"""% {'id': pop.id},
            'init': """    pop%(id)s.init_population();\n""" % {'id': pop.id}
        }

        # Generate the calls to be made in the main ANNarchy.cpp
        if len(pop.neuron_type.description['variables']) > 0 or 'update_variables' in pop._specific_template.keys():
            if update_variables != "":
                pop_desc['update'] = """    pop%(id)s.update();\n""" % {'id': pop.id}

        if len(pop.neuron_type.description['random_distributions']) > 0:
            pop_desc['rng_update'] = """    pop%(id)s.update_rng();\n""" % {'id': pop.id}

        if pop.max_delay > 1:
            pop_desc['delay_update'] = """    pop%(id)s.update_delay();\n""" % {'id': pop.id}

        if len(pop.global_operations) > 0:
            pop_desc['gops_update'] = """    pop%(id)s.update_global_ops();\n""" % {'id': pop.id}

        return pop_desc

    def _init_random_dist(self, pop):
        """
        Initialize random distribution sources.

        Parameters:
            * *pop* Population object

        Return:
            * code piece to initialize contained random objects.
        """
        target_container_code = ""
        dist_code = ""
        if len(pop.neuron_type.description['random_distributions']) > 0:

            for rd in pop.neuron_type.description['random_distributions']:
                if Global._check_paradigm("openmp"):
                    # in principal only important for openmp
                    rng_def = {
                        'id': pop.id,
                        'float_prec': Global.config['precision'],
                        'global_index': ''
                    }

                    # RNG declaration, only for openmp
                    rng_ids = {
                        'id': pop.id,
                        'rd_name': rd['name'],
                        'type': rd['ctype'],
                        'rd_init': rd['definition'] % rng_def,
                    }
                    target_container_code += self._templates['rng'][rd['locality']]['init'] % rng_ids
                    dist_code += self._templates['rng'][rd['locality']]['init_dist'] % rng_ids
                else:
                    # Nothing to do here:
                    #   CUDA initializes in his inherited function
                    pass

        return dist_code, target_container_code

    ##################################################
    # Reset compute sums
    ##################################################
    def reset_computesum(self, pop):
        """
        For rate-coded neurons each step the weighted sum of inputs is computed. The implementation
        codes of the computes_rate kernel expect zeroed arrays. The same applies for the AccProjections
        used for the computation of the BOLD signal.

        Hint: this method is called directly by CodeGenerator.
        """
        code = ""

        # HD: use set to remove doublons
        for target in sorted(set(pop.targets)):
            code += self._templates['rate_psp']['reset'] % {
                'id': pop.id,
                'name': pop.name,
                'target': target,
                'float_prec': Global.config['precision']
            }

        return code

    ##################################################
    # Delays
    ##################################################
    def _delay_code(self, pop):
        """
        Generate code for delayed variables, comprising of initialization
        and update codes.

        Parameters:
            * population object

        Templates:
            attribute_delayed

        TODO:
            extract several templates, reorganize template dictionary
        """
        # Retrieve the template
        delay_tpl = self._templates['attribute_delayed']

        # Declaration
        declare_code = """
    // Delayed variables"""

        if pop.neuron_type.type == "rate":
            for var in pop.delayed_variables:
                attr = self._get_attr(pop, var)
                attr_dict = {'name': attr['name'], 'type': attr['ctype']}

                if attr['locality'] == "local":
                    declare_code += """
    std::deque< std::vector< %(type)s > > _delayed_%(name)s; """ % attr_dict
                else:
                    declare_code += """
    std::deque< %(type)s > _delayed_%(name)s; """ % attr_dict
        else:
            # Spiking networks should only exchange spikes
            declare_code += """
    // Delays for spike population
    std::deque< std::vector<int> > _delayed_spike;
"""
            for var in pop.delayed_variables:
                attr = self._get_attr(pop, var)
                attr_dict = {'name': attr['name'], 'type': attr['ctype']}

                if attr['locality'] == "local":
                    declare_code += """
    std::deque< std::vector< %(type)s > > _delayed_%(name)s; """ % attr_dict
                else:
                    declare_code += """
    std::deque< %(type)s > _delayed_%(name)s; """ % attr_dict

        # Initialization
        init_code = """
        // Delayed variables"""
        update_code = ""
        reset_code = ""
        resize_code = """
        if(value <= max_delay){ // nothing to do
            return;
        }
        max_delay = value;
        """
        for var in pop.delayed_variables:
            attr = self._get_attr(pop, var)
            init_code += delay_tpl[attr['locality']]['init'] % {'name': attr['name'], 'type': attr['ctype']}
            update_code += delay_tpl[attr['locality']]['update'] % {'name' : var}
            reset_code += delay_tpl[attr['locality']]['reset'] % {'id': pop.id, 'name' : var}
            resize_code += delay_tpl[attr['locality']]['resize'] % {'id': pop.id, 'name' : var, 'type': attr['ctype']}

        # Delaying spike events is done differently
        if pop.neuron_type.type == 'spike':
            init_code += """
        _delayed_spike = std::deque< std::vector<int> >(max_delay, std::vector<int>());"""

            update_code += """
            _delayed_spike.push_front(spiked);
            _delayed_spike.pop_back();
"""
            reset_code += """
        _delayed_spike.clear();
        _delayed_spike = std::deque< std::vector<int> >(max_delay, std::vector<int>());"""

            resize_code += """
        _delayed_spike.resize(max_delay, std::vector<int>());
"""

        update_code = """
        if ( _active ) {
%(code)s
        }""" % {'code': update_code}

        return declare_code, init_code, update_code, resize_code, reset_code

    ##################################################
    # Local functions
    ##################################################
    def _local_functions(self, pop):
        """
        Definition of user-defined local functions attached to
        a neuron. These functions will take place in the
        population header.
        """
        # Local functions
        if len(pop.neuron_type.description['functions']) == 0:
            return ""

        declaration = """
    // Local functions
"""
        for func in pop.neuron_type.description['functions']:
            declaration += ' '*4 + func['cpp'] + '\n'

        return declaration

    ##################################################
    # Stop condition
    ##################################################
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
            'semiglobal_index': '',
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


    ##################################################
    # Mean firing rate
    ##################################################
    def _init_fr(self, pop):
        "Declares arrays for computing the mean FR of a spiking neuron"
        declare_FR = ""; init_FR = ""
        if pop.neuron_type.description['type'] == 'spike':
            declare_FR = """
    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    %(float_prec)s _mean_fr_rate;
    void compute_firing_rate(%(float_prec)s window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = 1000./window;
        }
    };""" % {'float_prec': Global.config['precision']}
            init_FR = """
        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;"""

        return declare_FR, init_FR

    def _update_fr(self, pop):
        "Computes the average firing rate based on history"
        mean_FR_push = ""; mean_FR_update = ""
        if pop.neuron_type.description['type'] == 'spike':
            mean_FR_push = """
                    // Update the mean firing rate
                    if(_mean_fr_window> 0)
                        _spike_history[i].push(t);
            """
            mean_FR_update = """
                // Update the mean firing rate
                if(_mean_fr_window> 0){
                    while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                        _spike_history[i].pop(); // Suppress spikes outside the window
                    }
                    r[i] = _mean_fr_rate * %(float_prec)s(_spike_history[i].size());
                }
            """ % {'float_prec': Global.config['precision']}

        return mean_FR_push, mean_FR_update

    ##################################################
    # Global operations
    ##################################################
    def _update_globalops(self, pop):
        """
        Update of global functions is a call of pre-implemented
        functions defined in GlobalOperationTemplate. In case of
        OpenMP this calls will take place in the population header.
        """
        if len(pop.global_operations) == 0:
            return ""

        code = ""
        for op in pop.global_operations:
            code += """
            _%(op)s_%(var)s = %(op)s_value(%(var)s.data(), size);
""" % {'op': op['function'], 'var': op['variable']}

        return """
    if ( _active ){
%(code)s
    }""" % {'code': code}

    def _update_random_distributions(self, pop):
        """
        Generate the C++ for drawing pseudo-random numbers in each step.
        The random variables are drawn sequentially from the same source.
        """
        if len(pop.neuron_type.description['random_distributions']) == 0:
            return ""

        res = """
        if (_active){
%(update_rng_global)s
            for(int i = 0; i < size; i++) {
%(update_rng_local)s
            }
        }
        """
        local_code = ""
        global_code = ""
        for rd in pop.neuron_type.description['random_distributions']:
            if rd['locality'] == 'local':
                local_code += self._templates['rng'][rd['locality']]['update'] % {'id': pop.id, 'rd_name': rd['name']}
            else:
                global_code += self._templates['rng'][rd['locality']]['update'] % {'id': pop.id, 'rd_name': rd['name']}

        # Final code consists of local and global variables
        final_code = res % {
            'update_rng_local': local_code,
            'update_rng_global': global_code
        }

        # if profiling enabled, annotate with profiling code
        if self._prof_gen:
            final_code = self._prof_gen.annotate_update_rng(pop, final_code)

        return final_code

    ##################################################
    # Neural variables
    ##################################################
    def _update_rate_neuron(self, pop):
        """
        Generate the code template for neural update step, more precise updating of variables.
        The code comprise of two major parts: global and local update, second one parallelized
        with an openmp for construct, if number of threads is greater than one and the number
        of neurons exceed a minimum amount of neurons ( defined as Global.OMP_MIN_NB_NEURONS)
        """
        code = ""
        id_dict = {
            'id': pop.id,
            'local_index': "[i]",
            'semiglobal_index': '',
            'global_index': ''
        }

        # Random distributions
        deps =[]
        for rd in pop.neuron_type.description['random_distributions']:
            for dep in rd['dependencies']:
                deps += dep

        # Global variables
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, locality='global', padding=3)
        eqs = eqs % id_dict
        if eqs.strip() != "":
            code += """
            // Updating the global variables
%(eqs)s
""" % {'eqs': eqs}

        # Gather pre-loop declaration (dt/tau for ODEs)
        pre_code =""
        for var in pop.neuron_type.description['variables']:
            if 'pre_loop' in var.keys() and len(var['pre_loop']) > 0:
                pre_code += var['ctype'] + ' ' + var['pre_loop']['name'] + ' = ' + var['pre_loop']['value'] + ';\n'
        code = tabify(pre_code, 3) % {'id': pop.id, 'local_index': "[i]", 'semiglobal_index': '', 'global_index': ''} + code

        eqs = ""
        # sum() must generate _sum___all__[i] = _sum_exc[i] + sum_inh[i] + ... at the beginning
        if '__all__' in pop.neuron_type.description['targets']:
            eqs += " "*16 + "// Sum over all targets\n"
            eqs += " "*16 + "_sum___all__[i] = "
            for target in pop.targets:
                eqs += "_sum_" + target + '[i] + '
            eqs = eqs[:-2]
            eqs += ';\n\n'

        # Local variables, evaluated in parallel
        eqs = generate_equation_code(
            pop.id, pop.neuron_type.description, 'local', padding=4)
        eqs = eqs % id_dict
        omp_code = "#pragma omp parallel for simd" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else "#pragma omp simd"
        if eqs.strip() != "":
            code += """
            // Updating the local variables
            %(omp_code)s
            for(int i = 0; i < size; i++){
%(eqs)s
            }
""" % {'eqs': eqs, 'omp_code': omp_code}

        # finish code
        final_code = """
        if( _active ) {
%(code)s
        } // active
""" % {'code': code}

        # if profiling enabled, annotate with profiling code
        if self._prof_gen:
            final_code = self._prof_gen.annotate_update_neuron(pop, final_code)

        return final_code

    def _update_spiking_neuron(self, pop):
        """
        Update code for the spiking neurons comprise of two parts, update of
        ODE and test spiking condition.
        """
        id_dict = {
            'id': pop.id,
            'local_index': "[i]",
            'semiglobal_index': '',
            'global_index': ''
        }

        # Is there a refractory period?
        if pop.neuron_type.refractory or pop.refractory:

            # Identify the refractory variable.
            # By default, it is refractory, but users can specify another one
            refrac_var = "refractory[i]"
            if isinstance(pop.neuron_type.refractory, str):
                found = False
                for param in pop.neuron_type.description["parameters"] + pop.neuron_type.description["variables"]:
                    if param["name"] == pop.neuron_type.refractory:
                        if param['locality'] == 'local':
                            refrac_var = "int(" + pop.neuron_type.refractory + "[i]/dt)"
                        else:
                            refrac_var = "int(" + pop.neuron_type.refractory + "/dt)"
                        found = True
                        break
                if not found:
                    Global._error("refractory = "+ pop.neuron_type.refractory + ": parameter or variable does not exist.")

            # Get the equations
            eqs = generate_equation_code(
                pop.id,
                pop.neuron_type.description,
                'local',
                conductance_only=True,
                padding=4) % id_dict

            # Generate the code snippet for ODEs
            code = """
            // Refractory period
            if( refractory_remaining[i] > 0){
%(eqs)s
                // Decrement the refractory period
                refractory_remaining[i]--;
                continue;
            }
        """ %  {'eqs': eqs}

            refrac_inc = "refractory_remaining[i] = %(refrac_var)s;"%{'refrac_var': refrac_var}

        else:
            code = ""
            refrac_inc = ""

        # Is there an axonal spike condition?
        if pop.neuron_type.axon_spike:
            # get the conditions for axonal and neural spike event
            axon_cond = pop.neuron_type.description['axon_spike']['spike_cond'] % id_dict
            neur_cond = pop.neuron_type.description['spike']['spike_cond'] % id_dict

            # state changes if axonal spike occur
            axon_reset = ""
            for eq in pop.neuron_type.description['axon_spike']['spike_reset']:
                axon_reset += """
                    %(reset)s
            """ % { 'reset': eq['cpp'] % id_dict }

            # Simply extent the spiking vector, as the axonal spike
            # either manipulate neuron state nor consider refractoriness.
            # TODO: what happens for plastic networks? As it seems to be unclear,
            # after discussion with JB it is currently disabled (HD: 21.01.2019)
            axon_spike_code = """
                // Axon Spike Event, only if there was not already an event
                if( (%(axon_condition)s) && !(%(neur_condition)s) ) {
                    axonal.push_back(i);

                    %(axon_reset)s
                }
""" % {
    'axon_condition': axon_cond,
    'neur_condition': neur_cond,
    'axon_reset': axon_reset
}
        else:
            axon_spike_code = ""

        # Global variables
        global_code = ""
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global', padding=3) % id_dict
        if eqs.strip() != "":
            global_code += """
            // Updating the global variables
%(eqs)s
""" % {'eqs': eqs}

        # Gather pre-loop declaration (dt/tau for ODEs)
        pre_code = ""
        for var in pop.neuron_type.description['variables']:
            if 'pre_loop' in var.keys() and len(var['pre_loop']) > 0:
                pre_code += var['ctype'] + ' ' + var['pre_loop']['name'] + ' = ' + var['pre_loop']['value'] + ';\n'
        if len(pre_code) > 0:
            pre_code = """
            // Updating the step sizes
""" + tabify(pre_code, 3)
            global_code = pre_code % id_dict + global_code

        # OMP code
        omp_code = "#pragma omp parallel for simd" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else "#pragma omp simd"
        omp_critical_code = "#pragma omp critical" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""

        # Local variables, evaluated in parallel
        code += generate_equation_code(pop.id, pop.neuron_type.description, 'local', padding=4) % id_dict

        # Process the condition
        cond = pop.neuron_type.description['spike']['spike_cond'] % id_dict

        # Reset equations
        reset = ""
        for eq in pop.neuron_type.description['spike']['spike_reset']:
            reset += """
                    %(reset)s
""" % { 'reset': eq['cpp'] % id_dict }

        # Mean Firing rate
        mean_FR_push, mean_FR_update = self._update_fr(pop)

        # Gather code
        final_spike_gather = """
        if( _active ) {
            for (int i = 0; i < size; i++) {
                // Spike emission
                if(%(condition)s){ // Condition is met
                    // Reset variables
%(reset)s
                    // Store the spike
                    %(omp_critical_code)s
                    {
                    spiked.push_back(i);
                    }
                    last_spike[i] = t;

                    // Refractory period
                    %(refrac_inc)s
                    %(mean_FR_push)s
                }
                %(mean_FR_update)s

%(axon_spike_code)s
            }
        } // active
""" % {
    'condition' : cond,
    'reset': reset,
    'refrac_inc': refrac_inc,
    'mean_FR_push': mean_FR_push,
    'mean_FR_update': mean_FR_update,
    'omp_critical_code': omp_critical_code,
    'axon_spike_code': axon_spike_code
}

        # If axonal events are defined
        if pop.neuron_type.axon_spike:
            global_code = "axonal.clear();\n"+ global_code

        # finish code
        final_eq = """
        if( _active ) {
            spiked.clear();
%(global_code)s
            // Updating local variables
            %(omp_code)s
            for(int i = 0; i < size; i++){
%(code)s
            }
        } // active
""" % {
       'code': code,
       'global_code': global_code,
       'omp_code': omp_code
       }

        # if profiling enabled, annotate with profiling code
        if self._prof_gen:
            final_eq = self._prof_gen.annotate_update_neuron(pop, final_eq)
            final_spike_gather = self._prof_gen.annotate_spike_cond(pop, final_spike_gather)

        return final_eq, final_spike_gather
