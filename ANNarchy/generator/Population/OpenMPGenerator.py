"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from copy import deepcopy

import ANNarchy

from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_omp_extern as global_op_extern_dict
from ANNarchy.generator.Utils import generate_equation_code, tabify, remove_trailing_spaces
from ANNarchy.core import Global
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages

from ANNarchy.generator.Population.PopulationGenerator import PopulationGenerator
from ANNarchy.generator.Population.OpenMPTemplates import openmp_templates

class OpenMPGenerator(PopulationGenerator):
    """
    Generate the header for a Population object to run either on single core
    or multi-cores with OpenMP.
    """
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
        self._templates = deepcopy(openmp_templates)

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
            extern_global_operations += global_op_extern_dict[op['function']] % {'type': ConfigManager().get('precision', self._net_id)}

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
        declare_FR, init_FR, reset_FR = self._init_fr(pop)
        reset_spike += reset_FR

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
            update_variables = self._update_spiking_neuron(pop)
            test_spike_cond = self._spike_gather(pop)

        # Stop condition
        stop_condition = self._stop_condition(pop)

        # Memory management
        size_in_bytes = self._size_in_bytes(pop)
        clear_container = self._clear_container(pop)

        # Profiling
        if self._prof_gen:
            include_profile = """#include "Profiling.hpp"\n"""
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
            # version tag
            'annarchy_version': ANNarchy.__release__,
            # fill code templates
            'float_prec': ConfigManager().get('precision', self._net_id),
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
            'size_in_bytes': size_in_bytes,
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
            'extern': """extern PopStruct%(id)s *pop%(id)s;\n"""% {'id': pop.id},
            'instance': """PopStruct%(id)s *pop%(id)s;\n"""% {'id': pop.id},
            'init': """    pop%(id)s->init_population();\n""" % {'id': pop.id}
        }

        # Generate the calls to be made in the main ANNarchy.cpp
        if len(pop.neuron_type.description['variables']) > 0 or 'update_variables' in pop._specific_template.keys():
            if update_variables != "":
                pop_desc['update'] = """\tpop%(id)s->update(tid);\n""" % {'id': pop.id}

                if pop.neuron_type.type == "spike":
                    pop_desc['update'] += """\tpop%(id)s->spike_gather(tid);\n""" % {'id': pop.id}
            else:
                if "spike_gather_code" in pop._specific_template.keys():
                    if pop._specific_template["spike_gather_code"] != "":
                        pop_desc['update'] = """\tpop%(id)s->spike_gather(tid);\n""" % {'id': pop.id}
            
        if len(pop.neuron_type.description['random_distributions']) > 0:
            pop_desc['rng_update'] = """\tpop%(id)s->update_rng(tid);\n""" % {'id': pop.id}

        if pop.max_delay > 1:
            pop_desc['delay_update'] = tabify("""pop%(id)s->update_delay();\n""" % {'id': pop.id}, 1)

        if len(pop.global_operations) > 0:
            pop_desc['gops_update'] = """\tpop%(id)s->update_global_ops(tid, nt);\n""" % {'id': pop.id}

        return pop_desc

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
                'float_prec': ConfigManager().get('precision', self._net_id)
            }

        # we need to sync the memsets
        if len(code) > 1:
            code += tabify("#pragma omp barrier\n", 1)

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
            #pragma omp single
            {
                _delayed_spike.push_front(spiked);
                _delayed_spike.pop_back();
            }
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

        # if profiling enabled, annotate with profiling code
        if self._prof_gen:
            update_code = self._prof_gen.annotate_update_delay(pop, update_code)

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

        # Special case for early return based on emitted events
        if pop.stop_condition.replace(" ", "") == "spiked:any":
            stop_code = """
    // Stop condition (any)
    bool stop_condition() {
        return !spiked.empty();
    } """
            return stop_code

        elif pop.stop_condition.replace(" ", "") == "spiked:all":
            stop_code = """
    // Stop condition (all)
    bool stop_condition() {
        return spiked.size() == size;
    } """
            return stop_code

        # Process the stop condition
        pop.neuron_type.description['stop_condition'] = {'eq': pop.stop_condition}
        from ANNarchy.parser.Extraction import extract_stop_condition
        extract_stop_condition(pop.neuron_type.description, pop.net_id)

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
        declare_FR = ""; init_FR = ""; reset_FR = ""
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
            if (_spike_history.empty())
                _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        }
    };""" % {'float_prec': ConfigManager().get('precision', self._net_id)}
            init_FR = """
        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >();
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;"""
            reset_FR = """
        // Mean Firing Rate
        for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
            if (!it->empty()) {
                auto empty_queue = std::queue<long int>();
                it->swap(empty_queue);
            }
        }
"""

        return declare_FR, init_FR, reset_FR

    def _update_fr(self, pop):
        "Computes the average firing rate based on history"
        mean_FR_push = ""; mean_FR_update = ""
        if pop.neuron_type.description['type'] == 'spike':
            mean_FR_push = """
                    // Update the mean firing rate
                    if (_mean_fr_window > 0)
                        _spike_history[i].push(t);
            """
            mean_FR_update = """if (_mean_fr_window > 0) {
            #pragma omp for
            for (int i = 0; i < size; i++) {
                while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                    _spike_history[i].pop(); // Suppress spikes outside the window
                }
                r[i] = _mean_fr_rate * %(float_prec)s(_spike_history[i].size());
            }
        } """ % {'float_prec': ConfigManager().get('precision', self._net_id)}

        return mean_FR_push, mean_FR_update

    ##################################################
    # Global operations
    ##################################################
    def _update_globalops(self, pop):
        """
        Update of global functions is a call of pre-implemented functions defined in GlobalOperationTemplate. In case
        of openMP this calls will take place in the population header.

        We consider two cases:

        a) the number of neurons is small, then we compute the operation thread-wise with openMP tasks (TODO: overhead??)
        b) the number of neurons is high enough for a parallel implementation, where we first compute the local result
           and then reduce over all available threads.
        """
        if len(pop.global_operations) == 0:
            return ""

        if True: # parallel reduction is currently disabled
            from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_st_call as call_template

            code = ""
            for op in pop.global_operations:
                code += call_template[op['function']] % {'var': op['variable']}

            return """
        if ( _active ){
            // register tasks
            #pragma omp single nowait
            {
%(code)s
            }

            #pragma omp taskwait
        }""" % {'code': code}

        else:
            from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_omp_call as call_template
            from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_omp_reduce as red_template

            code = ""
            for op in pop.global_operations:
                code += call_template[op['function']] % {'var': op['variable'] }
                code += red_template[op['function']] % {'var': op['variable'] }

            return """
        if ( _active ){
%(code)s
        }""" % {'code': tabify(code,3)}

    def _update_random_distributions(self, pop):
        """
        Generate the C++ for drawing pseudo-random numbers in each step.
        The random variables are drawn sequentially from the same source.
        """
        if len(pop.neuron_type.description['random_distributions']) == 0:
            return ""

        if ConfigManager().get('disable_parallel_rng', self._net_id):
            use_parallel_rng = False
        else:
            use_parallel_rng = True

        rng_code = self._templates['rng']['omp_code_seq'] if not use_parallel_rng else self._templates['rng']['omp_code_par']

        local_code = ""
        global_code = ""
        rng_dist_code = ""
        for rd in pop.neuron_type.description['random_distributions']:

            rng_dist_code += self._templates['rng']['dist_decl'] % {
                'rd_name': rd['name'],
                'rd_init': rd['definition'] % {
                    'id': pop.id,
                    'float_prec': ConfigManager().get('precision', self._net_id),
                    'global_index': ''
                }
            }

            if rd['locality'] == 'local':
                if not use_parallel_rng:
                    local_code += self._templates['rng'][rd['locality']]['update'] % {'id': pop.id, 'rd_name': rd['name'], 'index': 0}
                else:
                    local_code += self._templates['rng'][rd['locality']]['update'] % {'id': pop.id, 'rd_name': rd['name'], 'index': "tid"}
            else:
                global_code += self._templates['rng'][rd['locality']]['update'] % {'id': pop.id, 'rd_name': rd['name']}

        if use_parallel_rng:
            if len(global_code.strip()) > 0:
                global_code = """\t\t\t// global attributes
            #pragma omp single nowait
            {
            %(update_rng_global)s
            }""" % {'update_rng_global': tabify(global_code,1)}

            if len(local_code.strip()) > 0:
                local_code = """\t\t\t// local attributes
            #pragma omp for
            for (int i = 0; i < size; i++) {
                %(update_rng_local)s
            }""" % {'update_rng_local': local_code}

        else:
            if len(local_code.strip()) > 0:
                local_code = """\t\t\t\t// local attributes
                for (int i = 0; i < size; i++) {
                %(update_rng_local)s
                }""" % {'update_rng_local': tabify(local_code, 1)}

        # Final code consists of local and global variables
        final_code = rng_code % {
            'rng_dist': tabify(rng_dist_code, 3),
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
        if "update_variables" in pop._specific_template.keys():
            final_eq = pop._specific_template["update_variables"]
            # if profiling enabled, annotate with profiling code
            if self._prof_gen:
                final_eq = self._prof_gen.annotate_update_neuron(pop, final_eq)
            return final_eq

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
        eqs = generate_equation_code(pop.neuron_type.description, locality='global', padding=3)
        eqs = eqs % id_dict
        if eqs.strip() != "":
            code += """
            // Updating the global variables
            #pragma omp single
            {
%(eqs)s
}
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
        eqs = generate_equation_code(pop.neuron_type.description, 'local', padding=4)
        eqs = eqs % id_dict
        if eqs.strip() != "":
            code += """
            // Updating the local variables
            %(omp_code)s
            for (int i = 0; i < size; i++) {
%(eqs)s
            }
""" % {
    'omp_code': "#pragma omp for simd" if not ConfigManager().get('disable_SIMD_Eq', self._net_id) else "#pragma omp for",
    'eqs': eqs
}

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
        Update code for the spiking neurons.
        """
        if "update_variables" in pop._specific_template.keys():
            final_eq = pop._specific_template["update_variables"]
            # if profiling enabled, annotate with profiling code
            if self._prof_gen:
                final_eq = self._prof_gen.annotate_update_neuron(pop, final_eq)
            return final_eq

        id_dict = {
            'id': pop.id,
            'local_index': "[i]",
            'semiglobal_index': '',
            'global_index': ''
        }

        # Global variables
        global_code = ""
        eqs = generate_equation_code(pop.neuron_type.description, locality='global', mask_variable=None, padding=3) % id_dict
        if eqs.strip() != "":
            global_code += """
            // Updating the global variables
            #pragma omp single
            {
%(eqs)s
}
""" % {'eqs': eqs}

        # Is there a refractory period?
        has_refractory = "in_ref" if (pop.neuron_type.refractory or pop.refractory) else None

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

        # Local variables, evaluated in parallel
        local_code = generate_equation_code(pop.neuron_type.description, locality='local', mask_variable=has_refractory, padding=4) % id_dict
        
        # Decrement of refractoriness
        if has_refractory:
            local_code += tabify("""
// Decrement the refractory period
refractory_remaining[i] -= (1 - in_ref[i]);
""", 4)

        # Increment of the refractory variable
        if has_refractory:
            omp_code = "#pragma omp for" if pop.size > Global.OMP_MIN_NB_NEURONS else ""
            comp_inref = """
            // compute which neurons are in refractory
            #pragma omp for
            for (int i = 0; i < size; i++) {
                in_ref[i] = (refractory_remaining[i] > 0) ? 0 : 1;
            }
            """ % {'omp_code': omp_code }

        else:
            comp_inref = ""

        # If axonal events are defined
        if pop.neuron_type.axon_spike:
            global_code = "axonal.clear();\n"+ global_code

        # finish code
        final_eq = """
        if( _active ) {
%(comp_inref)s

%(global_code)s

            // Updating local variables
            %(omp_code)s
            for (int i = 0; i < size; i++) {
%(local_code)s
            }

        } // active
""" % {
    'omp_code': "#pragma omp for simd" if not ConfigManager().get('disable_SIMD_Eq', self._net_id) else "#pragma omp for",
    'comp_inref': comp_inref,
    'local_code': local_code,
    'global_code': global_code,
    }

        # if profiling enabled, annotate with profiling code
        if self._prof_gen:
            final_eq = self._prof_gen.annotate_update_neuron(pop, final_eq)

        return final_eq

    def _spike_gather(self, pop):
        """
        Generate the code for emitting spike events. Dependent
        on the number of neurons, this is either done in parallel
        or by one thread of the worker group.
        """
        if "spike_gather_code" in pop._specific_template.keys():
            final_spike_gather = pop._specific_template["spike_gather_code"]
            # if profiling enabled, annotate with profiling code
            if self._prof_gen:
                final_spike_gather = self._prof_gen.annotate_spike_cond(pop, final_spike_gather)

            return final_spike_gather

        id_dict = {
            'id': pop.id,
            'local_index': "[i]",
            'semiglobal_index': '',
            'global_index': ''
        }

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

        # Is there a refractory period?
        has_refractory = True if (pop.neuron_type.refractory or pop.refractory) else False

        # Mean Firing rate
        mean_FR_push, mean_FR_update = self._update_fr(pop)

        # Increment of the refractory variable
        if has_refractory:
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
                    Messages._error("refractory = "+ pop.neuron_type.refractory + ": parameter or variable does not exist.")

            refrac_inc = "refractory_remaining[i] = %(refrac_var)s;"%{'refrac_var': refrac_var}
            omp_code = "#pragma omp for" if pop.size > Global.OMP_MIN_NB_NEURONS else ""

        else:
            refrac_inc = ""

        # Process the condition
        cond = pop.neuron_type.description['spike']['spike_cond'] % id_dict

        # Reset equations
        reset = ""
        for eq in pop.neuron_type.description['spike']['spike_reset']:
            reset += """
                    %(reset)s
""" % { 'reset': eq['cpp'] % id_dict }

        # Gather code
        omp_critical_code = "#pragma omp critical" if pop.size > Global.OMP_MIN_NB_NEURONS else ""
        refrac_check = tabify("""
    // check if neuron is in refractory
    if (in_ref[i]==0)
        continue;
""", 3)

        # If a population is too small, the overhead of local arrays and
        # fusion afterwards would be to high.
        if pop.size > Global.OMP_MIN_NB_NEURONS:
            mean_FR_update = tabify(mean_FR_update,1)

            gather_code = """
        if ( _active ) {
            auto local_spikes = std::vector<int>();

            #pragma omp for nowait
            for (int i = 0; i < size; i++) {
%(refrac_check)s

                // Spike emission
                if( %(condition)s ) { // Condition is met
                    // Reset variables
%(reset)s
                    // Store the spike
                    local_spikes.push_back(i);
                    last_spike[i] = t;

                    // Refractory period
                    %(refrac_inc)s
                    %(mean_FR_push)s
                }

%(axon_spike_code)s
            }

            // Update mean firing rate
        %(mean_FR_update)s

            local_spiked_sizes[tid+1] = local_spikes.size();
            #pragma omp barrier

            #pragma omp single
            {
            #ifdef _DEBUG_SPIKE_GATHER
                std::cout << "time step - " << t << ": ";
                for (auto it = local_spiked_sizes.begin(); it != local_spiked_sizes.end(); it++) {
                    std::cout << *it << ", ";
                }
                std::cout << " --> ";
            #endif

                // compute storage offsets
                for (int i = 1; i < (global_num_threads+1); i++) {
                    local_spiked_sizes[i] += local_spiked_sizes[i-1];
                }

            #ifdef _DEBUG_SPIKE_GATHER
                for (auto it = local_spiked_sizes.begin(); it != local_spiked_sizes.end(); it++) {
                    std::cout << *it << ", ";
                }
            #endif

                // set the result container to the correct size
                spiked.resize(local_spiked_sizes[global_num_threads]);

            #ifdef _DEBUG_SPIKE_GATHER
                std::cout << "(" << std::to_string(static_cast<long long>(spiked.size())) << " events)" << std::endl;
            #endif
            } // implicit barrier

            // each thread computes it local data to the shared container
            if (!local_spikes.empty())
                std::copy(local_spikes.begin(), local_spikes.end(), spiked.begin() + local_spiked_sizes[tid]);
        } // active
"""
        else:
            gather_code = """
        if ( _active ) {
        #pragma omp single
        {
            spiked.clear();
            for (int i = 0; i < size; i++) {
%(refrac_check)s

                // Spike emission
                if( %(condition)s ) { // Condition is met
                    // Reset variables
%(reset)s
                    // Store the spike
                    spiked.push_back(i);
                    last_spike[i] = t;

                    // Refractory period
                    %(refrac_inc)s
                    %(mean_FR_push)s
                }

%(axon_spike_code)s
            }
        } // omp single

        // Update mean firing rate
        %(mean_FR_update)s

        } // active
"""

        final_spike_gather = gather_code % {
    'condition' : cond,
    'refrac_check': refrac_check if has_refractory else "",
    'reset': reset,
    'refrac_inc': refrac_inc,
    'mean_FR_push': mean_FR_push,
    'mean_FR_update': mean_FR_update,
    'omp_critical_code': omp_critical_code,
    'axon_spike_code': axon_spike_code
}

        # if profiling enabled, annotate with profiling code
        if self._prof_gen:
            final_spike_gather = self._prof_gen.annotate_spike_cond(pop, final_spike_gather)

        return final_spike_gather
