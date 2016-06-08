"""

    PopulationGenerator.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
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

import ANNarchy.generator.Template.PopulationTemplate as PopTemplate
from .Template.GlobalOperationTemplate import global_operation_templates_extern as global_op_extern_dict

class PopulationGenerator(object):

    def __init__(self, profile_generator, net_id):
        """
        Initialize PopulationGenerator.
        """
        self._prof_gen = profile_generator
        self._net_id = net_id

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, pop, annarchy_dir):
        """
        Generate the c-style struct definition for a population object.

        Parameters:

            pop: Population object
            annarchy_dir: working directory

        Returns:

            dictionary: include directive, pointer definition, call statements and other
                        informations needed to create the ANNarchy.cpp/.cu code

                        Please note, the returned dictionary will vary dependent on
                        population type and used parallelization paradigm

        Templates:

            header_struct, parameter_decl, parameter_acc, variable_decl, variable_acc
        """
        # Basic informations common to all populations
        pop_desc = {
            'include': """#include "pop%(id)s.hpp"\n""" % { 'id': pop.id },
            'extern': """extern PopStruct%(id)s pop%(id)s;\n"""% { 'id': pop.id },
            'instance': """PopStruct%(id)s pop%(id)s;\n"""% { 'id': pop.id },
            'init': """    pop%(id)s.init_population();\n""" % {'id': pop.id}
        }

        ## We first make a normal generation
        # Retrieve the correct template
        if Global.config['paradigm'] == "openmp":
            base_template = PopTemplate.header_struct_omp

        elif Global.config['paradigm'] == "cuda":
            base_template = PopTemplate.header_struct_cuda

        # Generate declaration and accessors of all parameters and variables
        declaration_parameters_variables, access_parameters_variables = self.generate_decl_and_acc(pop)

        # Additional includes and structures
        include_additional = ""; struct_additional = ""
        declare_additional = ""; init_additional = ""; reset_additional = ""

        # Declare global operations as extern at the beginning of the file
        extern_global_operations = ""
        for op in pop.global_operations:
            extern_global_operations += global_op_extern_dict[op['function']]

        # Initialize parameters and variables
        init_parameters_variables = self.init_population(pop)

        # Spike-specific stuff
        reset_spike = ""; declare_spike=""; init_spike = ""
        if pop.neuron_type.description['type'] == 'spike':
            # Main data for spiking pops
            declare_spike += PopTemplate.spike_specific['declare_spike'] % {'id': pop.id}
            init_spike += PopTemplate.spike_specific['init_spike'] % {'id': pop.id}
            reset_spike += PopTemplate.spike_specific['reset_spike'] % {'id': pop.id}
            # If there is a refractory period
            if pop.neuron_type.refractory or pop.refractory:
                declare_spike += PopTemplate.spike_specific['declare_refractory'] % {'id': pop.id}
                init_spike += PopTemplate.spike_specific['init_refractory'] % {'id': pop.id}
                reset_spike += PopTemplate.spike_specific['reset_refractory'] % {'id': pop.id}

        # Process eventual delay
        declare_delay = ""; init_delay = ""; update_delay=""; reset_delay = ""
        if pop.max_delay > 1:
            declare_delay, init_delay, update_delay, reset_delay = self.delay_code(pop)

        # Process mean FR computations
        declare_FR, init_FR = self.init_fr(pop)

        # Update random distributions
        update_rng = self.update_random_distributions(pop)

        # Update global operations
        update_global_ops = self.update_globalops(pop)

        # Update the neural variables
        if Global.config['paradigm']== "openmp":
            if pop.neuron_type.type == 'rate':
                update_variables = self.update_rate_neuron_openmp(pop)
            else:
                update_variables = self.update_spike_neuron(pop)
        else:
            if pop.neuron_type.type == 'rate':
                body, header, update_call = self.update_rate_neuron_cuda(pop)
                update_variables = ""
            else:
                Global._error("Spiking neurons on GPUs are currently not supported")

        # Stop condition
        stop_condition = self.stop_condition(pop)

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
        if 'reset_additional' in pop._specific_template.keys() and pop.max_delay > 1:
            reset_additional = pop._specific_template['reset_additional']
        if 'update_variables' in pop._specific_template.keys():
            update_variables = pop._specific_template['update_variables']
        if 'update_rng' in pop._specific_template.keys():
            update_rng = pop._specific_template['update_rng']
        if 'update_delay' in pop._specific_template.keys() and pop.max_delay > 1:
            update_delay = pop._specific_template['update_delay']
        if 'update_global_ops' in pop._specific_template.keys():
            update_global_ops = pop._specific_template['update_global_ops']

        # Fill the template
        code = base_template % { 'id': pop.id,
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
                                 'update_rng': update_rng,
                                 'update_delay': update_delay,
                                 'update_global_ops': update_global_ops,
                                 'stop_condition': stop_condition
                                }

        # Store the complete header definition in a single file
        with open(annarchy_dir+'/generate/net'+str(self._net_id)+'/pop'+str(pop.id)+'.hpp', 'w') as ofile:
            ofile.write(code)

        # Generate the calls to be made in the main ANNarchy.cpp
        if Global.config['paradigm'] == "openmp":
            if len(pop.neuron_type.description['variables']) > 0 or 'update_variables' in pop._specific_template.keys():
                if update_variables != "":
                    pop_desc['update'] = """    pop%(id)s.update();\n""" % { 'id': pop.id }

            if len(pop.neuron_type.description['random_distributions']) > 0:
                pop_desc['rng_update'] = """    pop%(id)s.update_rng();\n""" % { 'id': pop.id }

            if pop.max_delay > 1:
                pop_desc['delay_update'] = """    pop%(id)s.update_delay();\n""" % { 'id': pop.id }

            if len(pop.global_operations) > 0:
                pop_desc['gops_update'] = """    pop%(id)s.update_global_ops();\n""" % { 'id': pop.id }
        else:
            pop_desc['update'] = update_call
            pop_desc['update_body'] =  body
            pop_desc['update_header'] =  header
            pop_desc['update_delay'] = """    pop%(id)s.update_delay();\n""" % {'id': pop.id} if pop.max_delay > 1 else ""

            if len(pop.global_operations) > 0:
                update_global_ops = self.update_globalops(pop)
                pop_desc['gops_update'] = update_global_ops % { 'id': pop.id }

            host_to_device, device_to_host = self._cuda_memory_transfers(pop)
            pop_desc['host_to_device'] = host_to_device
            pop_desc['device_to_host'] = device_to_host

        return pop_desc

    def _generate_cuda(self, pop, pop_desc, annarchy_dir):
        "Generate complete code corresponding to CUDA"
        glops_extern = ""
        ## We first make a normal generation
        # Retrieve the correct template
        base_template = PopTemplate.header_struct_cuda

        # Generate declaration and accessors of all parameters and variables
        declaration_parameters_variables, access_parameters_variables = self.generate_decl_and_acc(pop)

        init = self.init_population(pop)
        reset = ""
        if pop.max_delay > 1:
            delay_init, delay_update, delay_reset = self._delay_code(pop)
            #delay_update = delay_update.replace("pop"+str(pop.id)+".", "") #TODO: adjust prefixes in parser
            init += delay_init
            reset += delay_reset

        update_rng = self.update_random_distributions(pop)#.replace("pop"+str(pop.id)+".", "") #TODO: adjust prefixes in parser

        if pop.neuron_type.type == 'rate':
            body, header, update_call = self.update_rate_neuron_cuda(pop)
        else:
            Global._error("Spiking neurons on GPUs are currently not supported")


        code = base_template % { 'id': pop.id,
                                 'name': pop.name,
                                 'size': pop.size,
                                 'include_additional': include_additional,
                                 'struct_additional': struct_additional,
                                 'extern_global_operations': extern_global_operations,
                                 'declare_spike_arrays': declare_spike,
                                 'declare_parameters_variables': declaration_parameters_variables,
                                 'declare_additional': declare_additional,
                                 'declare_delay': declare_delay,
                                 'access_parameters_variables': access_parameters_variables,
                                 'init_parameters_variables': init_parameters_variables,
                                 'init_spike': init_spike,
                                 'init_delay': init_delay,
                                 'init_additional': init_additional,
                                 'reset_spike': reset_spike,
                                 'reset_delay': reset_delay,
                                 'reset_additional': reset_additional,
                                 'update_variables': update_variables,
                                 'update_rng': update_rng,
                                 'update_delay': update_delay,
                                 'update_global_ops': update_global_ops,
                                 'stop_condition': stop_condition
                                }

        # Store the complete header definition in a single file
        with open(annarchy_dir+'/generate/net'+str(self._net_id)+'/pop'+str(pop.id)+'.hpp', 'w') as ofile:
            ofile.write(code)

        pop_desc['update'] = update_call
        pop_desc['update_body'] =  body
        pop_desc['update_header'] =  header
        pop_desc['update_delay'] = """    pop%(id)s.update_delay();\n""" % {'id': pop.id} if pop.max_delay > 1 else ""

        if len(pop.global_operations) > 0:
            update_global_ops = self.update_globalops(pop)
            pop_desc['gops_update'] = update_global_ops % { 'id': pop.id }

        host_to_device, device_to_host = self._cuda_memory_transfers(pop)
        pop_desc['host_to_device'] = host_to_device
        pop_desc['device_to_host'] = device_to_host

        return pop_desc

    def generate_decl_and_acc(self, pop):
        # Pick basic template based on neuron type
        attr_template = PopTemplate.attribute_decl[Global.config['paradigm']]
        acc_template = PopTemplate.attribute_acc[Global.config['paradigm']]

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

        # Arrays for the presynaptic sums
        declaration += """
    // Targets
"""
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                declaration += PopTemplate.rate_psp[Global.config['paradigm']]['decl'] % {'target': target}

        # Global operations
        declaration += """
    // Global operations
"""
        for op in pop.global_operations:
            if Global.config['paradigm']=="openmp":
                declaration += """    double _%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}
            else:
                declaration += """    double _%(op)s_%(var)s;
    double *_gpu_%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}
        # Arrays for the random numbers
        declaration += """
    // Random numbers
"""
        for rd in pop.neuron_type.description['random_distributions']:
            if Global.config['paradigm']=="openmp":
                declaration += PopTemplate.cpp_11_rng[rd['locality']]['decl'] % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}
            else:
                declaration += PopTemplate.cuda_rng[rd['locality']]['decl'] % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}


        # Local functions
        if len(pop.neuron_type.description['functions'])>0:
            declaration += """
    // Local functions
"""
            for func in pop.neuron_type.description['functions']:
                declaration += ' '*4 + func['cpp'] + '\n'

        return declaration, accessors

#######################################################################
############## BODY: initialization codes #############################
#######################################################################
    def init_population(self, pop):
        # active is true by default
        code = ""

        attr_tpl = PopTemplate.attribute_cpp_init[Global.config['paradigm']]

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
                if Global.config['paradigm'] == "openmp":
                    code += PopTemplate.cpp_11_rng[rd['locality']]['init'] % {'id': pop.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': pop.id}}
                else:
                    code += PopTemplate.cuda_rng[rd['locality']]['init'] % {'id': pop.id, 'rd_name': rd['name'] }

        # Global operations
        code += self.init_globalops(pop)

        # Targets
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                code += PopTemplate.rate_psp[Global.config['paradigm']]['init'] % {'id': pop.id, 'target': target}

        return code

    def init_globalops(self, pop):
        if len(pop.global_operations)==0:
            return ""

        code = "// Initialize global operations\n"
        for op in pop.global_operations:
            if Global.config['paradigm'] == "openmp":
                code += """    _%(op)s_%(var)s = 0.0;
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}
            else:
                code += """    _%(op)s_%(var)s = 0.0;
    cudaMalloc((void**)&_gpu_%(op)s_%(var)s, sizeof(double));
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}

        return code

    def delay_code(self, pop):
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
        delay_tpl = PopTemplate.attribute_delayed[Global.config['paradigm']]

        # Declaration
        declare_code = """
    // Delayed variables"""

        if Global.config['paradigm'] == "openmp":
            if pop.neuron_type.type == "rate":
                for var in pop.delayed_variables:
                    if var in pop.neuron_type.description['local']:
                        declare_code += """
    std::deque< std::vector<double> > _delayed_%(var)s; """ % {'var': var}
                    else:
                        declare_code += """
    std::deque< double > _delayed_%(var)s; """ % {'var': var}
            else: # Spiking networks should only exchange spikes
                declare_code += """
    // Delays for spike population
    std::deque< std::vector<int> > _delayed_spike;
"""
                for var in pop.delayed_variables:
                    if var in pop.neuron_type.description['local']:
                        declare_code += """
    std::deque< std::vector<double> > _delayed_%(var)s; """ % {'var': var}
                    else:
                        declare_code += """
    std::deque< double > _delayed_%(var)s; """ % {'var': var}
        else: #CUDA
            if pop.neuron_type.type == "rate":
                for var in pop.delayed_variables:
                    if var in pop.neuron_type.description['local']:
                        declare_code += """
std::deque< double* > gpu_delayed_%(var)s; // list of gpu arrays""" % {'var': var}
                    else:
                        #TODO:
                        continue
            else:
                Global._error("Synaptic delays for spiking neurons are not implemented yet with CUDA...")


        # Initialization
        init_code = """
        // Delayed variables"""

        for var in pop.delayed_variables:
            locality = "local" if var in pop.neuron_type.description['local'] else "global"
            init_code += delay_tpl[locality] % {'delay': pop.max_delay, 'var': var}

        # Update
        update_code = ""
        reset_code = ""
        if Global.config['paradigm'] == "openmp":
            for var in pop.delayed_variables:
                update_code += """
        _delayed_%(var)s.push_front(%(var)s);
        _delayed_%(var)s.pop_back();
""" % {'id': pop.id, 'var' : var}

                # reset
                reset_code += PopTemplate.attribute_delayed['openmp']['reset'] % {'id': pop.id, 'var' : var}

        else:
            """
            Implementation Note: (HD: 15.06.2015)

                Currently I see no better way to implement delays, as consequence of missing device-device memory transfers ...

                This implementation is from a performance point of view problematic, cause of low host-device memory bandwith,
                maybe enhancable through pinned memory (CC 2.x), or asynchronous device transfers (CC 3.x)

            Algorithm:

                * get reference of last list in queue (last_%(var)s)
                * cycle the last_%(var)s pointer from back to front
                * get current value from %(var)s and store it in tmp_%(var)s
                * store the data in queue
            """
            for var in pop.delayed_variables:
                update_code += """
        double* last_%(var)s = gpu_delayed_%(var)s.back();
        gpu_delayed_%(var)s.pop_back();
        gpu_delayed_%(var)s.push_front(last_%(var)s);
        std::vector<double> tmp_%(var)s = std::vector<double>( size, 0.0);
        cudaMemcpy( last_%(var)s, gpu_%(var)s, sizeof(double) * size, cudaMemcpyDeviceToDevice );
    #ifdef _DEBUG
        cudaError_t err_%(var)s = cudaGetLastError();
        if (err_%(var)s != cudaSuccess)
            std::cout << "pop%(id)s - delay %(var)s: " << cudaGetErrorString(err_%(var)s) << std::endl;
    #endif
""" % {'id': pop.id, 'name' : pop.name, 'var': var }

                # reset
                reset_code += PopTemplate.attribute_delayed['cuda']['reset'] % {'id': pop.id, 'var' : var}

        # Delaying spike events is done differently
        if pop.neuron_type.type == 'spike':
            if Global.config['paradigm']=="openmp":
                init_code += """
        _delayed_spike = std::deque< std::vector<int> >(%(delay)s, std::vector<int>());""" % {'delay': pop.max_delay}

                update_code += """
            _delayed_spike.push_front(spiked);
            _delayed_spike.pop_back();
"""
                reset_code += """
        _delayed_spike.clear();
        _delayed_spike = std::deque< std::vector<int> >(%(delay)s, std::vector<int>());""" % {'delay': pop.max_delay}
            else:
                Global._error("No synaptic delays for spiking synapses on CUDA implemented ...")


        update_code = """
        if ( _active ) {
%(code)s
        }""" % {'code': update_code }

        return declare_code, init_code, update_code, reset_code

#######################################################################
############## BODY: update variables codes ###########################
#######################################################################
    def update_rate_neuron_openmp(self, pop):
        """
        Generate the code template for neural update step, more precise updating of variables.
        The code comprise of two major parts: global and local update, second one parallelized
        with an openmp for construct, if number of threads is greater than one and the number
        of neurons exceed a minimum amount of neurons ( defined as Global.OMP_MIN_NB_NEURONS)
        """
        from .Utils import generate_equation_code
        code = ""

        # Global variables
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global', padding=3) % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}
        if eqs.strip() != "":
            code += """
            // Updating the global variables
%(eqs)s
""" % {'eqs': eqs}

        # Local variables, evaluated in parallel
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local', padding=4) % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}
        if eqs.strip() != "":
            omp_code = "#pragma omp parallel for" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""
            code += """
            // Updating the local variables
            %(omp_code)s
            for(int i = 0; i < %(size)s; i++){
%(eqs)s
            }
""" % {'id': pop.id, 'size': pop.size, 'name' : pop.name, 'eqs': eqs, 'omp_code': omp_code}

        if code != "":
            # finish code
            final_code = """
        if( _active ) {
%(code)s
        } // active
""" % {'code': code}

            # if profiling enabled, annotate with profiling code
            if self._prof_gen:
                final_code = self._prof_gen.annotate_update_neuron_omp(pop, final_code)
        else:
            final_code = ""

        return final_code

    def update_rate_neuron_cuda(self, pop):
        """
        Generate the code template for neural update step, more precise updating of variables.
        The code comprise of several parts: creating of local and global update code, generating
        function prototype and finally calling statement.

        Returns:

            a tuple of three strings, comprising of:

                * body:    kernel implementation
                * header:  kernel prototypes
                * call:    kernel call
        """
        # Is there any variable?
        if len(pop.neuron_type.description['variables']) == 0:
            return "", "", ""

        # Neural update
        from .Utils import generate_equation_code

        header = ""
        body = ""
        call = ""

        # determine variables and attributes
        var = ""
        par = ""
        tar = ""
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] in pop.neuron_type.description['local']:
                var += """, %(type)s* %(name)s""" % { 'type': attr['ctype'], 'name': attr['name'] }
            else:
                par += """, %(type)s %(name)s""" % { 'type': attr['ctype'], 'name': attr['name'] }

        # random variables
        for rd in pop.neuron_type.description['random_distributions']:
            var += """, curandState* %(rd_name)s""" % { 'rd_name' : rd['name'] }

        # global operations
        for op in pop.global_operations:
            par += """, double _%(op)s_%(var)s """ % {'op': op['function'], 'var': op['variable']}

        # targets
        for target in sorted(pop.neuron_type.description['targets']):
            tar += """, double* _sum_%(target)s""" % {'target' : target}

        #Global variables
        glob_eqs = ""
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global') % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}
        if eqs.strip() != "":
            glob_eqs = """
    if ( threadIdx.x == 0)
    {
%(eqs)s
    }
""" % {'id': pop.id, 'eqs': eqs }
            #glob_eqs = glob_eqs.replace("pop"+str(pop.id)+".", "")

        # Local variables
        loc_eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local') % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}

        # we replace the rand_%(id)s by the corresponding curand... term
        for rd in pop.neuron_type.description['random_distributions']:
            if rd['dist'] == "Uniform":
                term = """curand_uniform_double( &%(rd)s[i] ) * (%(max)s - %(min)s) + %(min)s""" % { 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
                term = """curand_uniform_double( &%(rd)s[0] ) * (%(max)s - %(min)s) + %(min)s""" % { 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1] };
                glob_eqs = glob_eqs.replace(rd['name'], term)
            elif rd['dist'] == "Normal":
                term = """curand_normal_double( &%(rd)s[i] )""" % { 'rd': rd['name'] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
                term = """curand_normal_double( &%(rd)s[0] )""" % { 'rd': rd['name'] };
                glob_eqs = glob_eqs.replace(rd['name'], term)
            elif rd['dist'] == "LogNormal":
                term = """curand_log_normal_double( &%(rd)s[i], %(mean)s, %(std_dev)s)""" % { 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
                term = """curand_log_normal_double( &%(rd)s[0], %(mean)s, %(std_dev)s)""" % { 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1] };
                glob_eqs = glob_eqs.replace(rd['name'], term)
            else:
                Global._error("Unsupported random distribution on GPUs: " + rd['dist'])

        # remove all types
        repl_types = ["double*", "float*", "int*", "curandState*", "double", "float", "int"]
        tar_wo_types = tar
        var_wo_types = var
        par_wo_types = par
        for type in repl_types:
            tar_wo_types = tar_wo_types.replace(type, "")
            var_wo_types = var_wo_types.replace(type, "")
            par_wo_types = par_wo_types.replace(type, "")

        #
        # create kernel prototypes
        body += PopTemplate.cuda_pop_kernel % {
                                'id': pop.id,
                                'local_eqs': loc_eqs,
                                'global_eqs': glob_eqs,
                                'pop_size': str(pop.size),
                                'tar': tar,
                                'tar2': tar_wo_types,
                                'var': var,
                                'var2': var_wo_types,
                                'par': par,
                                'par2': par_wo_types
                             }

        #
        # create kernel prototypes
        header += """
__global__ void cuPop%(id)s_step( double dt%(tar)s%(var)s%(par)s );
""" % { 'id': pop.id, 'tar': tar, 'var': var, 'par': par }

        #
        #    for calling entites we need to determine again all members
        var = ""
        par = ""
        tar = ""
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] in pop.neuron_type.description['local']:
                var += """, pop%(id)s.gpu_%(name)s""" % { 'id': pop.id, 'name': attr['name'] }
            else:
                par += """, pop%(id)s.%(name)s""" % { 'id': pop.id, 'name': attr['name'] }

        # random variables
        for rd in pop.neuron_type.description['random_distributions']:
            var += """, pop%(id)s.gpu_%(rd_name)s""" % { 'id': pop.id, 'rd_name' : rd['name'] }

        # targets
        for target in sorted(pop.neuron_type.description['targets']):
            tar += """, pop%(id)s.gpu_sum_%(target)s""" % { 'id': pop.id, 'target' : target}

        # global operations
        for op in pop.global_operations:
            par += """, pop%(id)s._%(op)s_%(var)s""" % { 'id': pop.id, 'op': op['function'], 'var': op['variable'] }

        call += PopTemplate.cuda_pop_kernel_call % {
            'id': pop.id,
            'tar': tar.replace("double*","").replace("int*",""),
            'var': var.replace("double*","").replace("int*",""),
            'par': par.replace("double","").replace("int","")
        }

        return body, header, call

    ################################
    ### Spiking neurons
    ################################

    def update_spike_neuron(self, pop):
        # Neural update
        from .Utils import generate_equation_code

        # Is there a refractory period?
        if pop.neuron_type.refractory or pop.refractory:
            eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local', conductance_only=True, padding=4) % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}
            code = """
            if( refractory_remaining[i] > 0){ // Refractory period
%(eqs)s
                // Decrement the refractory period
                refractory_remaining[i]--;
                continue;
            }
        """ %  {'id': pop.id, 'eqs': eqs}
            refrac_inc = "refractory_remaining[i] = refractory[i];" %  {'id': pop.id}
        else:
            code = ""
            refrac_inc = ""

        # Global variables
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global', padding=2) % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}
        if eqs.strip() != "":
            global_code = eqs
        else:
            global_code = ""

        # OMP code
        omp_code = "#pragma omp parallel for" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""
        omp_critical_code = "#pragma omp critical" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""

        # Local variables, evaluated in parallel
        code += generate_equation_code(pop.id, pop.neuron_type.description, 'local', padding=3) % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}

        # Process the condition
        cond =  pop.neuron_type.description['spike']['spike_cond'] % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}

        # reset equations
        reset = ""
        for eq in pop.neuron_type.description['spike']['spike_reset']:
            reset += """
                %(reset)s
""" % {'reset': eq['cpp'] % {'id': pop.id, 'local_index': "[i]", 'global_index': ''}}

        # Mean Firing rate
        mean_FR_push, mean_FR_update = self.update_fr(pop)

        # Gather code
        spike_gather = """
            if(%(condition)s){ // Emit a spike
%(reset)s
                %(omp_critical_code)s
                {
                    spiked.push_back(i);
                }
                last_spike[i] = t;
                %(refrac_inc)s
                %(mean_FR_push)s
            }
            %(mean_FR_update)s
"""% {  'id': pop.id, 'name': pop.name, 'size': pop.size,
        'condition' : cond, 'reset': reset,
        'refrac_inc': refrac_inc,
        'mean_FR_push': mean_FR_push,
        'mean_FR_update': mean_FR_update,
        'omp_critical_code': omp_critical_code}

        code += spike_gather

        # finish code
        final_code = """
    if( _active ) {
        spiked.clear();
%(global_code)s
        %(omp_code)s
        for(int i = 0; i < %(size)s; i++){
%(code)s
        }
    } // active
""" % {'id': pop.id, 'size': pop.size, 'name': pop.name, 'code': code, 'global_code': global_code, 'omp_code': omp_code }

        # if profiling enabled, annotate with profiling code
        if self._prof_gen:
            final_code = self._prof_gen.annotate_update_neuron_omp(pop, final_code)

        return final_code

    ################################
    ### Reset
    ################################

    def reset_computesum(self, pop):
        code = ""
        for target in sorted(pop.targets):
            code += """
    if (pop%(id)s._active)
        memset( pop%(id)s._sum_%(target)s.data(), 0.0, pop%(id)s._sum_%(target)s.size() * sizeof(double));
""" % {'id': pop.id, 'target': target}
        return code

    ################################
    ### Global operations
    ################################

    def update_globalops(self, pop):
        """
        Update of global functions is a call of pre-implemented
        functions defined in GlobalOperationTemplate. In case of
        OpenMP this calls will take place in the population header.
        In case of CUDA the call semantic will be placed in ANNarchy.cu
        file as part of the host section.
        """
        if len(pop.global_operations) == 0:
            return ""

        code = ""
        if Global.config['paradigm'] == "openmp":
            for op in pop.global_operations:
                code += """
            _%(op)s_%(var)s = %(op)s_value(%(var)s.data(), size);
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}
        else:
            from Template.GlobalOperationTemplate import global_operation_templates_cuda as template
            for op in pop.global_operations:
                code += template[op['function']]['call'] % { 'id': pop.id, 'op': op['function'], 'var': op['variable'] }

        return """
    if (_active){
%(code)s
}""" % {'code':code}

    ################################
    ### Mean firing rate (spiking)
    ################################
    def init_fr(self, pop):
        "Declares arrays for computing the mean FR of a spiking neuron"
        declare_FR=""; init_FR = ""
        if pop.neuron_type.description['type'] == 'spike' and pop._compute_mean_fr != -1:
            declare_FR = """
    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;"""
            init_FR = """
        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());"""
        return declare_FR, init_FR

    def update_fr(self, pop):
        "Computes the average firing rate based on history"

        mean_FR_push = ""; mean_FR_update = ""
        if pop.neuron_type.description['type'] == 'spike' and pop._compute_mean_fr != -1:
            window = pop._compute_mean_fr
            window_int = long(window/Global.config['dt'])
            mean_FR_push = """
                // Update the mean firing rate
                _spike_history[i].push(t);
            """
            mean_FR_update = """
            while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - %(window)s)){
                _spike_history[i].pop(); // Suppress spikes outside the window
            }
            r[i] = %(freq)s * double(_spike_history[i].size());
            """ % {'window': str(window_int), 'freq': str(1000.0/window)}

        return mean_FR_push, mean_FR_update

    ################################
    ### Stop condition
    ################################

    def stop_condition(self, pop):
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

    ################################
    ### Random distributions
    ################################

    def update_random_distributions(self, pop):
        if len(pop.neuron_type.description['random_distributions']) == 0:
            return ""

        res = """        if (_active){
%(update_rng_global)s
            for(int i = 0; i < size; i++) {
%(update_rng_local)s
            }
        }
        """
        local_code = ""; global_code = ""
        for rd in pop.neuron_type.description['random_distributions']:
            if Global.config['paradigm']=="openmp":
                if rd['locality'] == 'local':
                    local_code += PopTemplate.cpp_11_rng[rd['locality']]['update'] % {'id': pop.id, 'rd_name': rd['name']}
                else:
                    global_code += PopTemplate.cpp_11_rng[rd['locality']]['update'] % {'id': pop.id, 'rd_name': rd['name']}
            else:
                # HD (27.04.2016):
                # we dont need an update code here, as the drawing of random numbers is done in the Population::step()
                local_code += ""
                global_code += ""

        return res %{'update_rng_local': local_code, 'update_rng_global': global_code}

    ################################
    ### CUDA
    ################################

    def _cuda_memory_transfers(self, pop):
        """
        Before evaluation neuron/synaptic equations we need to update the data on
        the GPU. To synchronize the states of variables after simulate several steps,
        we need to transfer variables back to the host.

        Return:

            (str, str): host_device_transfer, device_host_transfer

        Notice:

            these codes are part of the run() or step() method (defined in ANNarchy.cu).
        """
        host_device_transfer = ""
        device_host_transfer = ""

        host_device_transfer += """
    // host to device transfers for %(pop_name)s""" % { 'pop_name': pop.name }
        for attr in pop.neuron_type.description['parameters']+pop.neuron_type.description['variables']:
            if attr['name'] in pop.neuron_type.description['local']:
                host_device_transfer += """
        // %(attr_name)s: local
        if( pop%(id)s.%(attr_name)s_dirty )
        {
            //std::cout << "Transfer pop%(id)s.%(attr_name)s" << std::endl;
            cudaMemcpy(pop%(id)s.gpu_%(attr_name)s, pop%(id)s.%(attr_name)s.data(), pop%(id)s.size * sizeof(%(type)s), cudaMemcpyHostToDevice);
            pop%(id)s.%(attr_name)s_dirty = false;
        }
""" % { 'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype'] }

        device_host_transfer += """
    // device to host transfers for %(pop_name)s\n""" % { 'pop_name': pop.name }
        for attr in pop.neuron_type.description['parameters']+pop.neuron_type.description['variables']:
            if attr['name'] in pop.neuron_type.description['local']:
                device_host_transfer += """\tcudaMemcpy(pop%(id)s.%(attr_name)s.data(), pop%(id)s.gpu_%(attr_name)s, pop%(id)s.size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
""" % { 'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype'] }

        return host_device_transfer, device_host_transfer
