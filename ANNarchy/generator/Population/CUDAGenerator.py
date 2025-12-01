"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import re
from math import ceil
from copy import deepcopy

import ANNarchy

from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_cuda as global_op_template
from ANNarchy.generator.Population import CUDATemplates
from ANNarchy.generator.Utils import generate_equation_code, tabify, check_and_apply_pow_fix
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages
from .PopulationGenerator import PopulationGenerator
from .CUDATemplates import cuda_templates

class CUDAGenerator(PopulationGenerator):
    """
    Generate the header for a Population object to use on CUDA devices.
    """
    def __init__(self, cuda_version, profile_generator, net_id):
        super(CUDAGenerator, self).__init__(profile_generator, net_id)
        self._cuda_version = cuda_version

    def header_struct(self, pop, annarchy_dir):
        """
        Specialized implementation of PopulationGenerator.header_struct() for
        generation of a CUDA header.
        """
        self._templates = deepcopy(cuda_templates)

        # Generate declaration and accessors of all parameters and variables
        declaration_parameters_variables, access_parameters_variables = self._generate_decl_and_acc(pop)

        # Additional includes and structures
        include_additional = ""
        struct_additional = ""
        declare_additional = ""
        init_additional = ""
        reset_additional = ""
        access_additional = ""

        # Initialize parameters and variables
        init_parameters_variables = self._init_population(pop)

        # Reset of device to host flags
        reset_read_flags = self._reset_read_flags(pop)

        # Spike-specific stuff
        reset_spike = ""
        declare_spike = ""
        init_spike = ""
        if pop.neuron_type.description['type'] == 'spike':
            spike_tpl = self._templates['spike_specific']

            # Main data for spiking pops
            declare_spike += spike_tpl['spike']['declare'] % {'id': pop.id}
            init_spike += spike_tpl['spike']['init'] % {'id': pop.id}
            reset_spike += spike_tpl['spike']['reset'] % {'id': pop.id}

            # If there is a refractory period
            if pop.neuron_type.refractory or pop.refractory:
                declare_spike += spike_tpl['refractory']['declare'] % {'id': pop.id}
                if isinstance(pop.neuron_type.description['refractory'], str): # no need to instantiate refractory
                    init_spike += spike_tpl['refractory']['init_extern'] % {'id': pop.id}
                else:
                    init_spike += spike_tpl['refractory']['init'] % {'id': pop.id}
                reset_spike += spike_tpl['refractory']['reset'] % {'id': pop.id}

        # Process eventual delay
        declare_delay = ""; init_delay = ""; update_delay = ""; update_max_delay = ""; reset_delay = ""
        if pop.max_delay > 1:
            declare_delay, init_delay, update_delay, update_max_delay, reset_delay = self._delay_code(pop)

        # Process mean FR computations
        declare_FR, init_FR, reset_FR = self._init_fr(pop)
        reset_spike += reset_FR

        update_FR = self._update_fr(pop)

        # Update random distributions
        update_rng = self._update_random_distributions(pop)

        # Update global operations
        update_global_ops = self._update_globalops(pop)

        # Update the neural variables
        if pop.neuron_type.type == "rate":
            body, invoke, header, call = self._update_rate_neuron(pop)
        else:
            update_body, update_invoke, update_header, update_call = self._update_spiking_neuron(pop)
            spike_body, spike_invoke, spike_header, spike_call = self._spike_gather(pop)
            body = update_body + spike_body
            invoke = update_invoke + spike_invoke
            header = update_header + spike_header
            call = update_call + spike_call
        update_variables = ""

        # Memory transfers
        host_to_device, device_to_host = self._memory_transfers(pop)

        # Stop condition
        stop_condition = self._stop_condition(pop)

        # Local functions
        host_local_func, device_local_func = self._local_functions(pop)
        declaration_parameters_variables += host_local_func

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
        if 'reset_read_flags' in pop._specific_template.keys():
            reset_read_flags = pop._specific_template['reset_read_flags']
        if 'reset_additional' in pop._specific_template.keys():
            reset_additional = pop._specific_template['reset_additional']
        if 'update_variables' in pop._specific_template.keys():
            update_variables = pop._specific_template['update_variables']
        if 'update_rng' in pop._specific_template.keys():
            update_rng = pop._specific_template['update_rng']
        if 'update_FR' in pop._specific_template.keys():
            update_FR = pop._specific_template['update_FR']
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
            'extern_global_operations': "", # CPU side global ops
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
            'reset_read_flags': reset_read_flags,
            'update_FR': update_FR,
            'update_variables': update_variables,
            'update_rng': update_rng,
            'update_delay': update_delay,
            'update_max_delay': update_max_delay,
            'update_global_ops': update_global_ops,
            'stop_condition': stop_condition,
            'host_to_device': host_to_device,
            'device_to_host': device_to_host,
            'size_in_bytes': size_in_bytes,
            'clear_container': clear_container
        }

        # Store the complete header definition in a single file
        with open(annarchy_dir+'/generate/net'+str(self._net_id)+'/pop'+str(pop.id)+'.hpp', 'w') as ofile:
            ofile.write(code)

        # Basic informations common to all populations
        pop_desc = {
            'include': """#include "pop%(id)s.hpp"\n""" % {'id': pop.id},
            'extern': """extern PopStruct%(id)s* pop%(id)s;\n"""% {'id': pop.id},
            'instance': """PopStruct%(id)s* pop%(id)s;\n"""% {'id': pop.id},
            'init': """    pop%(id)s->init_population();\n""" % {'id': pop.id}
        }

        pop_desc['custom_func'] = device_local_func
        pop_desc['update'] = call
        pop_desc['update_invoke'] = invoke
        pop_desc['update_body'] = body
        pop_desc['update_header'] = header
        pop_desc['update_delay'] = """    pop%(id)s->update_delay();\n""" % {'id': pop.id} if pop.max_delay > 1 else ""
        pop_desc['update_FR'] = """    pop%(id)s->update_FR();\n""" % {'id': pop.id} if pop.neuron_type.type == "spike" else ""

        if len(pop.global_operations) > 0:
            pop_desc['gops_update'] = self._update_globalops(pop) % {'id': pop.id}

        pop_desc['host_to_device'] = tabify("pop%(id)s->host_to_device();" % {'id':pop.id}, 1)+"\n"

        return pop_desc

    def _clear_container(self, pop):
        """
        Clear allocated data structures.

        The function overrrides the default behavior as we need a de-allocation
        on host and device side.
        """
        from ANNarchy.generator.Utils import tabify

        # Variables (host-side), which contains also Mean-FR code
        code = PopulationGenerator._clear_container(self, pop)

        # Variables (device side)
        code += "// parameters\n"
        for attr in pop.neuron_type.description['parameters']:
            if attr['locality'] == "local":
                code += """cudaFree(gpu_%(name)s); \n""" % {'name': attr['name']}

        code += "\n// variables\n"
        for attr in pop.neuron_type.description['variables']:
            code += """cudaFree(gpu_%(name)s); \n""" % {'name': attr['name']}

        code += "\n// delayed attributes\n"
        if pop.neuron_type.type == "rate":
            delay_tpl = self._templates['attribute_delayed']
            for var in pop.delayed_variables:
                if var in pop.neuron_type.description['local']:
                    code += delay_tpl['local']['clear'] % {'name': var}
                else:
                    continue

        # clear PSP targets ( for rate-coded neurons, for spiking they are part of population variables )
        code += "\n// targets\n"
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                code += """cudaFree(gpu__sum_%(target)s); \n""" % {'target': target}

        # just code layout
        code = tabify(code, 2)

        # Sanitiy check
        code += """
    #ifdef _DEBUG
        cudaError_t err_clear = cudaGetLastError();
        if ( err_clear != cudaSuccess )
            std::cout << "Pop%(id)s::clear() - cudaFree: " << cudaGetErrorString(err_clear) << std::endl;
    #endif
""" % {'id': pop.id}

        # code complete
        return code

    def reset_computesum(self, pop):
        """
        For rate-coded networks we need to reset the weighted sum in each step, as the kernels simply
        add up.
        """
        if pop.neuron_type.type != 'rate':
            return ""

        code = ""
        for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
            code += """
    #if defined (__pop%(id)s_nb__)
        call_clear_sum( RunConfig(__pop%(id)s_nb__, __pop%(id)s_tpb__, 0, pop%(id)s->stream), pop%(id)s->size, pop%(id)s->gpu__sum_%(target)s );
    #else
        call_clear_sum( RunConfig(pop%(id)s->_nb_blocks, pop%(id)s->_threads_per_block, 0, pop%(id)s->stream), pop%(id)s->size, pop%(id)s->gpu__sum_%(target)s );
    #endif
""" % {'id': pop.id, 'target': target}

        if code != "":
            code = """
    if ( pop%(id)s->_active ) {
%(reset_code)s
    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "clear_sum: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }
""" % {'reset_code': code, 'id': pop.id}

        return code

    def _reset_read_flags(self, pop):
        """
        Reset of device to host flags.

        The read-back from the GPUs happens if the first time is accessed the variable. To know, if the GPU simulated
        between two subsequent calls of such an getter, we store the time stamp of the last read-back. This must be set
        back to 0 in case of a reset.
        """
        reset_read_flags = "// read-back flags: variables\n"

        attributes = []
        for var in pop.neuron_type.description["variables"]:
            if var['name'] in attributes:
                continue

            reset_read_flags += var['name']+"_device_to_host = 0;\n"
            attributes.append(var['name'])

        reset_read_flags += "\n// read-back flags: targets\n"
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                reset_read_flags += "_sum_"+target+"_device_to_host = 0;\n"

        return tabify(reset_read_flags, 2)

    def _delay_code(self, pop):
        """
        Generate code for delayed variables, comprising of initialization
        and update codes.

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
        delay_tpl = self._templates['attribute_delayed']

        # Declaration
        declare_code = """
    // Delayed variables"""
        if pop.neuron_type.type == "rate":
            for var in pop.delayed_variables:
                attr = self._get_attr(pop, var)
                declare_code += delay_tpl[attr['locality']]['declare'] % {'var': var, 'type': attr['ctype']}
        else:
            # Spiking networks should only exchange spikes
            declare_code += """
    // Delays for spike population
    std::deque< int* > gpu_delayed_spiked;              // contains a set of device pointers
    std::deque< unsigned int > host_delayed_num_events; // number of events stored in each container
"""
            # TODO:
            if pop.delayed_variables != []:
                raise NotImplementedError

        # Initialization
        init_code = """
        // Delayed variables"""

        # Update and Reset for delayed variables
        update_code = ""
        reset_code = ""
        resize_code = ""
        for var in pop.delayed_variables:
            attr = self._get_attr(pop, var)

            attr_dict = {
                'id': pop.id,
                'name': attr['name'],
                'type': attr['ctype'],
                'delay': pop.max_delay
            }

            init_code += delay_tpl[attr['locality']]['init'] % attr_dict
            update_code += delay_tpl[attr['locality']]['update'] % attr_dict
            reset_code += delay_tpl[attr['locality']]['reset'] % attr_dict
            resize_code += delay_tpl[attr['locality']]['resize'] % attr_dict

        # Delaying spike events is done differently
        if pop.neuron_type.type == 'spike':
            init_code += """
            gpu_delayed_spiked = std::deque<int*>();
            host_delayed_num_events = std::deque<unsigned int>();
            int *dev_spiked;

            for(int i = 0; i < max_delay; i++) {
                // we allocate max size
                cudaMalloc((void**)&dev_spiked, size * sizeof(int));
                gpu_delayed_spiked.push_front(dev_spiked);

                // true number of events stored in dev_spiked
                host_delayed_num_events.push_front(static_cast<unsigned int>(0));
            }
            """ % {'max_delay': int(ceil(pop.max_delay/ConfigManager().get('dt', self._net_id)))}
            update_code += """
            // Take the last vector in queue and push it to front
            int* last_spiked = gpu_delayed_spiked.back();
            gpu_delayed_spiked.pop_back();
            gpu_delayed_spiked.push_front(last_spiked);

            // do not copy empty vectors!
            if (spike_count > 0) {
                cudaMemcpy( last_spiked, gpu_spiked, spike_count * sizeof(int), cudaMemcpyDeviceToDevice);

            #ifdef _DEBUG
                auto err1 = cudaGetLastError();
                if (err1 != cudaSuccess) {
                    std::cerr << "PopStruct%(id)s::update_delay() - spiked :" << cudaGetErrorString(err1) << std::endl;
                }
            #endif
            }

            host_delayed_num_events.pop_back();
            host_delayed_num_events.push_front(spike_count);

        #ifdef _DEBUG
            std::cout << "PopStruct::update_delay() at t = " << t << std::endl;
            std::cout << "[";
            for (auto i = 0; i < host_delayed_num_events.size(); i++)
                std::cout << host_delayed_num_events[i] << ", ";
            std::cout << "]" << std::endl;
        #endif
            """ % {'id': pop.id}
            reset_code += ""

        update_code = """
        if ( _active ) {
%(code)s
        }""" % {'code': update_code}

        return declare_code, init_code, update_code, resize_code, reset_code

    def _init_fr(self, pop):
        """
        Declares arrays for computing the mean FR of a spiking neuron.

        HD ( 09. March 2017 ):

            As a queue is hard to realize on the device,
            we do the computation on the CPU - side for now.
        """
        declare_FR = ""; init_FR = ""; reset_FR = ""
        if pop.neuron_type.description['type'] == 'spike':
            declare_FR = """
    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    %(float_prec)s _mean_fr_rate;
    void compute_firing_rate( %(float_prec)s window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = %(float_prec)s(1000./%(float_prec)s(window));
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

    def _gen_kernel_args(self, pop, locality):
        """
        Generate the argument and call statemen for neural variables
        used in equations as well as there dependencies.
        """
        # Gather all variable names
        add_args_header = "const long int t, const %(type)s dt" % {'type':ConfigManager().get('precision', self._net_id)}
        add_args_invoke = "t, dt"
        add_args_call = "t, dt"

        deps = []

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['locality'] == locality:
                deps.append(var['name'])
                if 'dependencies' in var.keys():
                    deps += var['dependencies']

        # Random distributions
        for rd in pop.neuron_type.description['random_distributions']:
            for dep in rd['dependencies']:
                deps.append(dep)

        # Remove doublons / sort the attribute names to
        # prevent unnecessary recompiles
        deps = sorted(list(set(deps)))

        # Generate the header and call lines
        for dep in deps:
            attr_type, attr_dict = self._get_attr_and_type(pop, dep)
            if attr_type is None:
                continue

            ids = {
                'id': pop.id,
                'name': attr_dict['name'],
                'type': attr_dict['ctype']
            }

            if attr_type == 'par':
                if dep in pop.neuron_type.description['global']:
                    add_args_header += ", const %(type)s %(name)s" % ids
                    add_args_invoke += ", %(name)s" % ids
                    add_args_call += ", pop%(id)s->%(name)s" % ids
                else:
                    add_args_header += ", %(type)s* __restrict__ %(name)s" % ids
                    add_args_invoke += ", %(name)s" % ids
                    add_args_call += ", pop%(id)s->gpu_%(name)s" % ids
            elif attr_type == 'var':
                add_args_header += ", %(type)s* __restrict__ %(name)s" % ids
                add_args_invoke += ", %(name)s" % ids
                add_args_call += ", pop%(id)s->gpu_%(name)s" % ids
            elif attr_type == 'rand':
                add_args_header += ", curandState* state_%(name)s" % ids
                add_args_invoke += ", state_%(name)s" % ids
                add_args_call += ", pop%(id)s->gpu_%(name)s" % ids
            else:
                raise NotImplementedError

        return add_args_header, add_args_invoke, add_args_call

    def _local_functions(self, pop):
        """
        Definition of user-defined local functions attached to
        a neuron. These functions will take place in the
        ANNarchyDevice.cu file.

        As the local functions can be occur repeatadly in the same file,
        there are modified with pop[id]_ to unique them.

        Return:

            * host_define, device_define
        """
        # Local functions
        if len(pop.neuron_type.description['functions']) == 0:
            return "", ""

        host_code = ""
        device_code = ""
        for func in pop.neuron_type.description['functions']:
            cpp_func = func['cpp'] + '\n'

            host_code += cpp_func
            # TODO: improve code
            if (ConfigManager().get('precision', self._net_id) == "float"):
                device_code += cpp_func.replace('float ' + func['name'], '__device__ float pop%(id)s_%(func)s' % {'id': pop.id, 'func': func['name']})
            else:
                device_code += cpp_func.replace('double ' + func['name'], '__device__ double pop%(id)s_%(func)s' % {'id': pop.id, 'func': func['name']})

        return host_code, check_and_apply_pow_fix(device_code, self._cuda_version)

    def _replace_local_funcs(self, pop, glob_eqs, loc_eqs):
        """
        As the local functions can be occur repeatadly in the same file,
        there are modified with pop[id]_ to unique them. Now we need
        to adjust the call accordingly.

        Please note: the placeholder like %(global_idx)s etc. must be already filled.
        """
        for func in pop.neuron_type.description['functions']:
            search_term = r"%(name)s\([^\(]*\)" % {'name': func['name']}

            func_occur = re.findall(search_term, glob_eqs)
            for term in func_occur:
                glob_eqs = loc_eqs.replace(term, term.replace(func['name'], 'pop'+str(pop.id)+'_'+func['name']))

            func_occur = re.findall(search_term, loc_eqs)
            for term in func_occur:
                loc_eqs = loc_eqs.replace(term, term.replace(func['name'], 'pop'+str(pop.id)+'_'+func['name']))

        return glob_eqs, loc_eqs

    def _replace_random(self, loc_eqs, glob_eqs, random_distributions):
        """
        We replace the rand_%(id)s by the corresponding curand... term.

        Update (ANNarchy 4.6.10.1):

            Further we need to ensure to draw the variables before solving the ODEs, otherwise we obtain wrong results for
            higher order solving methods.
        """
        # double precision methods have a postfix
        prec_extension = "" if ConfigManager().get('precision', self._net_id) == "float" else "_double"

        loc_pre = ""
        glob_pre = ""
        pre_loop = ""
        post_loop = ""
        for rd in random_distributions:
            if rd['locality'] == "local":
                term = ""
                if rd['dist'] == "Uniform":
                    term = """( curand_uniform%(postfix)s( &loc_state_%(rd)s ) * (%(max)s - %(min)s) + %(min)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1]}
                elif rd['dist'] == "Normal":
                    term = """( curand_normal%(postfix)s( &loc_state_%(rd)s ) * %(sigma)s + %(mean)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(",")[0], 'sigma': rd['args'].split(",")[1]}
                elif rd['dist'] == "LogNormal":
                    term = """( curand_log_normal%(postfix)s( &loc_state_%(rd)s, %(mean)s, %(std_dev)s) )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1]}
                else:
                    Messages._error("Unsupported random distribution on GPUs: " + rd['dist'])

                # suppress local index
                loc_eqs = loc_eqs.replace(rd['name']+"%(local_index)s", rd['name'])

                # add the init
                loc_pre += "%(prec)s %(name)s = %(term)s;" % {'prec': ConfigManager().get('precision', self._net_id), 'name': rd['name'], 'term': term}

                # read-out/write-back of the RNG state
                pre_loop += f"curandState loc_state_{rd['name']} = state_{rd['name']}[tid];"
                post_loop += f"state_{rd['name']}[tid] = loc_state_{rd['name']};"

            else:
                # For global variables we directly access the RNG state.
                term = ""
                if rd['dist'] == "Uniform":
                    term = """( curand_uniform%(postfix)s( &state_%(rd)s[0] ) * (%(max)s - %(min)s) + %(min)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1]}
                elif rd['dist'] == "Normal":
                    term = """( curand_normal%(postfix)s( &state_%(rd)s[0] ) * %(sigma)s + %(mean)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(",")[0], 'sigma': rd['args'].split(",")[1]}
                elif rd['dist'] == "LogNormal":
                    term = """( curand_log_normal%(postfix)s( &state_%(rd)s[0], %(mean)s, %(std_dev)s) )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1]}
                else:
                    Messages._error("Unsupported random distribution on GPUs: " + rd['dist'])

                # suppress global index
                glob_eqs = glob_eqs.replace(rd['name']+"%(global_index)s", rd['name'])

                # add the init
                glob_pre += "%(prec)s %(name)s = %(term)s;" % {'prec': ConfigManager().get('precision', self._net_id), 'name': rd['name'], 'term': term}

        # check which equation blocks we need to extend
        if len(loc_pre) > 0:
            loc_eqs = tabify(loc_pre, 2) + "\n" + loc_eqs
        if len(glob_pre) > 0:
            glob_eqs = tabify(glob_pre, 1) + "\n" + glob_eqs

        return loc_eqs, glob_eqs, pre_loop, post_loop

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
            # HD (3rd May 2023): do not use spiked.empty as for single-thread/openMP
            #                    as this container is pre-allocated to speedup monitor readout.
            stop_code = """
    // Stop condition (any)
    bool stop_condition() {
        return (spike_count>0);
    } """
            return stop_code

        elif pop.stop_condition.replace(" ", "") == "spiked:all":
            stop_code = """
    // Stop condition (all)
    bool stop_condition() {
        return spike_count == size;
    } """
            return stop_code

        # Process the stop condition
        pop.neuron_type.description['stop_condition'] = {'eq': pop.stop_condition}
        from ANNarchy.parser.Extraction import extract_stop_condition
        extract_stop_condition(pop.neuron_type.description, pop.net_id)

        mem_transfer = ""
        for dep in pop.neuron_type.description['stop_condition']['dependencies']:
            attr = self._get_attr(pop, dep)

            if attr['locality'] == "local":
                mem_transfer += """
    cudaMemcpy( %(attr_name)s.data(),  gpu_%(attr_name)s, size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
""" % {'attr_name': attr['name'], 'type': attr['ctype']}

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
        %(mem_transfer)s
        for(int i=0; i<size; i++)
        {
            if(%(condition)s){
                return true;
            }
        }
        return false;
    }
    """ % {'condition': condition, 'mem_transfer': mem_transfer}
        else:
            stop_code = """
    // Stop condition (all)
    bool stop_condition(){
        %(mem_transfer)s
        for(int i=0; i<size; i++)
        {
            if(!(%(condition)s)){
                return false;
            }
        }
        return true;
    }
    """ % {'condition': condition, 'mem_transfer': mem_transfer}

        return stop_code

    def _update_fr(self, pop):
        """
        Computes the average firing rate based on history.

        HD ( 09. March 2017 ):

            As a queue is hard to realize on the device,
            we do the computation on the CPU - side for now.
        """
        mean_FR_update = ""
        if pop.neuron_type.description['type'] == 'spike':
            mean_FR_update = """
        if ( _mean_fr_window > 0) {
            // Update the queues
            r_host_to_device = false;

            for ( int i = 0; i < spike_count; i++ ) {
                _spike_history[spiked[i]].push(t);
                r_host_to_device = true; // the queue changed the length
            }

            // Recalculate the mean firing rate
            for (int i = 0; i < size; i++ ) {
                while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                    _spike_history[i].pop(); // Suppress spikes outside the window
                    r_host_to_device = true; // the queue changed the length
                }
                r[i] = _mean_fr_rate * float(_spike_history[i].size());
            }

            // transfer to device
            if ( r_host_to_device ) {
                cudaMemcpy(gpu_r, r.data(), size * sizeof(%(float_prec)s), cudaMemcpyHostToDevice);
                r_host_to_device = false;
            }
        }
"""
        return mean_FR_update % {'float_prec': ConfigManager().get('precision', self._net_id)}

    def _update_globalops(self, pop):
        """
        Update of global functions is a call of pre-implemented
        functions defined in GlobalOperationTemplate. In case of
        CUDA the call semantic will be placed in ANNarchy.cu
        file as part of the host section.
        """
        if len(pop.global_operations) == 0:
            return ""

        code = ""
        for op in pop.global_operations:
            code += global_op_template[op['function']]['call'] % {'id': pop.id, 'type': ConfigManager().get('precision', self._net_id), 'op': op['function'], 'var': op['variable']}

        return code

    def _update_random_distributions(self, pop):
        # HD (27.04.2016):
        # we dont need an update code here, as the drawing of random numbers is done in the Population::step()
        return ""

    def _update_rate_neuron(self, pop):
        """
        Generate the execution code for updating neural variables, more precise local and global ones.

        The resulting code comprise of several parts: creating of local and global update code, generating
        function prototype and finally calling statement.

        Returns:

            * tuple of four code snippets (device_kernel, device_invoke, kernel_decl, host_call)

        """
        # Use pre-defined code template
        if 'update_variables' in pop._specific_template.keys():
            try:
                return pop._specific_template['update_variable_body'], pop._specific_template['update_variable_invoke'], pop._specific_template['update_variable_header'], pop._specific_template['update_variable_call']
            except KeyError:
                Messages._error("\nCode generation error: if one attempts to override the population update on CUDA devices, one need to define all of the following fields of _specific_template dictionary:\n\tupdate_variables, update_variable_call, update_variable_header, update_variable_invoke, update_variable_body")

        # Is there any variable?
        if len(pop.neuron_type.description['variables']) == 0:
            return "", "", "", ""

        device_kernel = ""
        kernel_invoke = ""
        kernel_decl = ""
        local_call = ""
        global_call = ""

        # some defaults
        ids = {
            'id': pop.id,
            'local_index': "[i]",
            'global_index': '[0]'
        }

        # parse the equations
        glob_eqs = generate_equation_code(pop.neuron_type.description, locality='global', padding=1)
        loc_eqs = generate_equation_code(pop.neuron_type.description, locality='local', padding=2)

        # Remove %(global_index)s for global parameters
        for par in pop.neuron_type.description['parameters']:
            if par['locality'] == "global":
                glob_eqs = glob_eqs.replace(par['name']+"%(global_index)s", par['name'])
                loc_eqs = loc_eqs.replace(par['name']+"%(global_index)s", par['name'])

        # Gather pre-loop declaration (dt/tau for ODEs)
        pre_loop = ""
        for var in pop.neuron_type.description['variables']:
            if 'pre_loop' in var.keys() and len(var['pre_loop']) > 0:
                pre_loop += ConfigManager().get('precision', self._net_id) + ' ' + var['pre_loop']['name'] + ' = ' + var['pre_loop']['value'] + ';\n'
        if pre_loop.strip() != '':
            pre_loop = """
// Updating the step sizes
""" + pre_loop

        # sum() must generate _sum___all__[i] = _sum_exc[i] + sum_inh[i] + ... at the beginning of local equations
        if '__all__' in pop.neuron_type.description['targets']:
            eqs = " "*8 + "// Sum over all targets\n"
            eqs += " "*8 + "_sum___all__[i] = "
            for target in pop.targets:
                eqs += "_sum_" + target + '[i] + '
            eqs = eqs[:-2]
            eqs += ';\n\n'
            loc_eqs = eqs + loc_eqs

        # replace pow() for SDK < 6.5
        loc_eqs = check_and_apply_pow_fix(loc_eqs, self._cuda_version)
        glob_eqs = check_and_apply_pow_fix(glob_eqs, self._cuda_version)

        # replace the random distributions
        loc_eqs, glob_eqs, rng_pre_loop, rng_post_loop = self._replace_random(loc_eqs, glob_eqs, pop.neuron_type.description['random_distributions'])
        pre_loop += rng_pre_loop

        # Replace %(global_idx)s for global parameters
        for var in pop.neuron_type.description["global"]:
            attr_type, attr_dict = self._get_attr_and_type(pop, var)
            if attr_type == 'par':
                if pre_loop.strip() != '':
                    pre_loop = pre_loop.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])
                if glob_eqs.strip() != '':
                    glob_eqs = glob_eqs.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])
                if loc_eqs.strip() != '':
                    loc_eqs = loc_eqs.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])

        # Fill the placeholder
        glob_eqs = glob_eqs % ids
        loc_eqs = loc_eqs % ids
        pre_loop = pre_loop % ids
        rng_post_loop = rng_post_loop % ids

        # replace local function calls
        if len(pop.neuron_type.description['functions']) > 0:
            glob_eqs, loc_eqs = self._replace_local_funcs(pop, glob_eqs, loc_eqs)

        # Global operations
        if glob_eqs.strip() != '':
            add_args_header, add_args_invoke, add_args_call = self._gen_kernel_args(pop, 'global')

            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': ConfigManager().get('precision', self._net_id),
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s->_%(op)s_%(var)s""" % ids

            # finalize code templates
            device_kernel += CUDATemplates.population_update_kernel['global']['device_kernel'] % {
                'id': pop.id,
                'add_args': add_args_header,
                'global_eqs':glob_eqs,
                'pre_loop': tabify(pre_loop, 1)
            }
            kernel_invoke += CUDATemplates.population_update_kernel['global']['invoke_kernel'] % {
                'id': pop.id,
                'add_args': add_args_header,
                'add_args_call': add_args_invoke,
            }
            kernel_decl += CUDATemplates.population_update_kernel['global']['kernel_decl'] % {
                'id': pop.id, 'add_args': add_args_header
            }
            global_call = CUDATemplates.population_update_kernel['global']['host_call'] % {
                'id': pop.id, 'add_args': add_args_call
            }

        # Local variables
        if loc_eqs.strip() != '':
            add_args_header, add_args_invoke, add_args_call = self._gen_kernel_args(pop, 'local')

            # targets
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                add_args_header += """, %(type)s* _sum_%(target)s""" % {'type': ConfigManager().get('precision', self._net_id), 'target' : target}
                add_args_invoke += """, _sum_%(target)s""" % {'target' : target}
                add_args_call += """, pop%(id)s->gpu__sum_%(target)s""" % {'id': pop.id, 'target' : target}

            # global operations
            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': ConfigManager().get('precision', self._net_id),
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_invoke += """, _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s->_%(op)s_%(var)s""" % ids

            # finalize code templates
            device_kernel += CUDATemplates.population_update_kernel['local']['device_kernel'] % {
                'id': pop.id,
                'add_args': add_args_header,
                'pop_size': pop.size,
                'local_eqs': loc_eqs,
                'pre_loop': tabify(pre_loop, 1),
                'post_loop': tabify(rng_post_loop, 1)
            }
            kernel_invoke += CUDATemplates.population_update_kernel['local']['invoke_kernel'] % {
                'id': pop.id,
                'add_args': add_args_header,
                'add_args_call': add_args_invoke,
            }
            kernel_decl += CUDATemplates.population_update_kernel['local']['kernel_decl'] % {
                'id': pop.id,
                'add_args': add_args_header
            }
            local_call = CUDATemplates.population_update_kernel['local']['host_call'] % {
                'id': pop.id,
                'add_args': add_args_call
            }

        # Call statement consists of two parts
        host_call = """
    // Updating the local and global variables of population %(id)s
    if ( pop%(id)s->_active ) {
        %(global_call)s

        %(local_call)s
    }
""" % {'id':pop.id, 'global_call': global_call, 'local_call': local_call}

        if self._prof_gen:
            host_call = self._prof_gen.annotate_update_neuron(pop, host_call)

        return device_kernel, kernel_invoke, kernel_decl, host_call

    def _update_spiking_neuron(self, pop):
        """
        Generate the neural update code for GPU devices. We split up the
        calculation into three parts:

            * evolvement of global differential equations
            * evolvement of local differential equations
            * spike gathering

        Return:

            * tuple of four code snippets (device_kernel, device_invoke, kernel_decl, host_call)
        """
        # Use pre-defined code template
        if 'update_variables' in pop._specific_template.keys():
            try:
                return pop._specific_template['update_variable_body'], pop._specific_template['update_variable_invoke'], pop._specific_template['update_variable_header'], pop._specific_template['update_variable_call']
            except KeyError:
                Messages._error("\nCode generation error: if one attempts to override the population update on CUDA devices, one need to define all of the following fields of _specific_template dictionary:\n\tupdate_variables, update_variable_call, update_variable_header, update_variable_invoke, update_variable_body")

        # Is there any variable?
        if len(pop.neuron_type.description['variables']) == 0:
            return "", "", "", ""

        kernel_decl = ""
        device_kernel = ""
        device_invoke = ""
        local_call = ""
        global_call = ""

        # some defaults
        ids = {
            'id': pop.id,
            'local_index': "[i]",
            'global_index': "[0]"
        }

        #
        # Process the global and local equations and generate first set of kernels
        #

        # parse the equations
        glob_eqs = generate_equation_code(pop.neuron_type.description, locality='global', padding=1)
        loc_eqs = generate_equation_code(pop.neuron_type.description, locality='local', padding=2)

        # Gather pre-loop declaration (dt/tau for ODEs) and
        # update the related kernels
        pre_code = ""
        for var in pop.neuron_type.description['variables']:
            if 'pre_loop' in var.keys() and len(var['pre_loop']) > 0:
                pre_code += ConfigManager().get('precision', self._net_id) + ' ' + var['pre_loop']['name'] + ' = ' + var['pre_loop']['value'] + ';\n'
        if pre_code.strip() != '':
            pre_code = """
    // Updating the step sizes
""" + tabify(pre_code, 1)

        # replace pow() for SDK < 6.5
        loc_eqs = check_and_apply_pow_fix(loc_eqs, self._cuda_version)
        glob_eqs = check_and_apply_pow_fix(glob_eqs, self._cuda_version)

        # replace the random distributions
        loc_eqs, glob_eqs, rng_pre_loop, rng_post_loop = self._replace_random(loc_eqs, glob_eqs, pop.neuron_type.description['random_distributions'])
        pre_code += rng_pre_loop

        # within refractory perid, only conductance variables
        if pop.neuron_type.refractory or pop.refractory:
            refr_eqs = generate_equation_code(pop.neuron_type.description, 'local', conductance_only=True, padding=3)
            loc_eqs = tabify(loc_eqs, 1) # just for better code layout
        else:
            refr_eqs = ''

        # Replace %(global_idx)s for global parameters
        for var in pop.neuron_type.description["global"]:
            attr_type, attr_dict = self._get_attr_and_type(pop, var)
            if attr_type == 'par':
                if pre_code.strip() != '':
                    pre_code = pre_code.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])
                if refr_eqs.strip() != '':
                    refr_eqs = refr_eqs.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])
                if glob_eqs.strip() != '':
                    glob_eqs = glob_eqs.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])
                if loc_eqs.strip() != '':
                    loc_eqs = loc_eqs.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])

        # Fill the placeholder
        glob_eqs = glob_eqs % ids
        loc_eqs = loc_eqs % ids
        pre_code = pre_code % ids
        rng_post_loop = rng_post_loop % ids
        refr_eqs = refr_eqs % ids

        # replace local function calls
        if len(pop.neuron_type.description['functions']) > 0:
            glob_eqs, loc_eqs = self._replace_local_funcs(pop, glob_eqs, loc_eqs)

        # Global variables
        if glob_eqs.strip() != '':
            add_args_header, add_args_invoke, add_args_call = self._gen_kernel_args(pop, 'global')

            # global operations
            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': ConfigManager().get('precision', self._net_id),
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_invoke += """, _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s->_%(op)s_%(var)s""" % ids

            # finalize code templates
            body += CUDATemplates.population_update_kernel['global']['body'] % {
                'id': pop.id,
                'add_args': add_args_header,
                'pre_loop': pre_code,
                'post_loop': rng_post_loop,
                'global_eqs':glob_eqs
            }
            header += CUDATemplates.population_update_kernel['global']['header'] % {
                'id': pop.id, 'add_args': add_args_header
            }
            global_call = CUDATemplates.population_update_kernel['global']['call'] % {
                'id': pop.id, 'add_args': add_args_call, 'stream_id': pop.id
            }

        # Local variables
        if loc_eqs.strip() != '':
            add_args_header, add_args_invoke, add_args_call = self._gen_kernel_args(pop, 'local')

            # global operations
            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': ConfigManager().get('precision', self._net_id),
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_invoke += """, _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s->_%(op)s_%(var)s""" % ids

            # Is there a refractory period?
            if pop.neuron_type.refractory or pop.refractory:
                # 'old' loc_eqs is now only executed ouside refractory period
                loc_eqs = """
        // Refractory period
        if( refractory_remaining[i] > 0){
%(refr_eqs)s
            // Decrement the refractory period
            refractory_remaining[i]--;
        } else{
%(loc_eqs)s
        }
""" %  {'refr_eqs': refr_eqs, 'loc_eqs': loc_eqs}

                add_args_header += ", int* refractory_remaining"
                add_args_invoke += ", refractory_remaining"
                add_args_call += """, pop%(id)s->gpu_refractory_remaining""" %{'id':pop.id}

            # finalize code templates
            device_kernel += CUDATemplates.population_update_kernel['local']['device_kernel'] % {
                'id': pop.id, 'add_args': add_args_header, 'pop_size': pop.size, 'pre_loop': tabify(pre_code, 1), 'post_loop': tabify(rng_post_loop, 1), 'local_eqs': loc_eqs
            }
            device_invoke += CUDATemplates.population_update_kernel['local']['invoke_kernel'] % {
                'id': pop.id, 'add_args': add_args_header, 'add_args_call': add_args_invoke
            }
            kernel_decl += CUDATemplates.population_update_kernel['local']['kernel_decl'] % {
                'id': pop.id, 'add_args': add_args_header
            }
            local_call = CUDATemplates.population_update_kernel['local']['host_call'] % {
                'id': pop.id, 'add_args': add_args_call, 'stream_id': pop.id
            }

        # Call statement consists of two parts
        host_call = """
    // Updating the local and global variables of population %(id)s
    if ( pop%(id)s->_active ) {
        %(global_call)s

        %(local_call)s
    }
""" % {'id':pop.id, 'global_call': global_call, 'local_call': local_call}

        if self._prof_gen:
            host_call = self._prof_gen.annotate_update_neuron(pop, host_call)

        return device_kernel, device_invoke, kernel_decl, host_call

    def _spike_gather(self, pop):
        """
        Process the spike condition and generate the corresponding kernel including call code.

        Returns: four separate strings containg:

        *device_kernel*:    device code
        *invoke_kernel*:    defice function invocation
        *kernel_decl*:      header definition for invoke_kernel
        *host_call*:        device function call
        """
        # some defaults
        ids = {
            'id': pop.id,
            'local_index': "[i]",
            'global_index': "[0]"
        }

        if 'spike_gather_body' in pop._specific_template.keys():
            try:
                return pop._specific_template['spike_gather_body'], pop._specific_template['spike_gather_invoke'], pop._specific_template['spike_gather_header'], pop._specific_template['spike_gather_call']
            except KeyError:
                Messages._error("\nCode generation error: if one attempts to override the spike gathering on CUDA devices, one need to define all of the following fields of _specific_template dictionary: spike_gather_call, spike_gather_header, spike_gather_body")

        cond = pop.neuron_type.description['spike']['spike_cond']
        reset = ""
        for eq in pop.neuron_type.description['spike']['spike_reset']:
            reset += """
            %(reset)s
""" % {'reset': eq['cpp']}

        # arguments
        header_args = ""
        header_invoke = ""
        call_args = ""

        # gather all attributes required by this kernel
        kernel_deps = []
        for var in pop.neuron_type.description['spike']['spike_cond_dependencies']:
            kernel_deps.append(var)
        for reset_eq in pop.neuron_type.description['spike']['spike_reset']:
            kernel_deps.append(reset_eq['name'])
            for var in reset_eq['dependencies']:
                kernel_deps.append(var)

        # remove doubled entries and sort to prevent unnecessary recompiles
        kernel_deps = sorted(list(set(kernel_deps)))

        # generate header, call and body args
        for var in kernel_deps:
            attr_type, attr_dict = self._get_attr_and_type(pop, var)

            if attr_type == 'par' and attr_dict['locality'] == "global":
                header_args += ", const " + attr_dict['ctype'] + " " + var
                header_invoke += ", " + var
                call_args += ", pop"+str(pop.id)+"->"+var

                cond = cond.replace(var+"%(global_index)s", var)
                reset = reset.replace(var+"%(global_index)s", var)
            else:
                header_args += ", " + attr_dict['ctype'] + "* " + var
                header_invoke += ", " + var
                call_args += ", pop"+str(pop.id)+"->gpu_"+var

        # Fill the templates with the correct ids
        cond = cond % ids
        reset = reset % ids

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
                    Messages._error("refractory = "+ pop.neuron_type.refractory + ": parameter or variable does not exist.")

                refrac_inc = "refractory_remaining[i] = %(refrac_var)s;" % {'refrac_var': refrac_var}
                header_args += ", %(type)s *%(name)s, int* refractory_remaining" % {'type': param['ctype'], 'name': param['name']}
                header_invoke += ", %(name)s, refractory_remaining" % {'name': param['name']}
                call_args += ", pop%(id)s->gpu_%(name)s, pop%(id)s->gpu_refractory_remaining" %{'id':pop.id, 'name': param['name']}

            else: # default case
                refrac_inc = "refractory_remaining[i] = %(refrac_var)s;" % {'refrac_var': refrac_var}
                header_args += ", int *refractory, int* refractory_remaining"
                header_invoke += ", refractory, refractory_remaining"
                call_args += """, pop%(id)s->gpu_refractory, pop%(id)s->gpu_refractory_remaining""" %{'id':pop.id}
        else:
            refrac_inc = ""

        # With ANNarchy 4.7.2 we introduced two different kernel:
        # a) single block (standard version prior to ANNarchy 4.7.2)
        # b) multiple blocks (new in ANNarchy 4.7.2)
        if pop.size < 32:
            launch_config = """int tpb = 32;\nint nb_blocks = 1;\n"""
        else:
            launch_config = """int tpb = 32;\nint nb_blocks = %(nb)s;\n""" % {'nb': int(min(65535, float(pop.size)/32.0))}
        launch_config = tabify(launch_config, 2)

        device_kernel = CUDATemplates.spike_gather_kernel['device_kernel'] % {
            'id': pop.id,
            'pop_size': str(pop.size),
            'float_prec': ConfigManager().get('precision', self._net_id),
            'args': header_args,
            'cond': cond,
            'reset': reset,
            'refrac_inc': refrac_inc
        }

        invoke_kernel = CUDATemplates.spike_gather_kernel['invoke_kernel'] % {
            'id': pop.id,
            'float_prec': ConfigManager().get('precision', self._net_id),
            'args': header_args,
            'args_call': header_invoke
        }

        kernel_decl = CUDATemplates.spike_gather_kernel['kernel_decl'] % {
            'id': pop.id,
            'default': 'const long int t, const %(float_prec)s dt, int* spiked, long int* last_spike' % {'float_prec': ConfigManager().get('precision', self._net_id)},
            'args': header_args
        }

        if pop.max_delay > 1:
            default_args = 't, dt, pop%(id)s->gpu_delayed_spiked.front(), pop%(id)s->gpu_last_spike' % {'id': pop.id}
        else: # no_delay
            default_args = 't, dt, pop%(id)s->gpu_spiked, pop%(id)s->gpu_last_spike' % {'id': pop.id}

        host_call = CUDATemplates.spike_gather_kernel['host_call'] % {
            'id': pop.id,
            'default': default_args,
            'args': call_args % {'id': pop.id},
            'stream_id': pop.id,
            'launch_config': launch_config
        }

        if self._prof_gen:
            host_call = self._prof_gen.annotate_spike_gather(pop, host_call)

        return device_kernel, invoke_kernel, kernel_decl, host_call

    def _memory_transfers(self, pop):
        """
        Before evaluation neuron/synaptic equations we need to update the data
        on the GPU. To synchronize the states of variables after simulation of
        several steps, we need to transfer variables back to the host.

        Return:

            (str, str): host_device_transfer, device_host_transfer

        Notice:

            these codes are part of the run() or step() method (defined in ANNarchy.cu).
        """
        host_device_transfer = ""
        device_host_transfer = ""

        # Variables/Parameters
        host_device_transfer += """
    // host to device transfers for %(name)s""" % {'name': pop.name}
        for attr in pop.neuron_type.description['variables']:
            ids = {'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype']}
            if attr['name'] in pop.neuron_type.description['local']:
                host_device_transfer += self._templates['attribute_transfer']['HtoD_local'] % ids
            else:
                host_device_transfer += self._templates['attribute_transfer']['HtoD_global'] % ids
        for attr in pop.neuron_type.description['parameters']:
            ids = {'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype']}
            if attr['name'] in pop.neuron_type.description['local']:
                host_device_transfer += self._templates['attribute_transfer']['HtoD_local'] % ids
            else:
                # nothing to do for global parameter
                continue

        # Rate-coded targets
        if pop.neuron_type.type == "rate":
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                ids = {'attr_name': "_sum_"+target, 'type': ConfigManager().get('precision', self._net_id), 'id': pop.id}
                host_device_transfer += self._templates['attribute_transfer']['HtoD_local'] % ids

        # Refractoriness
        if pop.neuron_type.type == "spike":
            if pop.neuron_type.refractory or pop.refractory:
                host_device_transfer += """
        // refractory
        if( refractory_dirty )
        {
        #ifdef _DEBUG
            std::cout << "HtoD refractory ( pop%(id)s )" << std::endl;
        #endif
            cudaMemcpy( gpu_refractory, refractory.data(), size * sizeof(int), cudaMemcpyHostToDevice);
            refractory_dirty = false;

        #ifdef _DEBUG
            cudaError_t err = cudaGetLastError();
            if ( err!= cudaSuccess )
                std::cout << "  error " << cudaGetErrorString(err) << std::endl;
        #endif
        }
""" % {'id': pop.id}

        # Write back variables. The operation should be performed only if it hasn't done
        # since the last simulate() / step() and for a given variable. 
        for attr in pop.neuron_type.description['variables']:
            ids = {'attr_name': attr['name'], 'type': attr['ctype'], 'id': pop.id}
            if attr['name'] in pop.neuron_type.description['local']:
                device_host_transfer += self._templates['attribute_transfer']['DtoH_local'] % ids
            else:
                device_host_transfer += self._templates['attribute_transfer']['DtoH_global'] % ids

        # Write back rate-coded targets
        if pop.neuron_type.type == "rate":
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                ids = {'attr_name': "_sum_"+target, 'type': ConfigManager().get('precision', self._net_id), 'id': pop.id}
                device_host_transfer += self._templates['attribute_transfer']['DtoH_local'] % ids

        if 'host_device_transfer' in pop._specific_template.keys():
            host_device_transfer = pop._specific_template['host_device_transfer']
        if 'device_host_transfer' in pop._specific_template.keys():
            device_host_transfer = pop._specific_template['device_host_transfer']

        return host_device_transfer, device_host_transfer
