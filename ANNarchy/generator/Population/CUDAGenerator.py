#===============================================================================
#
#     CUDAGenerator.py
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
import re
from math import ceil

from ANNarchy.core import Global
from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_extern as global_op_extern_dict
from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_cuda as global_op_template
from ANNarchy.generator.Population import CUDATemplates
from ANNarchy.generator.Utils import generate_equation_code, tabify, check_and_apply_pow_fix

from .PopulationGenerator import PopulationGenerator
from .CUDATemplates import cuda_templates

class CUDAGenerator(PopulationGenerator):
    """
    Generate the header for a Population object to use on CUDA devices.
    """
    PopulationGenerator._templates = cuda_templates

    def __init__(self, profile_generator, net_id):
        super(CUDAGenerator, self).__init__(profile_generator, net_id)

    def header_struct(self, pop, annarchy_dir):
        """
        Specialized implementation of PopulationGenerator.header_struct() for
        generation of an openMP header.
        """
        # Generate declaration and accessors of all parameters and variables
        declaration_parameters_variables, access_parameters_variables = self._generate_decl_and_acc(pop)

        # Additional includes and structures
        include_additional = ""
        struct_additional = ""
        declare_additional = ""
        init_additional = ""
        reset_additional = ""
        access_additional = ""

        # Declare global operations as extern at the beginning of the file
        extern_global_operations = ""
        for op in pop.global_operations:
            extern_global_operations += global_op_extern_dict[op['function']] % {'type': Global.config['precision']}

        # Initialize parameters and variables
        init_parameters_variables = self._init_population(pop)

        # Spike-specific stuff
        reset_spike = ""
        declare_spike = ""
        init_spike = ""
        if pop.neuron_type.description['type'] == 'spike':
            spike_tpl = self._templates['spike_specific']

            # Main data for spiking pops
            declare_spike += spike_tpl['declare_spike'] % {'id': pop.id}
            init_spike += spike_tpl['init_spike'] % {'id': pop.id}
            reset_spike += spike_tpl['reset_spike'] % {'id': pop.id}

            # If there is a refractory period
            if pop.neuron_type.refractory or pop.refractory:
                declare_spike += spike_tpl['declare_refractory'] % {'id': pop.id}
                init_spike += spike_tpl['init_refractory'] % {'id': pop.id}
                reset_spike += spike_tpl['reset_refractory'] % {'id': pop.id}

        # Process eventual delay
        declare_delay = ""; init_delay = ""; update_delay = ""; reset_delay = ""
        if pop.max_delay > 1:
            declare_delay, init_delay, update_delay, reset_delay = self._delay_code(pop)

        # Process mean FR computations
        declare_FR, init_FR = self._init_fr(pop)

        # Update random distributions
        update_rng = self._update_random_distributions(pop)

        # Update global operations
        update_global_ops = self._update_globalops(pop)

        # Update the neural variables
        if pop.neuron_type.type == "rate":
            body, header, update_call = self._update_rate_neuron(pop)
        else:
            body, header, update_call = self._update_spiking_neuron(pop)
        update_variables = ""

        # Memory transfers
        host_to_device, device_to_host = self._memory_transfers(pop)

        # Stop condition
        stop_condition = self._stop_condition(pop)

        # Local functions
        host_local_func, device_local_func = self._local_functions(pop)
        declaration_parameters_variables += host_local_func

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
        if 'update_rng' in pop._specific_template.keys():
            update_rng = pop._specific_template['update_rng']
        if 'update_delay' in pop._specific_template.keys() and pop.max_delay > 1:
            update_delay = pop._specific_template['update_delay']
        if 'update_global_ops' in pop._specific_template.keys():
            update_global_ops = pop._specific_template['update_global_ops']

        # Fill the template
        code = self._templates['population_header'] % {
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
            'init_profile': init_profile,
            'reset_spike': reset_spike,
            'reset_delay': reset_delay,
            'reset_additional': reset_additional,
            'update_variables': update_variables,
            'update_rng': update_rng,
            'update_delay': update_delay,
            'update_global_ops': update_global_ops,
            'stop_condition': stop_condition,
            'host_to_device': host_to_device,
            'device_to_host': device_to_host
        }

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

        pop_desc['custom_func'] = device_local_func
        pop_desc['update'] = update_call
        pop_desc['update_body'] = body
        pop_desc['update_header'] = header
        pop_desc['update_delay'] = """    pop%(id)s.update_delay();\n""" % {'id': pop.id} if pop.max_delay > 1 else ""

        if len(pop.global_operations) > 0:
            pop_desc['gops_update'] = self._update_globalops(pop) % {'id': pop.id}

        pop_desc['host_to_device'] = tabify("pop%(id)s.host_to_device();" % {'id':pop.id}, 1)+"\n"
        pop_desc['device_to_host'] = tabify("pop%(id)s.device_to_host();" % {'id':pop.id}, 1)+"\n"

        return pop_desc

    def reset_computesum(self, pop):
        code = ""

        for target in sorted(pop.neuron_type.description['targets']):
            if pop.neuron_type.type == 'rate':
                code += """
    if ( pop%(id)s._active ) {
        clear_sum <<< __pop%(id)s_nb__, __pop%(id)s_tpb__ >>> ( pop%(id)s.size, pop%(id)s.gpu__sum_%(target)s );
    #ifdef _DEBUG
        auto err = cudaGetLastError();
        if ( err != cudaSuccess ) {
            std::cout << "clear_sum: " << cudaGetErrorString(err) << std::endl;
        }
    #endif
    }
""" % {'id': pop.id, 'target': target}
        return code

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
                if var in pop.neuron_type.description['local']:
                    declare_code += delay_tpl['local']['declare'] % {'var': var, 'type': self._get_attr(pop, var)['ctype']}
                else:
                    raise NotImplementedError
        else:
            # Spiking networks should only exchange spikes
            declare_code += """
    // Delays for spike population
    std::deque< int* > gpu_delayed_spiked;        // contains a set of device pointers
    std::deque< unsigned int* > gpu_delayed_num_events;    // how many events
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
        for var in pop.delayed_variables:
            attr = self._get_attr(pop, var)

            if attr['locality'] == "local":
                attr_dict = {
                    'id': pop.id,
                    'name': attr['name'],
                    'type': attr['ctype'],
                    'delay': pop.max_delay
                }

                init_code += delay_tpl['local']['init'] % attr_dict
                update_code += delay_tpl['local']['update'] % attr_dict
                reset_code += delay_tpl['local']['reset'] % attr_dict
            else:
                raise NotImplementedError

        # Delaying spike events is done differently
        if pop.neuron_type.type == 'spike':
            init_code += """
            gpu_delayed_spiked = std::deque<int*>();
            gpu_delayed_num_events = std::deque<unsigned int*>();
            int *dev_spiked;
            unsigned int *dev_num_events;

            for(int i = 0; i < %(max_delay)s; i++) {

                // events
                cudaMalloc((void**)&dev_spiked, size * sizeof(int));
                gpu_delayed_spiked.push_front(dev_spiked);

                // event counter
                cudaMalloc((void**)&dev_num_events, sizeof(unsigned int));
                gpu_delayed_num_events.push_front(dev_num_events);
            }
            """ % {'max_delay': int(ceil(pop.max_delay/Global.config['dt']))}
            update_code += """
            int* last_spiked = gpu_delayed_spiked.back();
            gpu_delayed_spiked.pop_back();
            gpu_delayed_spiked.push_front(last_spiked);

            unsigned int* last_num_event = gpu_delayed_num_events.back();
            gpu_delayed_num_events.pop_back();
            gpu_delayed_num_events.push_front(last_num_event);

            cudaMemcpy( &num_events, gpu_delayed_num_events.front(), sizeof(unsigned int), cudaMemcpyDeviceToHost);
            gpu_spiked = gpu_delayed_spiked.front();
            """
            reset_code += ""

        update_code = """
        if ( _active ) {
%(code)s
        }""" % {'code': update_code}

        return declare_code, init_code, update_code, reset_code

    def _init_fr(self, pop):
        # TODO:
        return "", ""

    def _gen_kernel_args(self, pop, locality):
        """
        Generate the argument and call statemen for neural variables
        used in equations as well as there dependencies.
        """
        # Gather all variable names
        add_args_header = "%(type)s dt" % {'type':Global.config['precision']}
        add_args_call = "dt"

        deps = []
        # variables
        for var in pop.neuron_type.description['variables']:
            if var['locality'] == locality:
                deps.append(var['name'])
                if 'dependencies' in var.keys():
                    deps += var['dependencies']

        deps = list(set(deps))
        for dep in deps:
            attr_type, attr_dict = self._get_attr_and_type(pop, dep)
            ids = {
                'id': pop.id,
                'name': attr_dict['name'],
                'type': attr_dict['ctype']
            }

            if attr_type == 'attr':
                add_args_header += ", %(type)s* %(name)s" % ids
                add_args_call += ", pop%(id)s.gpu_%(name)s" % ids
            elif attr_type == 'rand':
                add_args_header += ", curandState* %(name)s" % ids
                add_args_call += ", pop%(id)s.gpu_%(name)s" % ids
            else:
                raise NotImplementedError

        return add_args_header, add_args_call

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
            device_code += cpp_func.replace('double '+func['name'], '__device__ double pop%(id)s_%(func)s'%{'id': pop.id, 'func':func['name']})

        return host_code, check_and_apply_pow_fix(device_code)

    def _replace_local_funcs(self, pop, glob_eqs, loc_eqs):
        """
        As the local functions can be occur repeatadly in the same file,
        there are modified with pop[id]_ to unique them. Now we need
        to adjust the call accordingly.
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
        we replace the rand_%(id)s by the corresponding curand... term
        """
        # double precision methods have a postfix
        prec_extension = "" if Global.config['precision'] == "float" else "_double"

        for rd in random_distributions:
            if rd['dist'] == "Uniform":
                term = """( curand_uniform%(postfix)s( &%(rd)s[i] ) * (%(max)s - %(min)s) + %(min)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1]}
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)

                term = """( curand_uniform%(postfix)s( &%(rd)s[0] ) * (%(max)s - %(min)s) + %(min)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1]}
                glob_eqs = glob_eqs.replace(rd['name']+"[0]", term)
            elif rd['dist'] == "Normal":
                term = """( curand_normal%(postfix)s( &%(rd)s[i] ) * %(sigma)s + %(mean)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(",")[0], 'sigma': rd['args'].split(",")[1]}
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)

                term = """( curand_normal%(postfix)s( &%(rd)s[0] ) * %(sigma)s + %(mean)s )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(",")[0], 'sigma': rd['args'].split(",")[1]}
                glob_eqs = glob_eqs.replace(rd['name']+"[0]", term)
            elif rd['dist'] == "LogNormal":
                term = """( curand_log_normal%(postfix)s( &%(rd)s[i], %(mean)s, %(std_dev)s) )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1]}
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)

                term = """( curand_log_normal%(postfix)s( &%(rd)s[0], %(mean)s, %(std_dev)s) )""" % {'postfix': prec_extension, 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1]}
                glob_eqs = glob_eqs.replace(rd['name']+"[0]", term)
            else:
                Global._error("Unsupported random distribution on GPUs: " + rd['dist'])

        return loc_eqs, glob_eqs

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
        raise NotImplementedError

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
            code += global_op_template[op['function']]['call'] % {'id': pop.id, 'type': Global.config['precision'], 'op': op['function'], 'var': op['variable']}

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

            a tuple of three strings, comprising of:

                * body:    kernel implementation
                * header:  kernel prototypes
                * call:    kernel call

        """
        # HD ( 18. Nov. 2016 )
        #
        # In some user-defined cases the host and device side need something to do to
        # in order to realize specific functionality. Yet I simply add a update()
        # call, if update_variables was set.
        if 'update_variables' in pop._specific_template.keys():
            call = """
        // host side update of neurons
        pop%(id)s.update();
""" % {'id': pop.id}
            return "", "", call

        # Is there any variable?
        if len(pop.neuron_type.description['variables']) == 0:
            return "", "", ""

        header = ""
        body = ""
        local_call = ""
        global_call = ""

        # some defaults
        ids = {
            'id': pop.id,
            'local_index': "[i]",
            'global_index': '[0]'
        }

        # parse the equations
        glob_eqs = generate_equation_code(pop.id, pop.neuron_type.description, locality='global', padding=1) % ids
        loc_eqs = generate_equation_code(pop.id, pop.neuron_type.description, locality='local', padding=2) % ids

        # replace pow() for SDK < 6.5
        loc_eqs = check_and_apply_pow_fix(loc_eqs)
        glob_eqs = check_and_apply_pow_fix(glob_eqs)

        # replace local function calls
        if len(pop.neuron_type.description['functions']) > 0:
            glob_eqs, loc_eqs = self._replace_local_funcs(pop, glob_eqs, loc_eqs)

        # replace the random distributions
        loc_eqs, glob_eqs = self._replace_random(loc_eqs, glob_eqs, pop.neuron_type.description['random_distributions'])

        # Global variables
        if glob_eqs.strip() != '':
            add_args_header, add_args_call = self._gen_kernel_args(pop, 'global')

            # global operations
            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': Global.config['precision'],
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s._%(op)s_%(var)s""" % ids

            # finalize code templates
            body += CUDATemplates.population_update_kernel['global']['body'] % {
                'id': pop.id, 'add_args': add_args_header, 'global_eqs':glob_eqs
            }
            header += CUDATemplates.population_update_kernel['global']['header'] % {
                'id': pop.id, 'add_args': add_args_header
            }
            global_call = CUDATemplates.population_update_kernel['global']['call'] % {
                'id': pop.id, 'add_args': add_args_call, 'stream_id': pop.id
            }

        # Local variables
        if loc_eqs.strip() != '':
            add_args_header, add_args_call = self._gen_kernel_args(pop, 'local')

            # targets
            for target in sorted(pop.neuron_type.description['targets']):
                add_args_header += """, %(type)s* _sum_%(target)s""" % {'type': Global.config['precision'], 'target' : target}
                add_args_call += """, pop%(id)s.gpu__sum_%(target)s""" % {'id': pop.id, 'target' : target}

            # global operations
            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': Global.config['precision'],
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s._%(op)s_%(var)s""" % ids

            # finalize code templates
            body += CUDATemplates.population_update_kernel['local']['body'] % {
                'id': pop.id, 'add_args': add_args_header, 'pop_size': pop.size, 'local_eqs': loc_eqs
            }
            header += CUDATemplates.population_update_kernel['local']['header'] % {
                'id': pop.id, 'add_args': add_args_header
            }
            local_call = CUDATemplates.population_update_kernel['local']['call'] % {
                'id': pop.id, 'add_args': add_args_call, 'stream_id': pop.id
            }

        # Call statement consists of two parts
        call = """
    // Updating the local and global variables of population %(id)s
    if ( pop%(id)s._active ) {
        %(global_call)s

        %(local_call)s
    }
""" % {'id':pop.id, 'global_call': global_call, 'local_call': local_call}

        if self._prof_gen:
            call = self._prof_gen.annotate_update_neuron(pop, call)

        return body, header, call

    def _update_spiking_neuron(self, pop):
        """
        Generate the neural update code for GPU devices. We split up the
        calculation into three parts:

            * evolvement of global differential equations
            * evolvement of local differential equations
            * spike gathering

        Return:

            * tuple of three code snippets (body, header, call)
        """
        # Is there any variable?
        if len(pop.neuron_type.description['variables']) == 0:
            return "", "", ""

        header = ""
        body = ""
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
        glob_eqs = generate_equation_code(pop.id, pop.neuron_type.description, locality='global', padding=1) % ids
        loc_eqs = generate_equation_code(pop.id, pop.neuron_type.description, locality='local', padding=2) % ids

        # replace pow() for SDK < 6.5
        loc_eqs = check_and_apply_pow_fix(loc_eqs)
        glob_eqs = check_and_apply_pow_fix(glob_eqs)

        # replace local function calls
        if len(pop.neuron_type.description['functions']) > 0:
            glob_eqs, loc_eqs = self._replace_local_funcs(pop, glob_eqs, loc_eqs)

        # replace the random distributions
        loc_eqs, glob_eqs = self._replace_random(loc_eqs, glob_eqs, pop.neuron_type.description['random_distributions'])

        # Global variables
        if glob_eqs.strip() != '':
            add_args_header, add_args_call = self._gen_kernel_args(pop, 'global')

            # global operations
            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': Global.config['precision'],
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s._%(op)s_%(var)s""" % ids

            # finalize code templates
            body += CUDATemplates.population_update_kernel['global']['body'] % {
                'id': pop.id, 'add_args': add_args_header, 'global_eqs':glob_eqs
            }
            header += CUDATemplates.population_update_kernel['global']['header'] % {
                'id': pop.id, 'add_args': add_args_header
            }
            global_call = CUDATemplates.population_update_kernel['global']['call'] % {
                'id': pop.id, 'add_args': add_args_call, 'stream_id': pop.id
            }

        # Local variables
        if loc_eqs.strip() != '':
            add_args_header, add_args_call = self._gen_kernel_args(pop, 'local')

            # global operations
            for op in pop.global_operations:
                ids = {
                    'id': pop.id,
                    'type': Global.config['precision'],
                    'op': op['function'],
                    'var': op['variable']
                }
                add_args_header += """, %(type)s _%(op)s_%(var)s """ % ids
                add_args_call += """, pop%(id)s._%(op)s_%(var)s""" % ids

            # Is there a refractory period?
            if pop.neuron_type.refractory or pop.refractory:
                # within refractory perid, only conductance variables
                refr_eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local', conductance_only=True, padding=3) % ids

                # 'old' loc_eqs is now only executed ouside refractory period
                loc_eqs = """
        if( refractory_remaining[i] > 0){ // Refractory period
%(eqs)s
            // Decrement the refractory period
            refractory_remaining[i]--;
        } else{
%(loc_eqs)s
        }
""" %  {'eqs': refr_eqs, 'loc_eqs': loc_eqs}

                add_args_header += ", int *refractory, int* refractory_remaining"
                add_args_call += """, pop%(id)s.gpu_refractory, pop%(id)s.gpu_refractory_remaining""" %{'id':pop.id}

            # finalize code templates
            body += CUDATemplates.population_update_kernel['local']['body'] % {
                'id': pop.id, 'add_args': add_args_header, 'pop_size': pop.size, 'local_eqs': loc_eqs
            }
            header += CUDATemplates.population_update_kernel['local']['header'] % {
                'id': pop.id, 'add_args': add_args_header
            }
            local_call = CUDATemplates.population_update_kernel['local']['call'] % {
                'id': pop.id, 'add_args': add_args_call, 'stream_id': pop.id
            }

        # Call statement consists of two parts
        call = """
    // Updating the local and global variables of population %(id)s
    if ( pop%(id)s._active ) {
        %(global_call)s

        %(local_call)s
    }
""" % {'id':pop.id, 'global_call': global_call, 'local_call': local_call}

        if self._prof_gen:
            call = self._prof_gen.annotate_update_neuron(pop, call)

        #
        # Process the spike condition and generate 2nd set of kernels
        #
        cond = pop.neuron_type.description['spike']['spike_cond'] % ids
        reset = ""
        for eq in pop.neuron_type.description['spike']['spike_reset']:
            reset += """
            %(reset)s
""" % {'reset': eq['cpp'] % ids}

        # arguments
        header_args = ""
        call_args = ""

        # gather all attributes required by this kernel
        kernel_deps = []
        for var in pop.neuron_type.description['spike']['spike_cond_dependencies']:
            kernel_deps.append(var)
        for reset_eq in pop.neuron_type.description['spike']['spike_reset']:
            kernel_deps.append(reset_eq['name'])
            for var in reset_eq['dependencies']:
                kernel_deps.append(var)
        kernel_deps = list(set(kernel_deps)) # remove doubled entries

        # generate header, call and body args
        for var in kernel_deps:
            attr = self._get_attr(pop, var)

            header_args += ", "+attr['ctype']+"* " + var
            call_args += ", pop"+str(pop.id)+".gpu_"+var

        # Is there a refractory period?
        if pop.neuron_type.refractory or pop.refractory:
            refrac_inc = "refractory_remaining[i] = refractory[i];"
            header_args += ", int *refractory, int* refractory_remaining"
            call_args += """, pop%(id)s.gpu_refractory, pop%(id)s.gpu_refractory_remaining""" %{'id':pop.id}
        else:
            refrac_inc = ""

        # dependencies of CSR storage_order
        if pop._storage_order == 'pre_to_post':        
            store_spike = """
            pos = atomicAdd ( &spike_count[0], 1);
            spiked[pos] = i;
            last_spike[i] = t;"""
            header_args += ", unsigned int* spike_count"
            call_args += ", pop"+str(pop.id)+".gpu_spike_count"
            spike_gather_decl = """volatile int pos = 0;
    *spike_count = 0;"""
            spike_count = """
// transfer back the spike counter (needed by record)
        cudaMemcpyAsync( &pop%(id)s.spike_count, pop%(id)s.gpu_spike_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[%(stream_id)s]);
    #ifdef _DEBUG
        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
            std::cout << "record_spike_count: " << cudaGetErrorString(err) << std::endl;
    #endif""" %{'id':pop.id, 'stream_id':pop.id}
            spike_count_cpy = """pop%(id)s.spike_count"""%{'id':pop.id}
            spike_count_else = ""
        else:
            store_spike = """
            spiked[i] = 1;
            last_spike[i] = t;"""
            spike_gather_decl = ""
            spike_count = ""
            spike_count_cpy = """pop%(id)s.size"""%{'id':pop.id}
            spike_count_else = """else{
            spiked[i] = 0;
        }"""



        spike_gather = """
        if ( %(cond)s ) {
            %(reset)s

            // store spike event
            %(store_spike)s

            // refractory
            %(refrac_inc)s
        }
        %(spike_count_else)s
""" % {'cond': cond, 'reset': reset, 'refrac_inc': refrac_inc, 'store_spike': store_spike, 'spike_count_else': spike_count_else}

        

        body += CUDATemplates.spike_gather_kernel % {
            'id': pop.id,
            'pop_size': str(pop.size),
            'default': Global.config['precision'] + ' dt, int* spiked, long int* last_spike',
            'args': header_args,
            'decl': spike_gather_decl,
            'spike_gather': spike_gather
        }

        header += CUDATemplates.spike_gather_header % {
            'id': pop.id,
            'default': Global.config['precision'] + ' dt, int* spiked, long int* last_spike',
            'args': header_args
        }

        if pop.max_delay > 1:
            default_args = 'dt, pop%(id)s.gpu_delayed_spiked.front(), pop%(id)s.gpu_last_spike' % {'id': pop.id}
        else: # no_delay
            default_args = 'dt, pop%(id)s.gpu_spiked, pop%(id)s.gpu_last_spike' % {'id': pop.id}

        spike_gather = CUDATemplates.spike_gather_call % {
            'id': pop.id,
            'default': default_args,
            'args': call_args % {'id': pop.id},
            'stream_id': pop.id,
            'spike_count': spike_count,
            'spike_count_cpy': spike_count_cpy
        }

        if self._prof_gen:
            spike_gather = self._prof_gen.annotate_spike_gather(pop, spike_gather)
        call += spike_gather

        return body, header, call

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
                host_device_transfer += self._templates['attribute_transfer']['HtoD_global'] % ids

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

        device_host_transfer += """
    // device to host transfers for %(name)s\n""" % {'name': pop.name}
        for attr in pop.neuron_type.description['variables']:
            ids = {'attr_name': attr['name'], 'type': attr['ctype']}
            if attr['name'] in pop.neuron_type.description['local']:
                device_host_transfer += self._templates['attribute_transfer']['DtoH_local'] % ids
            else:
                device_host_transfer += self._templates['attribute_transfer']['DtoH_global'] % ids
        for attr in pop.neuron_type.description['parameters']:
            if attr['name'] in pop.neuron_type.description['local']:
                ids = {'attr_name': attr['name'], 'type': attr['ctype']}
                device_host_transfer += self._templates['attribute_transfer']['DtoH_local'] % ids

        if 'host_device_transfer' in pop._specific_template.keys():
            host_device_transfer = pop._specific_template['host_device_transfer']
        if 'device_host_transfer' in pop._specific_template.keys():
            device_host_transfer = pop._specific_template['device_host_transfer']

        return host_device_transfer, device_host_transfer
