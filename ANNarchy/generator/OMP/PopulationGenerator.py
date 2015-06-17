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
import PopulationTemplate as PopTemplate
from ANNarchy import SpikeSourceArray
from GlobalOperationTemplate import global_operation_templates_extern as global_op_extern_dict

class PopulationGenerator(object):
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

        if pop._specific:
            # This population has a pre-defined implementation template,
            # e. g. SpikeSourceArrays
            code = pop.generate()

            # we definitely call update methods, if there are not implemented
            # we have some additional calls, we accept them at this point 
            pop_desc['update'] = """    pop%(id)s.update();\n""" % { 'id': pop.id }

            if pop.max_delay > 1:
                pop_desc['delay_update'] = """    pop%(id)s.update_delay();\n""" % { 'id': pop.id }

            if len(pop.neuron_type.description['random_distributions'])>1:
                pop_desc['rng_update'] = """    pop%(id)s.update_rng();\n""" % { 'id': pop.id }

            # Store the complete header definition in a single file
            with open(annarchy_dir+'/generate/pop'+str(pop.id)+'.hpp', 'w') as ofile:
                ofile.write(code)

            return pop_desc

        else:
            base_template = PopTemplate.header_struct[Global.config['paradigm']][pop.neuron_type.type]
            decleration, accessors = self._generate_decl_and_acc(pop)

            # Implementation for init_population, update functions
            # varies dependent on the parallelization paradigm
            if Global.config['paradigm'] == "openmp":
                glops_extern = ""
                for op in pop.global_operations:
                    glops_extern += global_op_extern_dict[op['function']]

                init = self.init_population(pop)

                if pop.max_delay > 1:
                    delay_init, delay_update = self._delay_code(pop)
                    delay_update = delay_update
                    init += delay_init

                update_rng = self.update_random_distributions(pop)
                update_global_ops = self.update_globalops(pop)

                if pop.neuron_type.type == 'rate':
                    update = self._update_rate_neuron_openmp(pop)
                else:
                    update = self.update_spike_neuron(pop)

                code = base_template % { 'id': pop.id,
                                         'gl_ops_extern': glops_extern,
                                         'additional': decleration,
                                         'accessor': accessors,
                                         'init': init,
                                         'update': update,
                                         'update_rng': update_rng,
                                         'update_delay': delay_update if pop.max_delay > 1 else "",
                                         'update_global_ops': update_global_ops
                                        }

                # Store the complete header definition in a single file
                with open(annarchy_dir+'/generate/pop'+str(pop.id)+'.hpp', 'w') as ofile:
                    ofile.write(code)

                # check if we have to add rng, delay calls
                if len(pop.neuron_type.description['variables']) > 0:
                    pop_desc['update'] = """    pop%(id)s.update();\n""" % { 'id': pop.id }

                if len(pop.neuron_type.description['random_distributions']) > 0:
                    pop_desc['rng_update'] = """    pop%(id)s.update_rng();\n""" % { 'id': pop.id }

                if pop.max_delay > 1:
                    pop_desc['delay_update'] = """    pop%(id)s.update_delay();\n""" % { 'id': pop.id }

                if len(pop.global_operations) > 0:
                    pop_desc['gops_update'] = """    pop%(id)s.update_global_ops();\n""" % { 'id': pop.id }

                return pop_desc
            else:
                glops_extern = ""

                init = self.init_population(pop)
                if pop.max_delay > 1:
                    delay_init, delay_update = self._delay_code(pop)
                    delay_update = delay_update.replace("pop"+str(pop.id)+".", "") #TODO: adjust prefixes in parser
                    init += delay_init

                update_rng = self.update_random_distributions(pop).replace("pop"+str(pop.id)+".", "") #TODO: adjust prefixes in parser

                if pop.neuron_type.type == 'rate':
                    body, header, update_call = self._update_rate_neuron_cuda(pop)
                else:
                    Global._error("Spiking neurons on GPUs are currently not supported")
                    exit(0)

                code = base_template % { 'id': pop.id,
                                         'gl_ops_extern': glops_extern,
                                         'additional': decleration,
                                         'accessor': accessors,
                                         'init': init,
                                         'update_rng': update_rng,
                                         'update_delay': delay_update if pop.max_delay > 1 else ""
                                        }

                # Store the complete header definition in a single file
                with open(annarchy_dir+'/generate/pop'+str(pop.id)+'.hpp', 'w') as ofile:
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

    def _generate_decl_and_acc(self, pop):
        # Pick basic template based on neuron type
        attr_template = PopTemplate.attribute_decl[Global.config['paradigm']]
        acc_template = PopTemplate.attribute_acc[Global.config['paradigm']]

        decleration = "" # member declarations
        accessors = "" # export member functions

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            decleration += attr_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
            accessors += acc_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}

        # Variables
        for var in pop.neuron_type.description['variables']:
            decleration += attr_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}
            accessors += acc_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}

        # Arrays for the presynaptic sums
        decleration += """
    // Targets
"""
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets']+pop.targets)):
                decleration += PopTemplate.rate_psp[Global.config['paradigm']]['decl'] % {'target': target}

        # Global operations
        decleration += """
    // Global operations
"""
        for op in pop.global_operations:
            if Global.config['paradigm']=="openmp":
                decleration += """    double _%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}
            else:
                decleration += """    double _%(op)s_%(var)s;
    double *_gpu_%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}
        # Arrays for the random numbers
        decleration += """
    // Random numbers
"""
        for rd in pop.neuron_type.description['random_distributions']:
            if Global.config['paradigm']=="openmp":
                decleration += PopTemplate.cpp_11_rng['decl'] % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}
            else:
                decleration += PopTemplate.cuda_rng['decl'] % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}

        # Delays
        if pop.max_delay > 1:
            decleration += """
    // Delayed variables"""

            if Global.config['paradigm'] == "openmp":
                if pop.neuron_type.type == "rate":
                    for var in pop.delayed_variables:
                        if var in pop.neuron_type.description['local']:
                            decleration += """
    std::deque< std::vector<double> > _delayed_%(var)s; """ % {'var': var}
                        else:
                            decleration += """
    std::deque< double > _delayed_%(var)s; """ % {'var': var}
                else: # Spiking networks should only exchange spikes
                    decleration += """
    // Delays for spike population
    std::deque< std::vector<int> > _delayed_spike;
"""
            else: #CUDA
                if pop.neuron_type.type == "rate":
                    for var in pop.delayed_variables:
                        if var in pop.neuron_type.description['local']:
                            decleration += """
    std::deque< double* > gpu_delayed_%(var)s; // list of gpu arrays""" % {'var': var}
                        else:
                            #TODO:
                            continue
                else:
                    Global._error("synaptic delays for spiking neurons are not implemented yet ...")
                    exit(0)

        # Local functions
        if len(pop.neuron_type.description['functions'])>0:
            decleration += """
    // Local functions
"""
            for func in pop.neuron_type.description['functions']:
                decleration += ' '*4 + func['cpp'] + '\n'

        return decleration, accessors

#######################################################################
############## BODY: initialization codes #############################
#######################################################################
    def init_population(self, pop):
        # active is true by default
        code = """
        size = %(size)s;
        _active = true;
""" % { 'id': pop.id, 'size': pop.size }

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
                    code += PopTemplate.cpp_11_rng['init'] % {'id': pop.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': pop.id}}
                else:
                    code += PopTemplate.cuda_rng['init'] % {'id': pop.id, 'rd_name': rd['name'] }

        # Global operations
        code += self._init_globalops(pop)

        # Targets
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                code += PopTemplate.rate_psp[Global.config['paradigm']]['init'] % {'id': pop.id, 'target': target}

        # Spike event and refractory
        if pop.neuron_type.type == 'spike':
            code += PopTemplate.model_specific_init['spike_event'] % {'id': pop.id}

        return code

    def _init_globalops(self, pop):
        if len(pop.global_operations)==0:
            return ""

        code = "//Initialize global operations\n"
        for op in pop.global_operations:
            if Global.config['paradigm'] == "openmp":
                code += """    _%(op)s_%(var)s = 0.0;
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}
            else:
                code += """    _%(op)s_%(var)s = 0.0;
    cudaMalloc((void**)&_gpu_%(op)s_%(var)s, sizeof(double));
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}

        return code

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
        # initialization
        delay_tpl = PopTemplate.attribute_delayed[Global.config['paradigm']]
        init_code = """
    // Delayed variables
"""
        for var in pop.delayed_variables:
            locality = "local" if var in pop.neuron_type.description['local'] else "global"
            init_code += delay_tpl[locality] % {'delay': pop.max_delay, 'var': var}

        # update
        update_code = ""
        if Global.config['paradigm'] == "openmp":
            for var in pop.delayed_variables:
                update_code += """
        _delayed_%(var)s.push_front(%(var)s);
        _delayed_%(var)s.pop_back();
""" % {'id': pop.id, 'var' : var}

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

        # spike event is handled seperatly
        if pop.neuron_type.type == 'spike':
            if Global.config['paradigm']=="openmp":
                init_code += """
    _delayed_spike = std::deque< std::vector<int> >(%(delay)s, std::vector<int>());""" % {'delay': pop.max_delay}

                update_code += """
            _delayed_spike.push_front(spiked);
            _delayed_spike.pop_back();
"""
            else:
                Global._error("no synaptic delays for spiking synapses on CUDA implemented ...")
                exit(0)

        update_code = """
        if ( _active ) {
%(code)s
        }""" % {'code': update_code }

        return init_code, update_code

#######################################################################
############## BODY: update variables codes ###########################
#######################################################################
    def _update_rate_neuron_openmp(self, pop):
        """
        Generate the code template for neural update step, more precise updating of variables.
        The code comprise of two major parts: global and local update, second one parallelized
        with an openmp for construct, if number of threads is greater than one.
        """
        from ..Utils import generate_equation_code
        code = ""

        # Global variables
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global', padding=2) % {'id': pop.id}
        if eqs.strip() != "":
            code += """
    // Updating the global variables
%(eqs)s
""" % {'eqs': eqs}

        # Local variables, evaluated in parallel
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local', padding=3) % {'id': pop.id}
        omp_code = "#pragma omp parallel for" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""
        code += """
        // Updating the local variables
        %(omp_code)s
        for(int i = 0; i < %(size)s; i++){
%(eqs)s
        }
""" % {'id': pop.id, 'size': pop.size, 'name' : pop.name, 'eqs': eqs, 'omp_code': omp_code}

        # finish code
        return """
    if( _active ) {
%(code)s
    } // active
""" % {'code': code}

    def _update_rate_neuron_cuda(self, pop):
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
        from ..Utils import generate_equation_code

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
        for target in pop.neuron_type.description['targets']:
            tar += """, double* _sum_%(target)s""" % {'target' : target}

        #Global variables
        glob_eqs = ""
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global') % {'id': pop.id}
        if eqs.strip() != "":
            glob_eqs = """
    if ( threadIdx.x == 0)
    {
%(eqs)s
    }
""" % {'id': pop.id, 'eqs': eqs }
            glob_eqs = glob_eqs.replace("pop"+str(pop.id)+".", "")

        # Local variables
        loc_eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local') % {'id': pop.id}

        # we replace the rand_%(id)s by the corresponding curand... term
        for rd in pop.neuron_type.description['random_distributions']:
            if rd['dist'] == "Uniform":
                term = """curand_uniform_double( &%(rd)s[i]) * (%(max)s - %(min)s) + %(min)s""" % { 'rd': rd['name'], 'min': rd['args'].split(',')[0], 'max': rd['args'].split(',')[1] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
            elif rd['dist'] == "Normal":
                term = """curand_normal_double( &%(rd)s[i])""" % { 'rd': rd['name'] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
            elif rd['dist'] == "LogNormal":
                term = """curand_log_normal_double( &%(rd)s[i], %(mean)s, %(std_dev)s)""" % { 'rd': rd['name'], 'mean': rd['args'].split(',')[0], 'std_dev': rd['args'].split(',')[1] };
                loc_eqs = loc_eqs.replace(rd['name']+"[i]", term)
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
        for target in pop.neuron_type.description['targets']:
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

    def update_spike_neuron(self, pop):
        # Neural update
        from ..Utils import generate_equation_code

        # Is there a refractory period?
        if pop.neuron_type.refractory or pop.refractory:
            eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local', conductance_only=True, padding=4) % {'id': pop.id}
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
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global', padding=2) % {'id': pop.id}
        if eqs.strip() != "":
            global_code = eqs
        else:
            global_code = ""

        # OMP code
        omp_code = "#pragma omp parallel for" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""
        omp_critical_code = "#pragma omp critical" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""

        # Local variables, evaluated in parallel
        code += generate_equation_code(pop.id, pop.neuron_type.description, 'local', padding=3) % {'id': pop.id}

        # if profiling enabled, annotate with profiling code
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            pGen = ProfileGenerator(Global._network[0]['populations'], Global._network[0]['projections'])
            code = pGen.annotate_update_neuron_omp(code)

        # Process the condition
        cond =  pop.neuron_type.description['spike']['spike_cond'] % {'id': pop.id}

        # reset equations
        reset = ""
        for eq in pop.neuron_type.description['spike']['spike_reset']:
            reset += """
                %(reset)s
""" % {'reset': eq['cpp'] % {'id': pop.id}}

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
            }
"""% {  'id': pop.id, 'name': pop.name, 'size': pop.size, 
        'condition' : cond, 'reset': reset, 
        'refrac_inc': refrac_inc,
        'omp_critical_code': omp_critical_code} 

        # if profiling enabled, annotate with profiling code
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            pGen = ProfileGenerator(Global._network[0]['populations'], Global._network[0]['projections'])
            spike_gather = pGen.annotate_spike_propagation_omp(spike_gather)

        code += spike_gather

        # finish code
        return """
    if( _active ) {
        spiked.clear();
%(global_code)s
        %(omp_code)s
        for(int i = 0; i < %(size)s; i++){
%(code)s
        }
    } // active
""" % {'id': pop.id, 'size': pop.size, 'name': pop.name, 'code': code, 'global_code': global_code, 'omp_code': omp_code }

    def reset_computesum(self, pop):
        code = ""
        for target in pop.targets:
            code += """
    if (pop%(id)s._active)
        memset( pop%(id)s._sum_%(target)s.data(), 0.0, pop%(id)s._sum_%(target)s.size() * sizeof(double));
""" % {'id': pop.id, 'target': target}
        return code

    def update_globalops(self, pop):
        """
        Update of global functions is a call of pre-implemented
        functions defined in GlobalOperationTemplate. In case of
        OpenMP this calls will take place in the population header.
        In case of CUDA the call semantic will be placed in ANNarchy.cu
        file.
        """
        code = ""

        if Global.config['paradigm'] == "openmp":
            for op in pop.global_operations:
                code += """            pop%(id)s._%(op)s_%(var)s = %(op)s_value(pop%(id)s.%(var)s.data(), pop%(id)s.size);
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}
        else:
            from .GlobalOperationTemplate import global_operation_templates_cuda as template
            for op in pop.global_operations:
                code += template[op['function']]['call'] % { 'id': pop.id, 'op': op['function'], 'var': op['variable'] }

        return code

    def stop_condition(self, pop):
        if not pop.stop_condition: # no stop condition has been defined
            return """
                case %(id)s:
                    pop_stop = false;
                    break;
""" % {'id': pop.id}

        pop.neuron_type.description['stop_condition'] = {'eq': pop.stop_condition}

        from ANNarchy.parser.Extraction import extract_stop_condition
        from ANNarchy.parser.SingleAnalysis import pattern_omp, pattern_cuda

        # Find the paradigm OMP or CUDA
        if Global.config['paradigm'] == 'cuda':
            pattern = pattern_cuda
        else:
            pattern = pattern_omp

        extract_stop_condition(pop.neuron_type.description, pattern)

        if pop.neuron_type.description['stop_condition']['type'] == 'any':
            stop_code = """
                    pop_stop = false;
                    for(int i=0; i<pop%(id)s.size; i++)
                    {
                        if(%(condition)s)
                            pop_stop = true;
                    }
    """ % {'id': pop.id, 'condition': pop.neuron_type.description['stop_condition']['cpp']% {'id': pop.id}}
        else:
            stop_code = """
                    pop_stop = true;
                    for(int i=0; i<pop%(id)s.size; i++)
                    {
                        if(!(%(condition)s))
                            pop_stop = false;
                    }
    """ % {'id': pop.id, 'condition': pop.neuron_type.description['stop_condition']['cpp']% {'id': pop.id}}

        return """
                case %(id)s:
%(stop_code)s
                    break;
""" % {'id': pop.id, 'stop_code': stop_code}

    def update_random_distributions(self, pop):
        code = ""
        for rd in pop.neuron_type.description['random_distributions']:
            if Global.config['paradigm']=="openmp":
                code += PopTemplate.cpp_11_rng['update'] % {'id': pop.id, 'rd_name': rd['name']}
            else:
                code += PopTemplate.cuda_rng['update']

        return code

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
