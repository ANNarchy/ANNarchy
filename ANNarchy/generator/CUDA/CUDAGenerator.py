import ANNarchy.core.Global as Global
from ANNarchy.core.PopulationView import PopulationView
from .PopulationGenerator import PopulationGenerator
from .ProjectionGenerator import ProjectionGenerator

import numpy as np

# TODO: INTERFACE
#
#    as first step I only moved the current implementation to the extra generator classes, in a second step, the code
#    generation could be refined ...
#
#    in general, the ANNarchy.cpp would only contain the the call method instead of the full buisness logic,
#    as a consequence it could be enough to forward the calls to the update_neuron, update_synapse etc. functions.
#    The classes Population-/ProjectionGenerator could be responsible alone for the creation of cuANNarchy.cu / cuANNarchy.h
class CUDAGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

        self.popgen = PopulationGenerator()
        self.projgen = ProjectionGenerator()
        
    def generate(self):

        # Propagte the global operations needed by the projections to the corresponding populations.
        self.propagate_global_ops()

        # Generate header code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchy.h', 'w') as ofile:
            ofile.write(self.generate_header())
            
        # Generate cpp and cuda code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchy.cpp', 'w') as ofile:
            ofile.write(self.generate_body())

        # Generate cython code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchyCore.pyx', 'w') as ofile:
            ofile.write(self.generate_pyx())

    def propagate_global_ops(self):

        # Analyse the populations
        for pop in self.populations:
            pop.global_operations = pop.neuron_type.description['global_operations']

        # Propagate the global operations from the projections to the populations
        for proj in self.projections:
            for op in proj.synapse.description['pre_global_operations']:
                if isinstance(proj.pre, PopulationView):
                    if not op in proj.pre.population.global_operations:
                        proj.pre.population.global_operations.append(op)
                else:
                    if not op in proj.pre.global_operations:
                        proj.pre.global_operations.append(op)

            for op in  proj.synapse.description['post_global_operations']:
                if isinstance(proj.post, PopulationView):
                    if not op in proj.post.population.global_operations:
                        proj.post.population.global_operations.append(op)
                else:
                    if not op in proj.post.global_operations:
                        proj.post.global_operations.append(op)

        # Make sure the operations are declared only once
        for pop in self.populations:
            pop.global_operations = list(np.unique(np.array(pop.global_operations)))



#######################################################################
############## HEADER #################################################
#######################################################################
    def generate_header(self):

        # struct declaration for each population
        pop_struct, pop_ptr = self.header_struct_pop()

        # struct declaration for each projection
        proj_struct, proj_ptr = self.header_struct_proj()

        from .HeaderTemplate import header_template
        return header_template % {
            'pop_struct': pop_struct,
            'proj_struct': proj_struct,
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
        }

    def header_struct_pop(self):
        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""
        for pop in self.populations:
            # Is it a specific population?
            if pop.generator['cuda']['header_pop_struct']:
                Global._error("Customized populations are not usable on CUDA yet.")
                continue

            if pop.neuron_type.type == 'spike':
                Global._error("Spiking populations are not usable on CUDA yet.")
                continue

            # Header struct
            pop_struct += self.popgen.header_struct(pop)      
            # Extern pointer
            pop_ptr += """extern PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        return pop_struct, pop_ptr

    def header_struct_proj(self):
        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in self.projections:
            # Header struct
            proj_struct += self.projgen.header_struct(proj)
            # Extern pointer
            proj_ptr += """extern ProjStruct%(id_proj)s proj%(id_proj)s;
"""% {'id_proj': proj.id}

        return proj_struct, proj_ptr
        


#######################################################################
############## BODY ###################################################
#######################################################################
    def generate_body(self):
        """
        generate the ANNarchy.cpp file containing initialization codes, host_to_device- and device_to_host
        transfers and kernel call entities next to the files: cuANNarchy.cu (kernel implementation) and
        cuANNarchy.h (kernel prototypes).
        """
        # struct declaration for each population
        pop_ptr = ""
        for pop in self.populations:
            # Declaration of the structure
            pop_ptr += """PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        # struct declaration for each projection
        proj_ptr = ""
        for proj in self.projections:
            # Declaration of the structure
            proj_ptr += """ProjStruct%(id)s proj%(id)s;
"""% {'id': proj.id}

        # Code for the global operations
        glob_ops_header, glob_ops_body = self.body_def_glops()

        # Compute presynaptic sums
        compute_sums_body, compute_sums_header, compute_sums_call = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()

        # Initialize device ptr
        device_init = self.body_init_device()

        # host to device and device to host transfers
        host_device_transfer, device_host_transfer = self.body_memory_transfers()

        # Initialize delayed arrays
        delay_init = self.body_init_delay()

        # Initialize spike arrays
        spike_init = "" #TODO: self.body_init_spike()

        # Initialize projections
        projection_init = self.body_init_projection()

        # Initialize global operations
        globalops_init = self.body_init_globalops()
        
        # call cuda-kernel for updating the neural variables
        update_neuron_body, update_neuron_header, update_neuron_call = self.body_update_neuron()

        # Enque delayed outputs
        delay_code = self.body_delay_neuron()

        # Global operations
        update_globalops = self.body_update_globalops()

        # Equations for the synaptic variables
        update_synapse_body, update_synapse_header, update_synapse_call = self.body_update_synapse()

        # Equations for the synaptic variables
        post_event = ""#TODO: self.body_postevent_proj()

        # Record
        record = "" #TODO: self.body_record()

        # determine number of threads per kernel
        # and concurrent kernel execution
        threads_per_kernel, stream_setup = self.body_kernel_config()

        # Custom functions
        custom_func = self.body_custom_functions()

        # Generate cuda header code for the analysed pops and projs
        from .cuBodyTemplate import cu_header_template
        cuda_header = cu_header_template % {
            'neuron': update_neuron_header,
            'compute_sum': compute_sums_header,
            'synapse': update_synapse_header,
            'glob_ops': glob_ops_header
        }
        with open(Global.annarchy_dir+'/generate/cuANNarchy.h', 'w') as ofile:
            ofile.write(cuda_header)

        # Generate cuda code for the analysed pops and projs
        from .cuBodyTemplate import cu_body_template
        cuda_body = cu_body_template % {
            'kernel_config': threads_per_kernel,
            'pop_kernel': update_neuron_body,
            'psp_kernel': compute_sums_body,
            'syn_kernel': update_synapse_body,
            'glob_ops_kernel': glob_ops_body,
            'custom_func': custom_func
        }
        with open(Global.annarchy_dir+'/generate/cuANNarchy.cu', 'w') as ofile:
            ofile.write(cuda_body)

        # Generate cpp code for the analysed pops and projs
        from .BodyTemplate import body_template
        return body_template % {
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'stream_setup': stream_setup,
            'compute_sums' : compute_sums_call,
            'update_neuron' : update_neuron_call,
            'update_globalops' : update_globalops,
            'update_synapse' : update_synapse_call,
            'host_device_transfer': host_device_transfer,
            'device_host_transfer': device_host_transfer,
            'device_init': device_init,
            'random_dist_init' : rd_init_code,
            'delay_init' : delay_init,
            'delay_code' : delay_code,
            'spike_init' : spike_init,
            'projection_init' : projection_init,
            'globalops_init' : globalops_init,
            'post_event' : post_event,
            'record' : record
        }

    def body_update_neuron(self):
        # TODO: INTERFACE
        header = ""
        body = ""
        call = ""

        for pop in self.populations:
            pop_header, pop_body, pop_call = self.popgen.update_neuron(pop)
            
            header += pop_header
            body += pop_body
            call += pop_call

        return header, body, call

    def body_delay_neuron(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.delay_code(pop)
        return code

    def body_computesum_proj(self):
        # TODO: INTERFACE
        header = ""
        body = ""
        call = ""

        for proj in self.projections:
            # Call the right generator depending on type
            if proj.synapse.type == 'rate':
                b, h, c = self.projgen.computesum_rate(proj)
            else:
                b, h, c = self.projgen.computesum_spiking(proj)

            header += h
            body += b
            call += c

        return body, header, call

    def body_update_synapse(self):
        # TODO: INTERFACE
        header = ""
        body = ""
        call = ""

        for proj in self.projections:
            b, h, c = self.projgen.update_synapse(proj)

            header += h
            body += b
            call += c

        return body, header, call

    def body_structural_plasticity(self):
        # Pruning if any
        pruning=""
        if Global.config['structural_plasticity'] :
            for proj in self.projections:
                if 'pruning' in proj.synapse.description.keys():
                    pruning += self.projgen.pruning(proj)

        return pruning

    def body_init_randomdistributions(self):
        code = """
    // Initialize RNG states
"""
        for pop in self.populations:
            code += self.popgen.init_random_distributions(pop)

        for proj in self.projections:
            code += self.projgen.init_random_distributions(proj)

        return code

    def body_init_globalops(self):
        code = """
    // Initialize global operations
"""
        for pop in self.populations:
            code += self.popgen.init_globalops(pop)

        return code

    def body_def_glops(self):
        ops = []
        for pop in self.populations:
            for op in pop.global_operations:
                ops.append( op['function'] )

        if ops == []:
            return "", ""

        from .GlobalOperationTemplate import global_operation_templates
        header = ""
        body = ""

        for op in list(set(ops)):
            header += global_operation_templates[op]['header']
            body += global_operation_templates[op]['body']

        return header, body

    def body_init_delay(self):
        code = ""
        for pop in self.populations:
            if pop.max_delay > 1: # no need to generate the code otherwise
                code += self.popgen.init_delay(pop)

        return code

    def body_init_population(self):
        code = """
    // Initialize populations
"""
        for pop in self.populations:
            code += self.popgen.init_population(pop)

        return code

    def body_init_projection(self):
        code = """
    // Initialize projections
"""
        for proj in self.projections:
            code += self.projgen.init_projection(proj)

        return code
        
    def body_update_globalops(self):
        code = ""
        for pop in self.populations:
            # Is it a specific population?
            if pop.generator['cuda']['body_update_globalops']:
                code += pop.generator['cuda']['body_update_globalops'] %{ 'id': pop.id}
                continue
            code += self.popgen.update_globalops(pop)

        if code.strip() != '':
            code = """
    double *tmp;
    cudaMalloc((void**)&tmp, sizeof(double));

%(code)s
    cudaFree(tmp);
""" % { 'code': code}

        return code

    def body_record(self):
        code = ""
        # Populations
        for pop in self.populations:
           code += self.popgen.record(pop)

        # Projections
        for proj in self.projections:
           code += self.projgen.record(proj)

        return code

    def body_run_until(self):
        code = ""
        for pop in self.populations:
            if not pop.stop_condition: # no stop condition has been defined
                code += """
                case %(id)s: 
                    pop_stop = false;
                    break;
""" % {'id': pop.id}
            else:
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

                code += """
                case %(id)s:
%(stop_code)s
                    break;
""" % {'id': pop.id, 'stop_code': stop_code}
        return code

    def body_custom_functions(self):
        """
        ATTENTION: the same as OMPGenerator.header_custom_func
        """
        if len(Global._functions) == 0:
            return ""

        Global._error("Not implemented yet: custom functions for GPGPU kernel ...")
        return code

#===============================================================================
#         code = ""
#         from ANNarchy.parser.Extraction import extract_functions
#         for func in Global._functions:
#             code +=  extract_functions(func, local_global=True)[0]['cpp'] + '\n'
#
#         return code
#===============================================================================



#######################################################################
############## PYX ####################################################
#######################################################################
    def generate_pyx(self):
        # struct declaration for each population
        pop_struct, pop_ptr = self.pyx_struct_pop()

        # struct declaration for each projection
        proj_struct, proj_ptr = self.pyx_struct_proj()

        # Cython wrappers for the populations
        pop_class = self.pyx_wrapper_pop()

        # Cython wrappers for the projections
        proj_class = self.pyx_wrapper_proj()


        from .PyxTemplate import pyx_template
        return pyx_template % {
            'pop_struct': pop_struct, 'pop_ptr': pop_ptr,
            'proj_struct': proj_struct, 'proj_ptr': proj_ptr,
            'pop_class' : pop_class, 'proj_class': proj_class
        }

    def pyx_struct_pop(self):
        pop_struct = ""
        pop_ptr = ""
        for pop in self.populations:
            # Is it a specific population?
            if pop.generator['cuda']['pyx_pop_struct']:
                pop_struct += pop.generator['cuda']['pyx_pop_struct'] %{'id': pop.id}
                pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {'id': pop.id}
                continue

            code = """
    # Population %(id)s (%(name)s)
    cdef struct PopStruct%(id)s :
        int size
        bool _active
"""            

            # Parameters
            for var in pop.neuron_type.description['parameters']:
                if var['name'] in pop.neuron_type.description['local']:
                    code += """
        # Local parameter %(name)s
        vector[%(type)s] %(name)s 
        bool %(name)s_dirty
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
        # Global parameter %(name)s
        %(type)s  %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in pop.neuron_type.description['variables']:
                if var['name'] in pop.neuron_type.description['local']:
                    code += """
        # Local variable %(name)s
        vector[%(type)s] %(name)s 
        bool %(name)s_dirty
        #vector[vector[%(type)s]] recorded_%(name)s 
        #bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
        # Global variable %(name)s
        %(type)s  %(name)s 
        #vector[%(type)s] recorded_%(name)s
        #bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Arrays for the presynaptic sums of rate-coded neurons
            if pop.neuron_type.type == 'rate':
                code += """
        # Targets
"""
                for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                    code += """        vector[double] _sum_%(target)s
""" % {'target' : target}

            # Finalize the code
            pop_struct += code % {'id': pop.id, 'name': pop.name}

            # Population instance
            pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {
    'id': pop.id,
}
        return pop_struct, pop_ptr

    def pyx_struct_proj(self):
        proj_struct = ""
        proj_ptr = ""
        for proj in self.projections:
            # Is it a specific projection?
            if proj.generator['cuda']['pyx_proj_struct']:
                proj_struct += proj.generator['cuda']['pyx_proj_struct']
                proj_ptr += """
    ProjStruct%(id_proj)s proj%(id_proj)s"""% {'id_proj': proj.id}
                continue

            code = """
    cdef struct ProjStruct%(id_proj)s :
        int size
        bool _learning
        vector[int] post_rank
        vector[vector[int]] pre_rank
"""         

            # Exact integration
            has_exact = False
            for var in proj.synapse.description['variables']:
                if var['method'] == 'exact':
                    has_exact = True
            if has_exact:
                code += """
        vector[vector[long]] _last_event
"""% {'id': proj.id}

            # Delays
            if proj.max_delay > 1 and proj.uniform_delay == -1:
                code +="""
        vector[vector[int]] delay
"""
            # Parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
        # Local parameter %(name)s
        vector[vector[%(type)s]] %(name)s
        bool %(name)s_dirty
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global parameter %(name)s
        vector[%(type)s]  %(name)s
        bool %(name)s_dirty 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
        # Local variable %(name)s
        vector[vector[%(type)s]] %(name)s 
        #vector[vector[vector[%(type)s]]] recorded_%(name)s 
        #vector[int] record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global variable %(name)s
        vector[%(type)s]  %(name)s 
        #vector[vector[%(type)s]] recorded_%(name)s
        #vector[int] record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Structural plasticity
            if Global.config['structural_plasticity']:
                Global._error("Structural plasticity is not supported yet on GPUs")

            # Finalize the code
            proj_struct += code % {'id_proj': proj.id}

            # Population instance
            proj_ptr += """
    ProjStruct%(id_proj)s proj%(id_proj)s"""% {
    'id_proj': proj.id,
}
        return proj_struct, proj_ptr

    def pyx_wrapper_pop(self):
        # Cython wrappers for the populations
        code = ""
        for pop in self.populations:
            # Is it a specific population?
            if pop.generator['cuda']['pyx_pop_class']:
                code += pop.generator['cuda']['pyx_pop_class'] %{'id': pop.id}
                continue

            # Init
            code += """
# Population %(id)s (%(name)s)
cdef class pop%(id)s_wrapper :

    def __cinit__(self, size):
        pop%(id)s.size = size"""% {'id': pop.id, 'name': pop.name}

            # Parameters
            for var in pop.neuron_type.description['parameters']:
                init = 0.0 if var['ctype'] == 'double' else 0
                if var['name'] in pop.neuron_type.description['local']:                    
                    code += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)
        pop%(id)s.%(name)s_dirty = True""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
        
                else: # global
                    code += """
        pop%(id)s.%(name)s = %(init)s""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Variables
            for var in pop.neuron_type.description['variables']:
                init = 0.0 if var['ctype'] == 'double' else 0
                if var['name'] in pop.neuron_type.description['local']:
                    code += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)
        #pop%(id)s.recorded_%(name)s = vector[vector[%(type)s]](0, vector[%(type)s](0,%(init)s))
        #pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
                else: # global
                    code += """
        pop%(id)s.%(name)s = %(init)s
        pop%(id)s.%(name)s_dirty = True
        #pop%(id)s.recorded_%(name)s = vector[%(type)s](0, %(init)s)
        #pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Targets
            if pop.neuron_type.type == 'rate':
                for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                    code += """
        pop%(id)s._sum_%(target)s = vector[double](size, 0.0)""" %{'id': pop.id, 'target': target}

            # Size property
            code += """

    property size:
        def __get__(self):
            return pop%(id)s.size
""" % {'id': pop.id}

            # Activate population
            code += """
    def activate(self, bool val):
        pop%(id)s._active = val
""" % {'id': pop.id}

            # Parameters
            for var in pop.neuron_type.description['parameters']:
                if var['name'] in pop.neuron_type.description['local']:
                    code += """
    # Local parameter %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s_dirty = True
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
    # Global parameter %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            # Variables
            for var in pop.neuron_type.description['variables']:
                if var['name'] in pop.neuron_type.description['local']:
                    code += """
    # Local variable %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s_dirty = True
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.%(name)s_dirty = True
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
    # Global variable %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

        return code

    def pyx_wrapper_proj(self):
        # Cython wrappers for the projections
        code = ""
        for proj in self.projections:
            # Is it a specific projection?
            if proj.generator['cuda']['pyx_proj_class']:
                code += proj.generator['cuda']['pyx_proj_class']
                continue

            # Init
            code += """
cdef class proj%(id)s_wrapper :

    def __cinit__(self, synapses):

        cdef CSR syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()

        proj%(id)s.size = size
        proj%(id)s.post_rank = syn.post_rank
        proj%(id)s.pre_rank = syn.pre_rank
        proj%(id)s.w = syn.w

        proj%(id)s._learning = True
"""% {'id': proj.id}

            # Exact integration
            has_exact = False
            for var in proj.synapse.description['variables']:
                if var['method'] == 'exact':
                    has_exact = True
            if has_exact:
                code += """
        proj%(id)s._last_event = vector[vector[long]](nb_post, vector[long]())
        for n in range(nb_post):
            proj%(id)s._last_event[n] = vector[long](proj%(id)s.pre_rank[n].size(), -10000)
"""% {'id': proj.id}

            # Delays
            if proj.max_delay > 1 and proj.uniform_delay == -1:
                code +="""
        proj%(id)s.delay = syn.delay
"""% {'id': proj.id}

            # Initialize parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] == 'w':
                    continue
                if var['name'] in proj.synapse.description['local']:
                    init = 0.0 if var['ctype'] == 'double' else 0
                    code += """
        proj%(id)s.%(name)s = vector[vector[%(type)s]](nb_post, vector[%(type)s]())
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
                else:
                    init = 0.0 if var['ctype'] == 'double' else 0
                    code += """
        proj%(id)s.%(name)s = vector[%(type)s](nb_post, %(init)s)
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Initialize variables
            for var in proj.synapse.description['variables']:
                if var['name'] == 'w':
                    continue
                if var['name'] in proj.synapse.description['local']:
                    init = 0.0 if var['ctype'] == 'double' else 0
                    code += """
        proj%(id)s.%(name)s = vector[vector[%(type)s]](nb_post, vector[%(type)s]())
        #proj%(id)s.record_%(name)s = vector[int]()
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
                else:
                    init = 0.0 if var['ctype'] == 'double' else 0
                    code += """
        proj%(id)s.%(name)s = vector[%(type)s](nb_post, %(init)s)
        #proj%(id)s.record_%(name)s = vector[int]()
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Size property
            code += """

    property size:
        def __get__(self):
            return proj%(id)s.size

    def nb_synapses(self, int n):
        return proj%(id)s.pre_rank[n].size()

    def _set_learning(self, bool l):
        proj%(id)s._learning = l

    def post_rank(self):
        return proj%(id)s.post_rank
    def pre_rank(self, int n):
        return proj%(id)s.pre_rank[n]
""" % {'id': proj.id}

            # Delays
            if proj.max_delay > 1 and proj.uniform_delay == -1:
                code +="""
    def get_delay(self):
        return proj%(id)s.delay
    def set_delay(self, value):
        proj%(id)s.delay = value
"""% {'id': proj.id}

            # Parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    # Local parameter %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s_dirty = True
        proj%(id)s.%(name)s = value
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.%(name)s_dirty = True
        proj%(id)s.%(name)s[rank] = value
    def get_synapse_%(name)s(self, int rank_post, int rank_pre):
        return proj%(id)s.%(name)s[rank_post][rank_pre]
    def set_synapse_%(name)s(self, int rank_post, int rank_pre, %(type)s value):
        proj%(id)s.%(name)s_dirty = True
        proj%(id)s.%(name)s[rank_post][rank_pre] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

                elif var['name'] in proj.synapse.description['global']:
                    code += """
    # Global parameter %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s_dirty = True
        proj%(id)s.%(name)s = value
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, %(type)s value):
        proj%(id)s.%(name)s_dirty = True
        proj%(id)s.%(name)s[rank] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    # Local variable %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s = value
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.%(name)s[rank] = value
    def get_synapse_%(name)s(self, int rank_post, int rank_pre):
        return proj%(id)s.%(name)s[rank_post][rank_pre]
    def set_synapse_%(name)s(self, int rank_post, int rank_pre, %(type)s value):
        proj%(id)s.%(name)s[rank_post][rank_pre] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

                elif var['name'] in proj.synapse.description['global']:
                    code += """
    # Global variable %(name)s
    def get_%(name)s(self):
        return proj%(id)s.%(name)s
    def set_%(name)s(self, value):
        proj%(id)s.%(name)s = value
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, %(type)s value):
        proj%(id)s.%(name)s[rank] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}

             # Structural plasticity
            if Global.config['structural_plasticity']:
                Global._error('structural plasticity is not supported yet on CUDA ...')

        return code

#######################################################################
############## HOST - DEVICE ##########################################
#######################################################################
    def body_memory_transfers(self):
        host_device_transfer = ""
        device_host_transfer = ""

        # transfers for populations
        for pop in self.populations:
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

        # transfers for projections
        for proj in self.projections:
            host_device_transfer += """\n    // host to device transfers for proj%(id)s\n""" % { 'id': proj.id }
            for attr in proj.synapse.description['parameters']+proj.synapse.description['variables']:
                if attr['name'] in proj.synapse.description['local']:
                    host_device_transfer += """
        // %(name)s: local
        if ( proj%(id)s.%(name)s_dirty )
        {
            auto flat_proj%(id)s_%(name)s = flattenArray<double>(proj%(id)s.%(name)s);
            cudaMemcpy(proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.data(), flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s), cudaMemcpyHostToDevice);
            flat_proj%(id)s_%(name)s.clear();
        }
""" % { 'id': proj.id, 'name': attr['name'], 'type': attr['ctype'] }
                else:
                    host_device_transfer += """
        // %(name)s: global
        if ( proj%(id)s.%(name)s_dirty )
        {
            cudaMemcpy(proj%(id)s.gpu_%(name)s, proj%(id)s.%(name)s.data(), pop%(post)s.size * sizeof(%(type)s), cudaMemcpyHostToDevice);
        }
""" % { 'id': proj.id, 'post': proj.post.id, 'name': attr['name'], 'type': attr['ctype'] }

            device_host_transfer += """
    // device to host transfers for proj%(id)s\n""" % { 'id': proj.id }
            for attr in proj.synapse.description['parameters']+proj.synapse.description['variables']:
                if attr['name'] in proj.synapse.description['local']:
                    device_host_transfer += """
            // %(name)s: local
            std::vector<%(type)s> flat_proj%(id)s_%(name)s = std::vector<%(type)s>(proj%(id)s.overallSynapses, 0);
            cudaMemcpy(flat_proj%(id)s_%(name)s.data(), proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s), cudaMemcpyDeviceToHost);
            proj%(id)s.%(name)s = deFlattenArray<%(type)s>(flat_proj%(id)s_%(name)s, proj%(id)s.flat_idx);
            flat_proj%(id)s_%(name)s.clear();
""" % { 'id': proj.id, 'name': attr['name'], 'type': attr['ctype'] }
                else:
                    device_host_transfer += """
            // %(name)s: global
            cudaMemcpy( proj%(id)s.%(name)s.data(), proj%(id)s.gpu_%(name)s, pop%(post)s.size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
""" % { 'id': proj.id, 'post': proj.post.id, 'name': attr['name'], 'type': attr['ctype'] }

        return host_device_transfer, device_host_transfer

    def body_init_device(self):
        code = ""

        dev_id = 0 # default cuda device
        if 'device' in Global.cuda_config.keys():
            dev_id = Global.cuda_config['device']

        code += """
    // set active cuda device
    auto status = cudaSetDevice(%(id)s);
    if ( status != cudaSuccess )
        std::cerr << "Error on setting cuda device ... " << std::endl;

    // initialize cuda-api
    cudaFree(0);
""" % { 'id': dev_id }

        for pop in self.populations:
            code += """\n\t// Initialize device memory for %(pop_name)s\n""" % { 'pop_name': pop.name }
            for attr in pop.neuron_type.description['parameters']+pop.neuron_type.description['variables']:
                if attr['name'] in pop.neuron_type.description['local']:
                    code += """\tcudaMalloc((void**)&pop%(id)s.gpu_%(attr_name)s, pop%(id)s.size * sizeof(%(type)s));
        pop%(id)s.%(attr_name)s_dirty = true;
""" % { 'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype'] }
            for target in pop.neuron_type.description['targets']:
                code += """\tcudaMalloc((void**)&pop%(id)s.gpu_sum_%(target)s, pop%(id)s.size * sizeof(double));
""" % { 'id': pop.id, 'target': target }

        for proj in self.projections:
            from .cuBodyTemplate import proj_basic_data

            # basic variables: post_rank, nb_synapses, off_synapses, pre_rank
            code += proj_basic_data % { 'id': proj.id }

            # other variables, parameters
            for attr in proj.synapse.description['parameters']+proj.synapse.description['variables']:
                if attr['name'] in proj.synapse.description['local']:
                    code += """
        // %(name)s
        auto flat_proj%(id)s_%(name)s = flattenArray<double>(proj%(id)s.%(name)s);
        cudaMalloc((void**)&proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s));
        cudaMemcpy(proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.data(), flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s), cudaMemcpyHostToDevice);
        flat_proj%(id)s_%(name)s.clear();
        proj%(id)s.%(name)s_dirty = false;
""" % { 'id': proj.id, 'name': attr['name'], 'type': attr['ctype'] }
                else:
                    code += """
        // %(name)s
        cudaMalloc((void**)&proj%(id)s.gpu_%(name)s, pop%(post)s.size * sizeof(%(type)s));
        cudaMemcpy(proj%(id)s.gpu_%(name)s, proj%(id)s.%(name)s.data(), pop%(post)s.size * sizeof(%(type)s), cudaMemcpyHostToDevice);
        proj%(id)s.%(name)s_dirty = false;
""" % { 'id': proj.id, 'post': proj.post.id, 'name': attr['name'], 'type': attr['ctype'] }

        return code

    def body_kernel_config(self):
        cu_config = Global.cuda_config

        code = "// Population config\n"
        for pop in self.populations:
            num_threads = 32
            if pop in cu_config.keys():
                num_threads = cu_config[pop]['num_threads']

            code+= """#define pop%(id)s %(nr)s\n""" % { 'id': pop.id, 'nr': num_threads }

        code += "\n// Population config\n"
        for proj in self.projections:
            num_threads = 192
            if proj in cu_config.keys():
                num_threads = cu_config[proj]['num_threads']

            code+= """#define pop%(pre)s_pop%(post)s_%(target)s %(nr)s\n""" % { 'pre': proj.pre.id, 'post': proj.post.id, 'target': proj.target, 'nr': num_threads }

        pop_assign = "    // populations\n"
        for pop in self.populations:
            if pop in Global.cuda_config.keys():
                pop_assign += """    pop%(pid)s.stream = streams[%(sid)s];
""" % {'pid': pop.id, 'sid': Global.cuda_config[pop]['stream'] }
            else:
                # default stream
                pop_assign += """    pop%(pid)s.stream = 0;
""" % {'pid': pop.id }

        proj_assign = "    // populations\n"
        for proj in self.projections:
            if proj in Global.cuda_config.keys():
                proj_assign += """    proj%(pid)s.stream = streams[%(sid)s];
""" % {'pid': proj.id, 'sid': Global.cuda_config[proj]['stream'] }
            else:
                # default stream
                proj_assign += """    proj%(pid)s.stream = 0;
""" % {'pid': proj.id }

        from .cuBodyTemplate import stream_setup
        stream_config = stream_setup % {
            'nbStreams': 2,
            'pop_assign': pop_assign,
            'proj_assign': proj_assign
        }

        return code, stream_config