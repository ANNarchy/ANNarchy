import ANNarchy.core.Global as Global
from ANNarchy.core.PopulationView import PopulationView

import numpy as np

class CUDAGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

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

        # Custom functions
        custom_func = self.header_custom_functions()

        from .HeaderTemplate import header_template
        return header_template % {
            'pop_struct': pop_struct,
            'proj_struct': proj_struct,
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'custom_func': custom_func
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

            # Generate the structure for the population.
            code = """
struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // concurrent kernel execution
    cudaStream_t stream;
"""
            # Spiking neurons have aditional data
            if pop.neuron_type.type == 'spike':
                # TODO
                code += """
    // Spiking population
    std::vector<bool> spike;
    std::vector<long int> last_spike;
    std::vector<int> spiked;
    std::vector<int> refractory;
    std::vector<int> refractory_remaining;
    bool record_spike;
    std::vector<std::vector<long> > recorded_spike;
"""

            # Parameters
            for var in pop.neuron_type.description['parameters']:
                if var['name'] in pop.neuron_type.description['local']:
                    code += """
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;    // host
    %(type)s *gpu_%(name)s;    // device
    bool %(name)s_dirty;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in pop.neuron_type.description['variables']:
                if var['name'] in pop.neuron_type.description['local']:
                    code += """
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;    // host
    %(type)s *gpu_%(name)s;    // device
    bool %(name)s_dirty;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
    // Global variable %(name)s
    %(type)s  %(name)s ;
    std::vector< %(type)s > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Arrays for the presynaptic sums
            code += """
    // Targets
"""
            for target in pop.neuron_type.description['targets']:
                code += """    std::vector<double> sum_%(target)s;    // host
    double *gpu_sum_%(target)s;    // device
""" % {'target' : target}

            # Global operations
            code += """
    // Global operations
"""
            for op in pop.global_operations:
                code += """    double _%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}

            # Arrays for the random numbers
            code += """
    // Random numbers
"""
            for rd in pop.neuron_type.description['random_distributions']:
                code += """    float* gpu_%(rd_name)s;
""" % { 'rd_name' : rd['name'] }

            # Delays (TODO: more variables could be delayed)
            if pop.max_delay > 1:
                code += """
    // Delays for rate-coded population
    std::deque< std::vector<double> > _delayed_r;
"""

            # Local functions
            if len(pop.neuron_type.description['functions'])>0:
                code += """
    // Local functions
"""
                for func in pop.neuron_type.description['functions']:
                    code += ' '*4 + func['cpp'] + '\n'

            # Finish the structure
            code += """
};
"""           
            pop_struct += code % {'id': pop.id}

            pop_ptr += """
extern PopStruct%(id)s pop%(id)s;
"""% {
    'id': pop.id,
}

        return pop_struct, pop_ptr

    def header_struct_proj(self):
        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in self.projections:
            code = """
struct ProjStruct%(id)s{
    int size;

    // stream
    cudaStream_t stream;
    
    // Learning flag
    bool _learning;
    // Connectivity
    std::vector<int> post_rank ;
    int* gpu_post_rank;
    std::vector< std::vector< int > > pre_rank ;
    int* gpu_pre_rank;
    int* gpu_nb_synapses;
    int* gpu_off_synapses;
    
    // flat connectivity parameters 
    int overallSynapses;
    std::vector<int> flat_idx;
    std::vector<int> flat_off;
"""

            # Delays
            if proj.max_delay > 1 and proj._synapses.uniform_delay == -1:
                code +="""
    std::vector< std::vector< int > > delay ;
    int* gpu_delay;
"""
            # Parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    // Local parameter %(name)s
    std::vector< std::vector< %(type)s > > %(name)s;    // host
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;    
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
    // Global parameter %(name)s
    std::vector<%(type)s>  %(name)s;
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;    // host
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;
    //std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    //bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
    // Global variable %(name)s
    std::vector<%(type)s>  %(name)s;    // host
    %(type)s* gpu_%(name)s;    // device
    bool %(name)s_dirty;
    //std::vector< %(type)s > recorded_%(name)s ;
    //bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Pre- or post_spike variables (including w)
            if proj.synapse.description['type'] == 'spike':
                for var in proj.synapse.description['pre_spike']:
                    if not var['name'] in proj.synapse.description['attributes'] + ['g_target']:
                        code += """
    // Local variable %(name)s added by default
    std::vector< std::vector< %(type)s > > %(name)s ;
    //std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    //bool record_%(name)s ;
""" % {'type' : 'double', 'name': var['name']}

            code += """
};    
""" 
            proj_struct += code % {'id': proj.id}

            proj_ptr += """
extern ProjStruct%(id)s proj%(id)s;
"""% {
    'id': proj.id,
}

        return proj_struct, proj_ptr

    def header_custom_functions(self):

        if len(Global._functions) == 0:
            return ""

        code = ""
        from ANNarchy.parser.Extraction import extract_functions
        for func in Global._functions:
            code +=  extract_functions(func, local_global=True)[0]['cpp'] + '\n'

        return code




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
            pop_ptr += """
PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        # struct declaration for each projection
        proj_ptr = ""
        for proj in self.projections:
            # Declaration of the structure
            proj_ptr += """
ProjStruct%(id)s proj%(id)s;
"""% {'id': proj.id}

        # Code for the global operations
        glob_ops_header, glob_ops_body = self.body_def_glops()

        # Compute presynaptic sums
        compute_sums_body, compute_sums_header, compute_sums_call = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()
        rd_update_code = self.body_update_randomdistributions()

        # Initialize device ptr
        device_init = self.body_init_device()

        # host to device and device to host transfers
        host_device_transfer, device_host_transfer = self.body_memory_transfers()

        # Initialize delayed arrays
        delay_init = "" #TODO: self.body_init_delay()

        # Initialize spike arrays
        spike_init = "" #TODO: self.body_init_spike()

        # Initialize projections
        projection_init = self.body_init_projection()

        # Initialize global operations
        globalops_init = self.body_init_globalops()
        
        # call cuda-kernel for updating the neural variables
        update_neuron_body, update_neuron_header, update_neuron_call = self.body_update_neuron()

        # Enque delayed outputs
        delay_code = "" #TODO: self.body_delay_neuron()

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
            'glob_ops_kernel': glob_ops_body
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
            'random_dist_update' : rd_update_code,
            'delay_init' : delay_init,
            'delay_code' : delay_code,
            'spike_init' : spike_init,
            'projection_init' : projection_init,
            'globalops_init' : globalops_init,
            'post_event' : post_event,
            'record' : record
        }

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

    def body_update_neuron(self):
        """
        returns all data needed for population step kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
        header = ""
        body = ""
        call = ""
        
        for pop in self.populations:
            if len(pop.neuron_type.description['variables']) == 0: # no variable
                continue

            # Neural update
            from ..Utils import generate_equation_code

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
                var += """, float* %(rd_name)s""" % { 'rd_name' : rd['name'] }

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
            loc_eqs = loc_eqs.replace("pop"+str(pop.id)+".", "")

            #
            # create kernel prototypes
            from .cuBodyTemplate import pop_kernel
            body += pop_kernel % {  'id': pop.id,
                                    'local_eqs': loc_eqs,
                                    'global_eqs': glob_eqs,
                                    'pop_size': str(pop.size),
                                    'tar': tar,
                                    'tar2': tar.replace("double*","").replace("float*","").replace("int*",""),
                                    'var': var,
                                    'var2': var.replace("double*","").replace("float*","").replace("int*",""),
                                    'par': par,
                                    'par2': par.replace("double","").replace("float*","").replace("int","")
                                 }

            #
            # create kernel prototypes
            header += """
void Pop%(id)s_step( cudaStream_t stream, double dt%(tar)s%(var)s%(par)s );
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

            from .cuBodyTemplate import pop_kernel_call
            call += pop_kernel_call % { 'id': pop.id,
                                        'tar': tar.replace("double*","").replace("int*",""),
                                        'var': var.replace("double*","").replace("int*",""),
                                        'par': par.replace("double","").replace("int","")
                                      }

        return body, header, call

    def body_computesum_proj(self):
        header = ""
        body = ""
        call = ""

        def rate_coded(proj):
            # Retrieve the psp code
            if not 'psp' in  proj.synapse.description.keys(): # default
                psp = "r[pre_rank[i]] * w[i];"
            else: # custom psp
                psp = proj.synapse.description['psp']['cpp'] % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

            # Take delays into account if any
            if proj.max_delay > 1:
                Global._error("synaptic delays are currently not supported.")

            from .cuBodyTemplate import psp_kernel
            body_code = psp_kernel % { 'id': proj.id,
                                       'pre': proj.pre.id,
                                       'post': proj.post.id,
                                       'target': proj.target,
                                       'psp': psp
                                      }

            header_code = """void Pop%(pre)s_Pop%(post)s_%(target)s_psp( cudaStream_t stream, int size, int* pre_rank, int* nb_synapses, int *offsets, double *r, double* w, double *sum_%(target)s );
""" % { 'id': proj.id,
        'pre': proj.pre.id,
        'post': proj.post.id,
        'target': proj.target,
      }

            from .cuBodyTemplate import psp_kernel_call
            call_code = psp_kernel_call % { 'id': proj.id,
                                            'pre': proj.pre.id,
                                            'post': proj.post.id,
                                            'target': proj.target,
                                          }

            return body_code, header_code, call_code

        def spiking(proj):
            Global._error("Spiking models are not supported currently on CUDA devices.")
            return "", "", ""

        # Sum over all synapses 
        for proj in self.projections:
            if proj.synapse.type == 'rate':
                b, h, c = rate_coded(proj)
            else:
                b, h, c = spiking(proj)

            header += h
            body += b
            call += c

        return body, header, call

    def body_update_synapse(self):
        header = ""
        body = ""
        call = ""

        for proj in self.projections:

            from ..Utils import generate_equation_code

            # Global variables
            global_eq = generate_equation_code(proj.id, proj.synapse.description, 'global', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

            # Local variables
            local_eq =  generate_equation_code(proj.id, proj.synapse.description, 'local', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}  

            if global_eq.strip() != '' or local_eq.strip() != '':

                # pre- and postsynaptic global operations
                pre_global_ops = []
                for pre_glob in proj.synapse.description['pre_global_operations']:
                    pre_global_ops.append( """_%(func)s_%(name)s""" % { 'func': pre_glob['function'], 'name': pre_glob['variable'] } )

                post_global_ops = []
                for post_glob in proj.synapse.description['post_global_operations']:
                    post_global_ops.append( """_%(func)s_%(name)s""" % { 'func': post_glob['function'], 'name': post_glob['variable'] } )

                # remove doubled entries
                pre_dependencies = list(set(proj.synapse.description['dependencies']['pre']))
                pre_global_ops = list(set(pre_global_ops))
                post_global_ops = list(set(post_global_ops))
                post_dependencies = list(set(proj.synapse.description['dependencies']['post']))

                # remove unnecessary stuff, transfrom index of OMP to CUDA
                local_eq = local_eq.replace("proj"+str(proj.id)+".","")
                local_eq = local_eq.replace("[i][j]","[j]")

                # remove unnecessary stuff
                global_eq = global_eq.replace("proj"+str(proj.id)+".","")

                var = ""
                par = ""
                # synaptic variables / parameters
                for attr in proj.synapse.description['variables'] + proj.synapse.description['parameters']:
                    var += """, %(type)s* %(name)s """ % { 'type': attr['ctype'], 'name': attr['name'] }

                # replace pre- and postsynaptic global operations / variable accesses
                for pre_var in pre_dependencies:
                    old = """pop%(id)s.%(name)s""" % { 'id': proj.pre.id, 'name': pre_var}
                    new = """pre_%(name)s""" % { 'name': pre_var}
                    var += ", double* " + new
                    local_eq = local_eq.replace(old, new)
                    global_eq = global_eq.replace(old, new)
                for g_op in pre_global_ops:
                    old = """pop%(id)s.%(name)s""" % { 'id': proj.pre.id, 'name': g_op}
                    new = """pre_%(name)s""" % { 'name': g_op}
                    par += ", double " + new
                    local_eq = local_eq.replace(old, new)
                    global_eq = global_eq.replace(old, new)
                for post_var in post_dependencies:
                    old = """pop%(id)s.%(name)s""" % { 'id': proj.post.id, 'name': post_var}
                    new = """post_%(name)s""" % { 'name': post_var}
                    var += ", double* " + new
                    local_eq = local_eq.replace(old, new)
                    global_eq = global_eq.replace(old, new)
                for g_op in post_global_ops:
                    old = """pop%(id)s.%(name)s""" % { 'id': proj.post.id, 'name': g_op}
                    new = """post_%(name)s""" % { 'name': g_op}
                    par += ", double " + new
                    local_eq = local_eq.replace(old, new)
                    global_eq = global_eq.replace(old, new)

                from .cuBodyTemplate import syn_kernel
                body += syn_kernel % { 'id': proj.id,
                                       'par': par,
                                       'par2': par.replace("double","").replace("int",""),
                                       'var': var,
                                       'var2': var.replace("double*","").replace("int*",""),
                                       'global_eqs': global_eq,
                                       'local_eqs': local_eq,
                                       'target': proj.target,
                                       'pre': proj.pre.id,
                                       'post': proj.post.id,
                                     }

                header += """void Proj%(id)s_step(cudaStream_t stream, int size, int* post_rank, int *pre_rank, int *offsets, int *nb_synapses, double dt%(var)s%(par)s);
    """ % { 'id': proj.id,
            'var': var,
            'par': par
          }

                #
                # calling entity
                local = ""
                for attr in proj.synapse.description['variables'] + proj.synapse.description['parameters']:
                    local += """, proj%(id)s.gpu_%(name)s """ % { 'id': proj.id, 'name': attr['name'] }

                for pre_var in pre_dependencies:
                    local += """, pop%(id)s.gpu_%(name)s """ % { 'id': proj.pre.id, 'name': pre_var }

                for post_var in post_dependencies:
                    local += """, pop%(id)s.gpu_%(name)s """ % { 'id': proj.post.id, 'name': post_var }

                glob = ""
                for g_op in pre_global_ops:
                    glob += """, pop%(id)s.%(name)s """ % { 'id': proj.pre.id, 'name': g_op }
                for g_op in post_global_ops:
                    glob += """, pop%(id)s.%(name)s """ % { 'id': proj.post.id, 'name': g_op }

                # generate code
                from .cuBodyTemplate import syn_kernel_call
                call += syn_kernel_call % { 'id': proj.id,
                                            'post': proj.post.id,
                                            'pre': proj.pre.id,
                                            'local': local,
                                            'glob': glob
                                          }

        return body, header, call

    def body_delay_neuron(self):
        code = ""
        for pop in self.populations:
            if pop.max_delay <= 1:
                continue
            code += """
    // Enqueuing outputs of pop%(id)s
    pop%(id)s._delayed_r.push_front(pop%(id)s.r);
    pop%(id)s._delayed_r.pop_back();
""" % {'id': pop.id }

        return code

    def body_postevent_proj(self):
        code = ""
        for proj in self.projections:
            if proj.synapse.type == 'spike':
                # Gather the equations
                post_code = ""
                for eq in proj.synapse.description['post_spike']:
                    post_code += ' ' * 20 + eq['eq'] %{'id_proj' : proj.id} + '\n'

                # Generate the code
                if post_code != "":
                    omp_code = '#pragma omp parallel for' if proj.post.size > 10 else ''
                    code += """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s
    if(proj%(id_proj)s._learning){
        %(omp_code)s 
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            if(pop%(id_post)s.spike[proj%(id_proj)s.post_rank[i]]){
                for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
%(post_event)s
                }
            }
        }
    }
"""%{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id,
    'post_event': post_code, 'omp_code': omp_code}

        return code

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

    def body_init_randomdistributions(self):
        code = """
    // Initialize random distribution objects
"""
        for pop in self.populations:
            for rd in pop.neuron_type.description['random_distributions']:
                code += """    
    cudaMalloc((void**)&pop%(id)s.gpu_%(rd_name)s, pop%(id)s.size * sizeof(float));
""" % {'id': pop.id, 'rd_name': rd['name'] }

        return code


    def body_init_globalops(self):
        code = """
    // Initialize global operations
"""
        for pop in self.populations:
            # Is it a specific population?
            if pop.generator['omp']['body_globalops_init']:
                code += pop.generator['omp']['body_globalops_init'] %{'id': pop.id}
                continue

            for op in pop.global_operations:
                code += """    pop%(id)s._%(op)s_%(var)s = 0.0;
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}

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
        code = """
    // Initialize delayed firing rates
"""
        for pop in self.populations:
            if pop.max_delay > 1:
                if pop.neuron_type.type == 'rate':
                    code += """    pop%(id)s._delayed_r = std::deque< std::vector<double> >(%(delay)s, std::vector<double>(pop%(id)s.size, 0.0));
""" % {'id': pop.id, 'delay': pop.max_delay}
                else: # TODO SPIKE
                    pass

        return code

    def body_init_spike(self):
        code = """
    // Initialize spike arrays
"""
        for pop in self.populations:
            if pop.neuron_type.type == 'spike':
                code += """    pop%(id)s.spike = std::vector<bool>(pop%(id)s.size, false);
    pop%(id)s.spiked = std::vector<int>(0, 0);
    pop%(id)s.last_spike = std::vector<long int>(pop%(id)s.size, -10000L);
    pop%(id)s.refractory_remaining = std::vector<int>(pop%(id)s.size, 0);
""" % {'id': pop.id}

        return code

    def body_init_projection(self):
        code = """
    // Initialize projections
"""
#         for proj in self.projections:
#                 code += """    proj%(id)s._learning = true;
# """ % {'id': proj.id}

        return code

    def body_update_randomdistributions(self):
        code = """
    // Compute random distributions""" 
        for pop in self.populations:
            if len(pop.neuron_type.description['random_distributions']) > 0:
                for rd in pop.neuron_type.description['random_distributions']:
                    code += """
    curandStatus_t %(rd_name)s_state = curandGenerateUniform(gen, pop%(id)s.gpu_%(rd_name)s, pop%(id)s.size);
    if ( %(rd_name)s_state != CURAND_STATUS_SUCCESS )
        std::cout << "curandError: " << %(rd_name)s_state << std::endl;
""" % {'id': pop.id, 'rd_name': rd['name']}

        return code

    def body_update_globalops(self):
        code = ""
        from .GlobalOperationTemplate import global_operation_templates

        code = """
    double *tmp;
    cudaMalloc((void**)&tmp, sizeof(double));
"""
        for pop in self.populations:
            # Is it a specific population?
            if pop.generator['omp']['body_update_globalops']:
                code += pop.generator['omp']['body_update_globalops'] %{ 'id': pop.id}
                continue

            for op in pop.global_operations:
                code += global_operation_templates[op['function']]['call'] % { 'id': pop.id, 'var': op['variable']  } 

        code += """
    cudaFree(tmp);
"""
        return code

    def body_record(self):
        code = ""
        for pop in self.populations:
            for var in pop.neuron_type.description['variables']:
                code += """
    if(pop%(id)s.record_%(name)s)
        pop%(id)s.recorded_%(name)s.push_back(pop%(id)s.%(name)s) ;
""" % {'id': pop.id, 'type' : var['ctype'], 'name': var['name']}
        return code

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
                Global._error("No self-defined populations / projections available on CUDA yet.")
                continue

            code = """
    cdef struct PopStruct%(id)s :
        int size
"""            
            # Spiking neurons have aditional data
            if pop.neuron_type.type == 'spike':
                code += """
        vector[int] refractory
        bool record_spike
        vector[vector[long]] recorded_spike
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
        vector[vector[%(type)s]] recorded_%(name)s 
        bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
        # Global variable %(name)s
        %(type)s  %(name)s 
        vector[%(type)s] recorded_%(name)s
        bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Arrays for the presynaptic sums of rate-coded neurons
            if pop.neuron_type.type == 'rate':
                code += """
        # Targets
"""
                for target in pop.neuron_type.description['targets']:
                    code += """        vector[double] sum_%(target)s
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
            code = """
    cdef struct ProjStruct%(id)s :
        int size
        bool _learning
        vector[int] post_rank
        vector[vector[int]] pre_rank
"""         
            # Delays
            if proj.max_delay > 1 and proj._synapses.uniform_delay == -1:
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
        #vector[vector[%(type)s]] recorded_%(name)s 
        #bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global variable %(name)s
        vector[%(type)s]  %(name)s 
        #vector[%(type)s] recorded_%(name)s
        #bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}


            # Pre- or post_spike variables (including w)
            if proj.synapse.description['type'] == 'spike':
                for var in proj.synapse.description['pre_spike']:
                    if not var['name'] in proj.synapse.description['attributes'] + ['g_target']:
                        code += """
        # Local variable %(name)s
        vector[vector[%(type)s]] %(name)s 
        #vector[vector[%(type)s]] recorded_%(name)s 
        #bool record_%(name)s 
""" % {'type' : 'double', 'name': var['name']}

            # Finalize the code
            proj_struct += code % {'id': proj.id}

            # Population instance
            proj_ptr += """
    ProjStruct%(id)s proj%(id)s"""% {
    'id': proj.id,
}
        return proj_struct, proj_ptr

    def pyx_wrapper_pop(self):
        # Cython wrappers for the populations
        code = ""
        for pop in self.populations:
            # Init
            code += """
cdef class pop%(id)s_wrapper :

    def __cinit__(self, size):
        pop%(id)s.size = size"""% {'id': pop.id}

            # Spiking neurons have aditional data
            if pop.neuron_type.type == 'spike':
                code += """
        # Spiking neuron
        pop%(id)s.refractory = vector[int](size, 0)
        pop%(id)s.record_spike = False
        pop%(id)s.recorded_spike = vector[vector[long]]()
        for i in xrange(pop%(id)s.size):
            pop%(id)s.recorded_spike.push_back(vector[long]())
"""% {'id': pop.id}

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
        pop%(id)s.recorded_%(name)s = vector[vector[%(type)s]](0, vector[%(type)s](0,%(init)s))
        pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
                else: # global
                    code += """
        pop%(id)s.%(name)s = %(init)s
        pop%(id)s.%(name)s_dirty = True
        pop%(id)s.recorded_%(name)s = vector[%(type)s](0, %(init)s)
        pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Targets
            if pop.neuron_type.type == 'rate':
                for target in pop.neuron_type.description['targets']:
                    code += """
        pop%(id)s.sum_%(target)s = vector[double](size, 0.0)""" %{'id': pop.id, 'target': target}

            # Size property
            code += """

    property size:
        def __get__(self):
            return pop%(id)s.size
""" % {'id': pop.id}

            # Spiking neurons have aditional data
            if pop.neuron_type.type == 'spike':
                code += """
    # Spiking neuron
    cpdef np.ndarray get_refractory(self):
        return pop%(id)s.refractory
    cpdef set_refractory(self, np.ndarray value):
        pop%(id)s.refractory = value

    def start_record_spike(self):
        pop%(id)s.record_spike = True
    def stop_record_spike(self):
        pop%(id)s.record_spike = False
    def get_record_spike(self):
        cdef vector[vector[long]] tmp = pop%(id)s.recorded_spike
        for i in xrange(self.size):
            pop%(id)s.recorded_spike[i].clear()
        return tmp

"""% {'id': pop.id}

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
    def start_record_%(name)s(self):
        pop%(id)s.record_%(name)s = True
    def stop_record_%(name)s(self):
        pop%(id)s.record_%(name)s = False
    def get_record_%(name)s(self):
        cdef vector[vector[%(type)s]] tmp = pop%(id)s.recorded_%(name)s
        for i in xrange(tmp.size()):
            pop%(id)s.recorded_%(name)s[i].clear()
        pop%(id)s.recorded_%(name)s.clear()
        return tmp
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

                elif var['name'] in pop.neuron_type.description['global']:
                    code += """
    # Global variable %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name']}

        return code

    def pyx_wrapper_proj(self):
        # Cython wrappers for the projections
        code = ""
        for proj in self.projections:
            # Init
            code += """
cdef class proj%(id)s_wrapper :

    def __cinit__(self, synapses):

        cdef CSR syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()
        
        proj%(id)s.size = size
        proj%(id)s._learning = True
        proj%(id)s.post_rank = syn.post_rank
        proj%(id)s.pre_rank = syn.pre_rank
        proj%(id)s.w = syn.w
"""% {'id': proj.id}

            # Delays
            if proj.max_delay > 1 and proj._synapses.uniform_delay == -1:
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

            # Initialize variables
            for var in proj.synapse.description['variables']:
                if var['name'] == 'w':
                    continue
                if var['name'] in proj.synapse.description['local']:
                    init = 0.0 if var['ctype'] == 'double' else 0
                    code += """
        proj%(id)s.%(name)s = vector[vector[%(type)s]](nb_post, vector[%(type)s]())
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

            # Pre- or post_spike variables (including w)
            if proj.synapse.description['type'] == 'spike':
                for var in proj.synapse.description['pre_spike']:
                    if not var['name'] in proj.synapse.description['attributes'] + ['g_target']:
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
""" % {'id' : proj.id, 'name': var['name'], 'type': 'double'}

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
        proj%(id)s.%(name)s = value
""" % {'id' : proj.id, 'name': var['name']}

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
""" % {'id' : proj.id, 'name': var['name']}

        return code
