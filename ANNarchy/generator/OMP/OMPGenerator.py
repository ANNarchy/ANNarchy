import ANNarchy.core.Global as Global

import numpy as np

class OMPGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

    def generate(self):

        # Propagte the global operations needed by the projections to the corresponding populations.
        self.propagate_global_ops()

        # Generate header code for the analysed pops and projs  
        with open(Global.annarchy_dir+'/generate/ANNarchy.h', 'w') as ofile:
            ofile.write(self.generate_header())
            
        # Generate cpp code for the analysed pops and projs  
        with open(Global.annarchy_dir+'/generate/ANNarchy.cpp', 'w') as ofile:
            ofile.write(self.generate_body())
            
        # Generate cython code for the analysed pops and projs  
        with open(Global.annarchy_dir+'/generate/ANNarchyCore.pyx', 'w') as ofile:
            ofile.write(self.generate_pyx())

    def propagate_global_ops(self):

        for name, proj in self.projections.iteritems():
            for op in proj.synapse.description['pre_global_operations']:
                proj.pre.global_operations.append(op)
            for op in  proj.synapse.description['post_global_operations']:
                proj.post.global_operations.append(op)



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
            'proj_ptr': proj_ptr
        }

    def header_struct_pop(self):
        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""
        for name, pop in self.populations.iteritems():
            code = """
struct PopStruct%(id)s{
    // Number of neurons
    int size;
"""
            # Spiking neurons have aditional data
            if pop.neuron.type == 'spike':
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
            for var in pop.neuron.description['parameters']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in pop.neuron.description['variables']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
    // Global variable %(name)s
    %(type)s  %(name)s ;
    std::vector< %(type)s > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Arrays for the presynaptic sums
            if pop.neuron.type == 'rate':
                code += """
    // Targets
"""
                for target in pop.neuron.description['targets']:
                    code += """    std::vector<double> sum_%(target)s;
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
            for rd in pop.neuron.description['random_distributions']:
                code += """    std::vector<double> %(rd_name)s;
    %(template)s dist_%(rd_name)s;
""" % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}


            # Delays (TODO: more variables could be delayed)
            if pop.max_delay > 1:
                code += """
    // Delays for rate-coded population
    std::deque< std::vector<double> > _delayed_r;
"""
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
        for name, proj in self.projections.iteritems():
            code = """
struct ProjStruct%(id)s{
    int size;
    // Learning flag
    bool _learning;
    // Connectivity
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;
"""

            # Delays
            if proj.max_delay > 1 and proj._synapses.uniform_delay == -1:
                code +="""
    std::vector< std::vector< int > > delay ;
"""
            # Parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    // Local parameter %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    //std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    //bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
    // Global variable %(name)s
    %(type)s  %(name)s ;
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


#######################################################################
############## BODY ###################################################
#######################################################################
    def generate_body(self):

        # struct declaration for each population
        pop_ptr = ""
        for name, pop in self.populations.iteritems():
            # Declaration of the structure
            pop_ptr += """
PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        # struct declaration for each projection
        proj_ptr = ""
        for name, proj in self.projections.iteritems():
            # Declaration of the structure
            proj_ptr += """
ProjStruct%(id)s proj%(id)s;
"""% {'id': proj.id}

        # Compute presynaptic sums
        compute_sums = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()
        rd_update_code = self.body_update_randomdistributions()

        # Initialize delayed arrays
        delay_init = self.body_init_delay()

        # Initialize spike arrays
        spike_init = self.body_init_spike()

        # Initialize projections
        projection_init = self.body_init_projection()

        # Initialize global operations
        globalops_init = self.body_init_globalops()

        # Equations for the neural variables
        update_neuron = self.body_update_neuron()

        # Enque delayed outputs
        delay_code = self.body_delay_neuron()

        # Global operations
        update_globalops = self.body_update_globalops()

        # Equations for the synaptic variables
        update_synapse = self.body_update_synapse()

        # Equations for the synaptic variables
        post_event = self.body_postevent_proj()

        # Record
        record = self.body_record()


        from .BodyTemplate import body_template
        return body_template % {
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'compute_sums' : compute_sums,
            'update_neuron' : update_neuron,
            'update_globalops' : update_globalops,
            'update_synapse' : update_synapse,
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

    def body_update_neuron(self):
        code = ""
        for name, pop in self.populations.iteritems():
            if len(pop.neuron.description['variables']) == 0: # no variable
                continue

            # Neural update
            from ..Utils import generate_equation_code

            # Global variables
            eqs = generate_equation_code(pop.id, pop.neuron.description, 'global') % {'pop': 'pop' + str(pop.id)}
            if eqs.strip() != "":
                code += """
    // Updating the global variables of population %(id)s
%(eqs)s
""" % {'id': pop.id, 'eqs': eqs}

            # Local variables
            eqs = generate_equation_code(pop.id, pop.neuron.description, 'local') % {'pop': 'pop' + str(pop.id)}
            code += """
    // Updating the local variables of population %(id)s
    #pragma omp parallel for
    for(int i = 0; i < pop%(id)s.size; i++){
%(eqs)s
""" % {'id': pop.id, 'eqs': eqs}

            # Spike emission
            if pop.neuron.type == 'spike':
                cond =  pop.neuron.description['spike']['spike_cond'] % {'pop': 'pop'+str(pop.id)}
                reset = ""; refrac = ""
                for eq in pop.neuron.description['spike']['spike_reset']:
                    reset += """
            %(reset)s
""" % {'reset': eq['cpp'] % {'pop': 'pop'+str(pop.id)}}
                    if not 'unless_refractory' in eq['constraint']:
                        refrac += """
            %(refrac)s
""" % {'refrac': eq['cpp'] % {'pop': 'pop'+str(pop.id)} }

                # Main code
                code += """
        // Emit spike depending on refractory period            
        if(%(pop)s.refractory_remaining[i] >0){ // Refractory period
%(refrac)s
            %(pop)s.refractory_remaining[i]--;
            %(pop)s.spike[i] = false;
        }
        else if(%(condition)s){
%(reset)s        

            %(pop)s.spike[i] = true;
            %(pop)s.last_spike[i] = t;
            %(pop)s.refractory_remaining[i] = %(pop)s.refractory[i];
        }
        else{
            %(pop)s.spike[i] = false;
        }

""" % {'condition' : cond, 'reset': reset, 'refrac': refrac, 'pop': 'pop'+str(pop.id) }

                # Finish parallel loop for the population
                code += """
    }
    // Gather spikes
    pop%(id)s.spiked.clear();
    for(int i=0; i< (int)pop%(id)s.size; i++){
        if(pop%(id)s.spike[i]){
            pop%(id)s.spiked.push_back(i);
            if(pop%(id)s.record_spike){
                pop%(id)s.recorded_spike[i].push_back(t);
            }

        }
"""% {'id': pop.id} 

                # End spike region


            # Finish parallel loop for the population
            code += """
    }
"""
            

        return code

    def body_delay_neuron(self):
        code = ""
        for name, pop in self.populations.iteritems():
            if pop.max_delay <= 1:
                continue
            code += """
    // Enqueuing outputs of pop%(id)s
    pop%(id)s._delayed_r.push_front(pop%(id)s.r);
    pop%(id)s._delayed_r.pop_back();
""" % {'id': pop.id }

        return code

    def body_computesum_proj(self):

        def rate_coded(proj):
            code = ""            
            # Retrieve the psp code
            if not 'psp' in  proj.synapse.description.keys(): # default
                psp = """proj%(id_proj)s.w[i][j] * pop%(id_pre)s.r[proj%(id_proj)s.pre_rank[i][j]];""" % {'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}
            else: # custom psp
                psp = proj.synapse.description['psp']['cpp'] % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id}
            # Take delays into account if any
            if proj.max_delay > 1:
                if proj._synapses.uniform_delay == -1 : # Non-uniform delays
                    psp = psp.replace(
                        'pop%(id_pre)s.r['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_r[proj%(id_proj)s.delay[i][j]-1]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id}
                    )
                else: # Uniform delays
                    psp = psp.replace(
                        'pop%(id_pre)s.r['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_r[%(delay)s]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj._synapses.uniform_delay-1)}
                    )
            # No need for openmp if less than 10 neurons
            omp_code = '#pragma omp parallel for private(sum)' if proj.post.size > 10 else ''

            code+= """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s
    %(omp_code)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        sum = 0.0;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            sum += %(psp)s
        }
        pop%(id_post)s.sum_%(target)s[proj%(id_proj)s.post_rank[i]] = sum;
    }
"""%{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'psp': psp, 'omp_code': omp_code}

            return code

        def spiking(proj):

            psp = """proj%(id_proj)s.w[i][j];"""%{'id_proj' : proj.id}
            pre_event_list = []

            for eq in proj.synapse.description['pre_spike']:
                if eq['name'] == 'g_target':
                    psp = """proj%(id_proj)s.w[i][j];"""%{'id_proj' : proj.id}
                else: 
                    pre_event_list.append(eq['eq'])

            pre_event = ""
            if len(pre_event_list) > 0: # There are other variables to update than g_target
                code = ""
                for eq in pre_event_list:
                    code += ' ' * 16 + eq % {'id_proj' : proj.id} + '\n'
                pre_event = """
                if(!pop%(id_post)s.spike[proj%(id_proj)s.post_rank[i]]){
%(pre_event)s
                }
"""%{'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'pre_event': code}

            # No need for openmp if less than 10 neurons
            omp_code = '#pragma omp parallel for firstprivate(proj%(id_proj)s_pre_spike) private(sum)' if proj.post.size > 10 else ''

            code = """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s
    std::vector<bool> proj%(id_proj)s_pre_spike = pop%(id_pre)s.spike;
    %(omp_code)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        sum = 0.0;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            if(proj%(id_proj)s_pre_spike[proj%(id_proj)s.pre_rank[i][j]]){
                sum += %(psp)s
                if(proj%(id_proj)s._learning){
%(pre_event)s
                }
            }
        }
        pop%(id_post)s.g_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
    }
"""%{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id,
    'pre_event': pre_event, 'psp': psp, 'omp_code': omp_code}

            return code

        # Reset code
        code = ""

        # Sum over all synapses 
        for name, proj in self.projections.iteritems():
            if proj.synapse.type == 'rate':
                code += rate_coded(proj)
            else:
                code += spiking(proj)

        return code

    def body_postevent_proj(self):
        code = ""
        for name, proj in self.projections.iteritems():
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


    def body_update_synapse(self):
        # Reset code
        code = ""
        # Sum over all synapses 
        for name, proj in self.projections.iteritems():
            from ..Utils import generate_equation_code
            # Global variables
            global_eq = generate_equation_code(proj.id, proj.synapse.description, 'global', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}
            code+= """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s
    if(proj%(id_proj)s._learning){
%(global)s
"""%{'id_proj' : proj.id, 'global': global_eq, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

            # Local variables
            local_eq =  generate_equation_code(proj.id, proj.synapse.description, 'local', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}  
            if local_eq.strip() != "": 
                code+= """
        #pragma omp parallel for 
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                %(local)s
            }
        }
"""%{'id_proj' : proj.id, 'local': local_eq}

            code += """
    }"""

        return code


    def body_init_randomdistributions(self):
        code = """
    // Initialize random distribution objects
"""
        for name, pop in self.populations.iteritems():
            for rd in pop.neuron.description['random_distributions']:
                code += """    pop%(id)s.%(rd_name)s = std::vector<double>(pop%(id)s.size, 0.0);
    pop%(id)s.dist_%(rd_name)s = %(rd_init)s;
""" % {'id': pop.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': pop.id}}

        return code


    def body_init_globalops(self):
        code = """
    // Initialize global operations
"""
        for name, pop in self.populations.iteritems():
            for op in pop.global_operations:
                code += """    pop%(id)s._%(op)s_%(var)s = 0.0;
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}

        return code

    def body_init_delay(self):
        code = """
    // Initialize delayed firing rates
"""
        for name, pop in self.populations.iteritems():
            if pop.max_delay > 1:
                if pop.neuron.type == 'rate':
                    code += """    pop%(id)s._delayed_r = std::deque< std::vector<double> >(%(delay)s, std::vector<double>(pop%(id)s.size, 0.0));
""" % {'id': pop.id, 'delay': pop.max_delay}
                else: # TODO SPIKE
                    pass

        return code

    def body_init_spike(self):
        code = """
    // Initialize spike arrays
"""
        for name, pop in self.populations.iteritems():
            if pop.neuron.type == 'spike':
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
#         for name, proj in self.projections.iteritems():
#                 code += """    proj%(id)s._learning = true;
# """ % {'id': proj.id}

        return code

    def body_update_randomdistributions(self):
        code = """
    // Compute random distributions""" 
        for name, pop in self.populations.iteritems():
            if len(pop.neuron.description['random_distributions']) > 0:
                code += """
    // RD of pop%(id)s
    #pragma omp parallel for
    for(int i = 0; i < pop%(id)s.size; i++)
    {
"""% {'id': pop.id}
                for rd in pop.neuron.description['random_distributions']:
                    code += """
        pop%(id)s.%(rd_name)s[i] = pop%(id)s.dist_%(rd_name)s(rng[omp_get_thread_num()]);
""" % {'id': pop.id, 'rd_name': rd['name']}

                code += """
    }
"""
        return code

    def body_update_globalops(self):
        code = ""
        for name, pop in self.populations.iteritems():
            for op in pop.global_operations:
                code += """    pop%(id)s._%(op)s_%(var)s = %(op)s_value(pop%(id)s.%(var)s);
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}
        return code

    def body_record(self):
        code = ""
        for name, pop in self.populations.iteritems():
            for var in pop.neuron.description['variables']:
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
        for name, pop in self.populations.iteritems():
            code = """
    cdef struct PopStruct%(id)s :
        int size
"""            
            # Spiking neurons have aditional data
            if pop.neuron.type == 'spike':
                code += """
        vector[int] refractory
        bool record_spike
        vector[vector[long]] recorded_spike
"""
            # Parameters
            for var in pop.neuron.description['parameters']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
        # Local parameter %(name)s
        vector[%(type)s] %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
        # Global parameter %(name)s
        %(type)s  %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in pop.neuron.description['variables']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
        # Local variable %(name)s
        vector[%(type)s] %(name)s 
        vector[vector[%(type)s]] recorded_%(name)s 
        bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
        # Global variable %(name)s
        %(type)s  %(name)s 
        vector[%(type)s] recorded_%(name)s
        bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Arrays for the presynaptic sums of rate-coded neurons
            if pop.neuron.type == 'rate':
                code += """
        # Targets
"""
                for target in pop.neuron.description['targets']:
                    code += """        vector[double] sum_%(target)s
""" % {'target' : target}

            # Finalize the code
            pop_struct += code % {'id': pop.id}

            # Population instance
            pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {
    'id': pop.id,
}
        return pop_struct, pop_ptr

    def pyx_struct_proj(self):
        proj_struct = ""
        proj_ptr = ""
        for name, proj in self.projections.iteritems():
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
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global parameter %(name)s
        %(type)s  %(name)s 
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
        %(type)s  %(name)s 
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
        for name, pop in self.populations.iteritems():
            # Init
            code += """
cdef class pop%(id)s_wrapper :

    def __cinit__(self, size):
        pop%(id)s.size = size"""% {'id': pop.id}

            # Spiking neurons have aditional data
            if pop.neuron.type == 'spike':
                code += """
        # Spiking neuron
        pop%(id)s.refractory = vector[int](size, 0)
        pop%(id)s.record_spike = False
        pop%(id)s.recorded_spike = vector[vector[long]]()
        for i in xrange(pop%(id)s.size):
            pop%(id)s.recorded_spike.push_back(vector[long]())
"""% {'id': pop.id}

            # Parameters
            for var in pop.neuron.description['parameters']:
                init = 0.0 if var['ctype'] == 'double' else 0
                if var['name'] in pop.neuron.description['local']:                    
                    code += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
                else: # global
                    code += """
        pop%(id)s.%(name)s = %(init)s""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Variables
            for var in pop.neuron.description['variables']:
                init = 0.0 if var['ctype'] == 'double' else 0
                if var['name'] in pop.neuron.description['local']:
                    code += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)
        pop%(id)s.recorded_%(name)s = vector[vector[%(type)s]](0, vector[%(type)s](0,%(init)s))
        pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
                else: # global
                    code += """
        pop%(id)s.%(name)s = %(init)s
        pop%(id)s.recorded_%(name)s = vector[%(type)s](0, %(init)s)
        pop%(id)s.record_%(name)s = False""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Targets
            if pop.neuron.type == 'rate':
                for target in pop.neuron.description['targets']:
                    code += """
        pop%(id)s.sum_%(target)s = vector[double](size, 0.0)""" %{'id': pop.id, 'target': target}

            # Size property
            code += """

    property size:
        def __get__(self):
            return pop%(id)s.size
""" % {'id': pop.id}

            # Spiking neurons have aditional data
            if pop.neuron.type == 'spike':
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
            for var in pop.neuron.description['parameters']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
    # Local parameter %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

                elif var['name'] in pop.neuron.description['global']:
                    code += """
    # Global parameter %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            # Variables
            for var in pop.neuron.description['variables']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
    # Local variable %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
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

                elif var['name'] in pop.neuron.description['global']:
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
        for name, proj in self.projections.iteritems():
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