import ANNarchy.core.Global as Global
from ANNarchy.core.PopulationView import PopulationView
from .PopulationGenerator import PopulationGenerator

import numpy as np

class OMPGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

        self.popgen = PopulationGenerator()

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
            # Generate the code
            code = self.popgen.header_struct(pop)      
            # Add the code to the file
            pop_struct += code % {'id': pop.id}
            pop_ptr += """extern PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        return pop_struct, pop_ptr

    def header_struct_proj(self):
        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in self.projections:
            
            # Is it a specific projection?
            if proj.generator['omp']['header_proj_struct']:
                proj_struct += proj.generator['omp']['header_proj_struct']
                proj_ptr += """extern ProjStruct%(id_proj)s proj%(id_proj)s;
"""% {'id_proj': proj.id}
                continue
            
            code = """
// %(pre_name)s -> %(post_name)s
struct ProjStruct%(id_proj)s{
    int size;
    // Learning flag
    bool _learning;
    // Connectivity
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;
"""

            # Spiking neurons have aditional data
            if proj.synapse.type == 'spike':
                code += """
    std::map< int, std::vector< std::pair<int, int> > > inv_rank ;
"""
            
            # Exact integration
            has_exact = False
            for var in proj.synapse.description['variables']:
                if var['method'] == 'exact':
                    has_exact = True
            if has_exact:
                code += """
    std::vector<std::vector<long> > _last_event;
"""
            # Delays
            if proj.max_delay > 1 and proj.uniform_delay == -1:
                code +="""
    std::vector< std::vector< int > > delay ;
"""

            # Arrays for the random numbers
            code += """
    // Random numbers
"""
            for rd in proj.synapse.description['random_distributions']:
                code += """    std::vector< std::vector<double> > %(rd_name)s;
    %(template)s dist_%(rd_name)s;
""" % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}

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
    std::vector<%(type)s>  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    std::vector< std::vector< std::vector< %(type)s > > > recorded_%(name)s ;
    std::vector< int > record_%(name)s ;
    std::vector< int > record_period_%(name)s ;
    std::vector< long int > record_offset_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
    // Global variable %(name)s
    std::vector<%(type)s>  %(name)s ;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    std::vector< int > record_%(name)s ;
    std::vector< int > record_period_%(name)s ;
    std::vector< long int > record_offset_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Local functions
            if len(proj.synapse.description['functions'])>0:
                code += """
    // Local functions
"""
                for func in proj.synapse.description['functions']:
                    code += ' '*4 + func['cpp'] + '\n'

            # Structural plasticity
            if Global.config['structural_plasticity']:
                # Pruning defined in the synapse
                if 'pruning' in proj.synapse.description.keys():
                    code +="""
    // Pruning
    bool _pruning;
    int _pruning_period;
    long int _pruning_offset;
"""
                # Retrieve the names of extra attributes   
                extra_args = ""
                add_code = ""
                remove_code = ""
                for var in proj.synapse.description['parameters'] + proj.synapse.description['variables']:
                    if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse.description['local']:
                        extra_args += ', ' + var['ctype'] + ' _' +  var['name'] 
                        add_code += ' '*8 + var['name'] + '[post].insert('+var['name']+'[post].begin() + idx, _' + var['name'] + ');\n'
                        remove_code += ' '*8 + var['name'] + '[post].erase(' + var['name'] + '[post].begin() + idx);\n'
                # Delays
                delay_code = ""
                if proj.max_delay > 1 and proj.uniform_delay == -1:
                    delay_code = "delay[post].insert(delay[post].begin() + idx, _delay)"
                # Spiking networks must update the inv_rank array
                spiking_addcode = ""
                spiking_removecode = ""
                if proj.synapse.type == 'spike':
                    spiking_addcode = """
        // Add the corresponding pair in inv_rank
        int idx_post = 0;
        for(int i=0; i<post_rank.size(); i++){
            if(post_rank[i] == post){
                idx_post = i;
                break;
            }
        }
        inv_rank[pre].push_back(std::pair<int, int>(idx_post, idx));
"""
                    spiking_removecode = """
        // Remove the corresponding pair in inv_rank
        for(int i=0; i<inv_rank[pre].size(); i++){
            if(inv_rank[pre][i].second == idx){
                inv_rank[pre].erase(inv_rank[pre].begin() + i);
                break;
            }
        }
"""
                # Generate the code
                code += """
    // Structural plasticity
    void addSynapse(int post, int pre, double weight, int _delay%(extra_args)s){
        // Find the index of the synapse
        int idx = 0;
        for(int i=0; i<pre_rank[post].size(); i++){
            if(pre_rank[post][i] > pre){
                idx = i;
                break;
            }
        }
        pre_rank[post].insert(pre_rank[post].begin() + idx, pre);
        w[post].insert(w[post].begin() + idx, weight);
        %(delay_code)s
%(add_code)s
%(spike_add)s
    };
    void removeSynapse(int post, int pre){
        // Find the index of the synapse
        int idx = 0;
        for(int i=0; i<pre_rank[post].size(); i++){
            if(pre_rank[post][i] == pre){
                idx = i;
                break;
            }
        }
        pre_rank[post].erase(pre_rank[post].begin() + idx);
        w[post].erase(w[post].begin() + idx);
%(remove_code)s  
%(spike_remove)s
    };
""" % {'extra_args': extra_args, 'delay_code': delay_code, 'add_code': add_code, 'remove_code': remove_code,
        'spike_add': spiking_addcode, 'spike_remove': spiking_removecode}

            # Finish the structure
            code += """
};    
""" 
            proj_struct += code % {'id_proj': proj.id, 'pre_name': proj.pre.name, 'post_name': proj.post.name}

            proj_ptr += """extern ProjStruct%(id_proj)s proj%(id_proj)s;
"""% {
    'id_proj': proj.id,
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
        glop_definition = self.body_def_glops()

        # Reset presynaptic sums
        reset_sums = self.body_resetcomputesum_pop()

        # Compute presynaptic sums
        compute_sums = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()
        rd_update_code = self.body_update_randomdistributions()

        # Initialize delayed arrays
        delay_init = self.body_init_delay()

        # Initialize populations
        pop_init = self.body_init_population()

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

        # Equations for the post-events
        post_event = self.body_postevent_proj()

        # Structural plasticity
        structural_plasticity = self.body_structural_plasticity()

        # Record
        record = self.body_record()

        # Early stopping
        run_until = self.body_run_until()


        from .BodyTemplate import body_template
        return body_template % {
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'glops_def': glop_definition,
            'run_until': run_until,
            'compute_sums' : compute_sums,
            'reset_sums' : reset_sums,
            'update_neuron' : update_neuron,
            'update_globalops' : update_globalops,
            'update_synapse' : update_synapse,
            'random_dist_init' : rd_init_code,
            'random_dist_update' : rd_update_code,
            'delay_init' : delay_init,
            'delay_code' : delay_code,
            'spike_init' : pop_init,
            'projection_init' : projection_init,
            'globalops_init' : globalops_init,
            'post_event' : post_event,
            'structural_plasticity': structural_plasticity,
            'record' : record
        }

    def body_update_neuron(self):
        code = ""
        for pop in self.populations:
            code  += self.popgen.update_neuron(pop)

        return code

    def body_delay_neuron(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.delay_code(pop)
        return code

    def body_computesum_proj(self):

        def rate_coded(proj):
            code = ""            
            # Retrieve the psp code
            if not 'psp' in  proj.synapse.description.keys(): # default
                psp = """proj%(id_proj)s.w[i][j] * pop%(id_pre)s.r[proj%(id_proj)s.pre_rank[i][j]];""" % {'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}
            else: # custom psp
                psp = (proj.synapse.description['psp']['cpp'] % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id}).replace('rk_pre', 'proj%(id_proj)s.pre_rank[i][j]'% {'id_proj' : proj.id})
            # Take delays into account if any
            if proj.max_delay > 1:
                if proj.uniform_delay == -1 : # Non-uniform delays
                    psp = psp.replace(
                        'pop%(id_pre)s.r['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_r[proj%(id_proj)s.delay[i][j]-1]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id}
                    )
                else: # Uniform delays
                    psp = psp.replace(
                        'pop%(id_pre)s.r['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_r[%(delay)s]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
                    )
            # No need for openmp if less than 10 neurons
            omp_code = '#pragma omp parallel for private(sum)' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''

            # Generate the code depending on the operation
            if proj.synapse.operation == 'sum': # normal summation
                code+= """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. operation = sum
    if (pop%(id_post)s._active){
    %(omp_code)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        sum = 0.0;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            sum += %(psp)s
        }
        pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
    }
    } // active
"""%{'id_proj' : proj.id, 'target': proj.target, 
    'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'name_post': proj.post.name, 'name_pre': proj.pre.name, 
    'psp': psp, 'omp_code': omp_code}

            elif proj.synapse.operation == 'max': # max pooling
                code+= """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. operation = max
    if (pop%(id_post)s._active){
    %(omp_code)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        int j= 0;
        sum = %(psp)s ;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            if(%(psp)s > sum){
                sum = %(psp)s ;
            }
        }
        pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
    }
    } // active
"""%{'id_proj' : proj.id, 'target': proj.target, 
    'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'name_post': proj.post.name, 'name_pre': proj.pre.name, 
    'psp': psp.replace(';', ''), 'omp_code': omp_code}

            elif proj.synapse.operation == 'min': # max pooling
                code+= """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. operation = min
    if (pop%(id_post)s._active){
    %(omp_code)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        int j= 0;
        sum = %(psp)s ;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            if(%(psp)s < sum){
                sum = %(psp)s ;
            }
        }
        pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
    }
    } // active
"""%{'id_proj' : proj.id, 'target': proj.target, 
    'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'name_post': proj.post.name, 'name_pre': proj.pre.name, 
    'psp': psp.replace(';', ''), 'omp_code': omp_code}

            elif proj.synapse.operation == 'mean': # max pooling
                code+= """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. operation = mean
    if (pop%(id_post)s._active){
    %(omp_code)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        sum = 0.0 ;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            sum += %(psp)s ;
        }
        pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum / (double)(proj%(id_proj)s.pre_rank[i].size());
    }
    } // active
"""%{'id_proj' : proj.id, 'target': proj.target, 
    'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'name_post': proj.post.name, 'name_pre': proj.pre.name, 
    'psp': psp.replace(';', ''), 'omp_code': omp_code}

            return code

        def spiking(proj):

            ids = {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target} 

            # Analyse all elements of pre_spike
            pre_event = ""
            pre_event_list = []
            learning = ""
            psp = ""
            for eq in proj.synapse.description['pre_spike']:
                if eq['name'] == 'w':
                    learning = """
                    if(proj0._learning){
                        %(eq)s 
                    }
""" % {'eq': eq['cpp'] % ids}
                elif eq['name'] == 'g_target':
                    psp = eq['cpp'].split('=')[1]
                else:
                    pre_event_list.append(eq['cpp'])

            # Is the summation event-based or psp-based?
            event_based = True
            psp_sum = None
            if 'psp' in  proj.synapse.description.keys(): # not event-based
                event_based = False
                psp_code = ""            
                # Event-based summation of psp
            elif psp == "": # default g_target += w
                psp_code = """pop%(id_post)s.g_%(target)s[proj%(id_proj)s.post_rank[i]] += proj%(id_proj)s.w[i][j]
""" % ids
            else:
                psp_code = """pop%(id_post)s.g_%(target)s[proj%(id_proj)s.post_rank[i]] += %(psp)s
""" % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target, 'psp': psp % ids}

            # Exact integration
            has_exact = False
            exact_code = ''
            for var in proj.synapse.description['variables']:
                if var['method'] == 'exact':
                    has_exact = True
                    exact_code += """
            %(exact)s
""" % {'exact': var['cpp'].replace('(t)', '(t-1)') %{'id_proj' : proj.id}}
            if has_exact:
                    event_based = False # to avoid the if not post.spike
                    exact_code += """
            proj%(id_proj)s._last_event[i][j] = t;
""" % {'id_proj' : proj.id, 'exact': var['cpp']}

            
            # Other event-driven variables
            if len(pre_event_list) > 0: # There are other variables to update than g_target
                code = ""
                for eq in pre_event_list:
                    code += ' ' * 12 + eq % {'id_proj' : proj.id} + '\n'

                if event_based:
                    pre_event += """
                // Event-based variables should not be updated when the postsynaptic neuron fires.
            if(!pop%(id_post)s.spike[proj%(id_proj)s.post_rank[i]]){
%(pre_event)s
%(learning)s
            }
"""% {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'pre_event': code, 'learning': learning}
                else:
                    pre_event += """
%(pre_event)s
%(learning)s
"""% {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'pre_event': code, 'learning': learning}

            # Take delays into account if any
            if proj.max_delay > 1:
                if proj.uniform_delay == -1 : # Non-uniform delays
                    Global._error('Non-uniform delays are not yet possible for spiking networks.')
                    exit()
                else: # Uniform delays
                    pre_array = "pop%(id_pre)s._delayed_spike[%(delay)s]" % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
            else:
                pre_array = "pop%(id_pre)s.spiked" % ids

            # No need for openmp if less than 10 neurons
            omp_code = """#pragma omp parallel for firstprivate(nb_post, inv_post) private(i, j)""" if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''

            code = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. event-based
    if (pop%(id_post)s._active){
        for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++){
            rk_j = %(pre_array)s[_idx_j];
            int nb_post = proj%(id_proj)s.inv_rank[rk_j].size();
            std::vector< std::pair<int, int> > inv_post = proj%(id_proj)s.inv_rank[rk_j];
            %(omp_code)s
            for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                i = inv_post[_idx_i].first;
                j = inv_post[_idx_i].second;
    %(exact)s
                %(psp)s
    %(pre_event)s
            }
        }
    } // active
"""%{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'name_post': proj.post.name, 'name_pre': proj.pre.name, 'pre_array': pre_array,
    'pre_event': pre_event, 'psp': psp_code , 'omp_code': omp_code,
    'exact': exact_code}

            # Not even-driven summation of psp
            if 'psp' in  proj.synapse.description.keys(): # not event-based
                omp_code = """#pragma omp parallel for private(sum)""" if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
                psp_sum = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. sum of psp
    if (pop%(id_post)s._active){
    %(omp_code)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        sum = 0.0;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            sum += %(psp)s
        }
        pop%(id_post)s.g_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
    }
    } // active
""" % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target, 
       'name_post': proj.post.name, 'name_pre': proj.pre.name, 
       'psp': proj.synapse.description['psp']['cpp'] % ids, 'omp_code': omp_code}

                code += psp_sum


            return code

        # Reset code
        code = ""

        # Sum over all synapses 
        for proj in self.projections:
            # Is it a specific projection?
            if proj.generator['omp']['body_compute_psp']:
                code += proj.generator['omp']['body_compute_psp'] 
                continue
            # Call the right generator depnding on type
            if proj.synapse.type == 'rate':
                code += rate_coded(proj)
            else:
                code += spiking(proj)

        return code

    def body_resetcomputesum_pop(self):
        code = ""
        for pop in self.populations:
            if pop.neuron_type.type == 'rate':
                code += self.popgen.reset_computesum(pop)
        
        return code

    def body_postevent_proj(self):
        code = ""
        for proj in self.projections:
            if proj.synapse.type == 'spike':
                if proj.synapse.description['post_spike'] == []:
                    continue

                post_code = ""

                # Exact integration
                has_exact = False
                for var in proj.synapse.description['variables']:
                    if var['method'] == 'exact':
                        has_exact = True
                        post_code += """
                    %(exact)s
""" % {'exact': var['cpp'] %{'id_proj' : proj.id}}
                if has_exact:
                    post_code += """
                    proj%(id_proj)s._last_event[i][j] = t;
""" % {'id_proj' : proj.id, 'exact': var['cpp']}

                # Gather the equations
                for eq in proj.synapse.description['post_spike']:
                    post_code += ' ' * 16 + eq['cpp'] %{'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id} + '\n'

                # Generate the code
                if post_code != "":
                    omp_code = '#pragma omp parallel for private(j) firstprivate(i)' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''

                    code += """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
    if(proj%(id_proj)s._learning && pop%(id_post)s._active){
        for(int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++){
            i = pop%(id_post)s.spiked[_idx_i];
            %(omp_code)s 
            for(j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
%(post_event)s
            }
        }
    }
"""%{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'name_post': proj.post.name, 'name_pre': proj.pre.name,
    'post_event': post_code, 'omp_code': omp_code}

        return code


    def body_update_synapse(self):
        # Reset code
        code = ""
        # Sum over all synapses 
        for proj in self.projections:

            from ..Utils import generate_equation_code
            
            # Global variables
            global_eq = generate_equation_code(proj.id, proj.synapse.description, 'global', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

            # Local variables
            local_eq =  generate_equation_code(proj.id, proj.synapse.description, 'local', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id} 

            # Generate the code
            if local_eq.strip() != '' or global_eq.strip() != '' :
                omp_code = '#pragma omp parallel for private(rk_pre, rk_post)' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
                code+= """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
    if(proj%(id_proj)s._learning && pop%(id_post)s._active){
        %(omp)s
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            rk_post = proj%(id_proj)s.post_rank[i];
%(global)s
"""%{'id_proj' : proj.id, 'global': global_eq, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'name_post': proj.post.name, 'name_pre': proj.pre.name, 'omp': omp_code}
 
                if local_eq.strip() != "": 
                    code+= """
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                rk_pre = proj%(id_proj)s.pre_rank[i][j];
                %(local)s
            }
"""%{'id_proj' : proj.id, 'local': local_eq}

                code += """
        }
    }
"""

            # Take delays into account if any
            if proj.max_delay > 1:
                if proj.uniform_delay == -1 : # Non-uniform delays
                    code = code.replace(
                        'pop%(id_pre)s.r['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_r[proj%(id_proj)s.delay[i][j]-1]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id}
                    )
                    code = code.replace(
                        'pop%(id_pre)s.spike['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_spike[proj%(id_proj)s.delay[i][j]-1]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id}
                    )
                else: # Uniform delays
                    code = code.replace(
                        'pop%(id_pre)s.r['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_r[%(delay)s]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
                    )
                    code = code.replace(
                        'pop%(id_pre)s.spike['%{'id_pre': proj.pre.id}, 
                        'pop%(id_pre)s._delayed_spike[%(delay)s]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
                    )

        return code


    def body_structural_plasticity(self):

        # Pruning if any
        pruning=""
        if Global.config['structural_plasticity'] :
            for proj in self.projections:
                if 'pruning' in proj.synapse.description.keys():
                    pruning_structure = proj.synapse.description['pruning']
                    pruning += """
    // proj%(id_proj)s pruning: %(eq)s
    if((proj%(id_proj)s._pruning)&&((t - proj%(id_proj)s._pruning_offset) %(modulo)s proj%(id_proj)s._pruning_period == 0)){
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            rk_post = proj%(id_proj)s.post_rank[i];
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                rk_pre = proj%(id_proj)s.pre_rank[i][j];
                if(%(condition)s){
                    proj%(id_proj)s.removeSynapse(i, rk_pre);
                }
            }
        }
    }
""" % {'id_proj' : proj.id, 'eq': pruning_structure['eq'], 'modulo': '%', 'condition': pruning_structure['cpp'] % {'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}}

        return pruning

    def body_init_randomdistributions(self):
        code = """
    // Initialize random distribution objects
"""
        for pop in self.populations:
            code += self.popgen.init_randomdistributions(pop)

        for proj in self.projections:
            # Is it a specific population?
            if proj.generator['omp']['body_random_dist_init']:
                code += proj.generator['omp']['body_random_dist_init'] %{'id_proj': proj.id}
                continue

            for rd in proj.synapse.description['random_distributions']:
                code += """    proj%(id_proj)s.%(rd_name)s = std::vector< std::vector<double> >(proj%(id_proj)s.post_rank.size(), std::vector<double>());
    for(int i=0; i<proj%(id_proj)s.post_rank.size(); i++){
        proj%(id_proj)s.%(rd_name)s[i] = std::vector<double>(proj%(id_proj)s.pre_rank[i].size(), 0.0);
    }
    proj%(id_proj)s.dist_%(rd_name)s = %(rd_init)s;
""" % {'id_proj': proj.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': proj.id}}

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
            return ""

        from .GlobalOperationTemplate import global_operation_templates
        code = ""
        for op in list(set(ops)):
            code += global_operation_templates[op]

        return code

    def body_init_delay(self):
        code = ""
        for pop in self.populations:
            if pop.max_delay > 1: # no need to generate the code otherwise
                code += self.popgen.init_delay(pop)

        return code

    def body_init_population(self):
        code = """
    // Initialize population variables
"""
        for pop in self.populations:
            code += self.popgen.init_population(pop)

        return code

    def body_init_projection(self):
        code = """
    // Initialize projections
"""
        for proj in self.projections:

            # Is it a specific projection?
            if proj.generator['omp']['body_proj_init']:
                code += proj.generator['omp']['body_proj_init'] 
                continue

            # Learning by default
            code += """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
    proj%(id_proj)s._learning = true;
""" % {'id_proj': proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'name_post': proj.post.name, 'name_pre': proj.pre.name}

            # Spiking neurons have aditional data
            if proj.synapse.type == 'spike':
                code += """
    proj%(id_proj)s.inv_rank =  std::map< int, std::vector< std::pair<int, int> > > ();
    for(int i=0; i<proj%(id_proj)s.pre_rank.size(); i++){
        for(int j=0; j<proj%(id_proj)s.pre_rank[i].size(); j++){
            proj%(id_proj)s.inv_rank[proj%(id_proj)s.pre_rank[i][j]].push_back(std::pair<int, int>(i,j));
        }
    }
"""% {'id_proj': proj.id}

                debug = """
// For debug...
    for (std::map< int, std::vector< std::pair<int, int> > >::iterator it=proj%(id_proj)s.inv_rank.begin(); it!=proj%(id_proj)s.inv_rank.end(); ++it) {
        std::cout << it->first << ": " ;
        for(int _id=0; _id<it->second.size(); _id++){
            std::pair<int, int> val = it->second[_id];
            std::cout << "(" << val.first << ", " << val.second << "), " ;
        }
        std::cout << std::endl ;
    }
"""

            # Recording
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """    proj%(id)s.recorded_%(name)s = std::vector< std::vector< std::vector< %(type)s > > > (proj%(id)s.post_rank.size(), std::vector< std::vector< %(type)s > >());
"""% {'id': proj.id, 'name': var['name'], 'type': var['ctype']}
                else:
                    code += """    proj%(id)s.recorded_%(name)s = std::vector< std::vector< %(type)s > > (proj%(id)s.post_rank.size(), std::vector< %(type)s >());
"""% {'id': proj.id, 'name': var['name'], 'type': var['ctype']}

            # Pruning
            if Global.config['structural_plasticity'] and 'pruning' in proj.synapse.description.keys():
                code +="""
    // Pruning
    proj%(id_proj)s._pruning = false;
    proj%(id_proj)s._pruning_period = 1;
    proj%(id_proj)s._pruning_offset = 0;
"""% {'id_proj': proj.id}
            

        return code
        
    def body_update_randomdistributions(self):
        code = "" 
        for pop in self.populations:
            code += self.popgen.update_random_distributions(pop)

        for proj in self.projections:
            # Is it a specific population?
            if proj.generator['omp']['body_random_dist_update']:
                code += proj.generator['omp']['body_random_dist_update'] %{'id': pop.id}
                continue

            if len(proj.synapse.description['random_distributions']) > 0:
                code += """
    // RD of proj%(id_proj)s
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
"""% {'id_proj': proj.id}

                for rd in proj.synapse.description['random_distributions']:
                    code += """
            proj%(id_proj)s.%(rd_name)s[i][j] = proj%(id_proj)s.dist_%(rd_name)s(rng);""" % {'id_proj': proj.id, 'rd_name': rd['name']}

                code += """
        }
    }
"""
        return code

    def body_update_globalops(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.update_globalops(pop)
        return code

    def body_record(self):
        code = ""
        # Populations
        for pop in self.populations:
           code += self.popgen.record(pop)

        # Projections
        for proj in self.projections:
            for var in proj.synapse.description['variables']:
                    code += """
    for(int i=0; i< proj%(id)s.record_%(name)s.size(); i++){
        if((t - proj%(id)s.record_offset_%(name)s[i]) %(modulo)s proj%(id)s.record_period_%(name)s[i] == 0)
            proj%(id)s.recorded_%(name)s[i].push_back(proj%(id)s.%(name)s[i]) ;
    }
""" % {'id': proj.id, 'name': var['name'], 'modulo': '%'}

        return code

    def body_run_until(self):
        # Check if it is useful to generate anything at all
        for pop in self.populations:
            if pop.stop_condition: # a condition has been defined
                break
        else:
            return """
    run(steps);
    return steps;
"""

        # Generate the conditional code
        complete_code = """
    bool stop = false;
    bool pop_stop = false;
    int nb = 0;
    for(int n = 0; n < steps; n++)
    {
        step();
        nb++;
        stop = or_and;
        for(int i=0; i<populations.size();i++)
        {
            // Check all populations
            switch(populations[i]){
%(run_until)s
            }

            // Accumulate the results
            if(or_and)
                stop = stop && pop_stop;
            else
                stop = stop || pop_stop;
        }
        if(stop)
            break;
    }
    return nb;

"""
        code = ""
        for pop in self.populations:
            code += self.popgen.stop_condition(pop)

        return complete_code % {'run_until': code}


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
            # Header Wrapper
            pop_struct += self.popgen.pyx_struct(pop)
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
            if proj.generator['omp']['pyx_proj_struct']:
                proj_struct += proj.generator['omp']['pyx_proj_struct']
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
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global parameter %(name)s
        vector[%(type)s]  %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
        # Local variable %(name)s
        vector[vector[%(type)s]] %(name)s 
        vector[vector[vector[%(type)s]]] recorded_%(name)s 
        vector[int] record_%(name)s 
        vector[int] record_period_%(name)s 
        vector[long] record_offset_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global variable %(name)s
        vector[%(type)s]  %(name)s 
        vector[vector[%(type)s]] recorded_%(name)s
        vector[int] record_%(name)s 
        vector[int] record_period_%(name)s 
        vector[long] record_offset_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Structural plasticity
            if Global.config['structural_plasticity']:

                # Pruning in the synapse
                if 'pruning' in proj.synapse.description.keys():
                    code += """
        # Pruning
        bool _pruning
        int _pruning_period
        long _pruning_offset
"""

                # Retrieve the names of extra attributes   
                extra_args = ""
                for var in proj.synapse.description['parameters'] + proj.synapse.description['variables']:
                    if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse.description['local']:
                        extra_args += ', ' + var['ctype'] + ' ' +  var['name'] 
                # Generate the code
                code += """
        # Structural plasticity
        void addSynapse(int post, int pre, double weight, int _delay%(extra_args)s)
        void removeSynapse(int post, int pre)
""" % {'extra_args': extra_args}

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
            code += self.popgen.pyx_wrapper(pop)
        return code

    def pyx_wrapper_proj(self):
        # Cython wrappers for the projections
        code = ""
        for proj in self.projections:
            # Is it a specific population?
            if proj.generator['omp']['pyx_proj_class']:
                code += proj.generator['omp']['pyx_proj_class'] 
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
        proj%(id)s.record_%(name)s = vector[int]()
        proj%(id)s.record_period_%(name)s = vector[int]()
        proj%(id)s.record_offset_%(name)s = vector[long]()
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
                else:
                    init = 0.0 if var['ctype'] == 'double' else 0
                    code += """
        proj%(id)s.%(name)s = vector[%(type)s](nb_post, %(init)s)
        proj%(id)s.record_%(name)s = vector[int]()
        proj%(id)s.record_period_%(name)s = vector[int]()
        proj%(id)s.record_offset_%(name)s = vector[long]()
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
    def get_dendrite_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_dendrite_%(name)s(self, int rank, %(type)s value):
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
    def start_record_%(name)s(self, int rank, int period, long int offset):
        if not rank in list(proj%(id)s.record_%(name)s):
            proj%(id)s.record_%(name)s.push_back(rank)
            proj%(id)s.record_period_%(name)s.push_back(period)
            proj%(id)s.record_offset_%(name)s.push_back(offset)
    def stop_record_%(name)s(self, int rank):
        cdef list tmp_record = list(proj%(id)s.record_%(name)s)
        cdef int idx = tmp_record.index(rank)
        proj%(id)s.record_%(name)s.erase(proj%(id)s.record_%(name)s.begin() + idx)
        proj%(id)s.record_period_%(name)s.erase(proj%(id)s.record_period_%(name)s.begin() + idx)
        proj%(id)s.record_offset_%(name)s.erase(proj%(id)s.record_offset_%(name)s.begin() + idx)
    def get_recorded_%(name)s(self, int rank):
        cdef vector[vector[%(type)s]] data = proj%(id)s.recorded_%(name)s[rank]
        proj%(id)s.recorded_%(name)s[rank] = vector[vector[%(type)s]]()
        return data
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
                # Pruning in the synapse
                if 'pruning' in proj.synapse.description.keys():
                    code += """
    # Pruning
    def start_pruning(self, int period, long offset):
        proj%(id)s._pruning = True
        proj%(id)s._pruning_period = period
        proj%(id)s._pruning_offset = offset
    def stop_pruning(self):
        proj%(id)s._pruning = False
"""% {'id' : proj.id}
                # Retrieve the names of extra attributes   
                extra_args = ""
                extra_values = ""
                for var in proj.synapse.description['parameters'] + proj.synapse.description['variables']:
                    if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse.description['local']:
                        extra_args += ', ' + var['ctype'] + ' ' +  var['name']   
                        extra_values += ', ' +  var['name']       

                # Generate the code        
                code += """
    # Structural plasticity
    def add_synapse(self, int post_rank, int pre_rank, double weight, int delay%(extra_args)s):
        proj%(id)s.addSynapse(post_rank, pre_rank, weight, delay%(extra_values)s)
    def remove_synapse(self, int post_rank, int pre_rank):
        proj%(id)s.removeSynapse(post_rank, pre_rank)
"""% {'id' : proj.id, 'extra_args': extra_args, 'extra_values': extra_values}


        return code