"""

    ProjectionGenerator.py

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
import ANNarchy.core.Global as Global

class ProjectionGenerator(object):

    def __init__(self):
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            self._prof_gen = ProfileGenerator(Global._populations, Global._projections)
        else:
            self._prof_gen = None

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, proj):
        # Is it a specific projection?
        if proj.generator['omp']['header_proj_struct']:
            return proj.generator['omp']['header_proj_struct']

        # create the code for non-specific projections
        code = """
// %(pre_name)s -> %(post_name)s
struct ProjStruct%(id_proj)s{
    // number of dendrites
    int size;

    // Learning flag
    bool _learning;

    // Connectivity
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;
"""

        # Spiking neurons have aditional data
        if proj.synapse.type == 'spike':
            # Inverse ranks
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
    std::vector< std::vector< %(type)s > > %(name)s;
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
    // Global parameter %(name)s
    std::vector<%(type)s>  %(name)s;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s;
    std::vector< std::vector< std::vector< %(type)s > > > recorded_%(name)s;
    std::vector< int > record_%(name)s;
    std::vector< int > record_period_%(name)s;
    std::vector< long int > record_offset_%(name)s;
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in proj.synapse.description['global']:
                code += """
    // Global variable %(name)s
    std::vector<%(type)s>  %(name)s;
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
            code += self.header_structural_plasticity(proj)

        # Finish the structure
        code += """
};    
""" 
        return code % {'id_proj': proj.id, 'pre_name': proj.pre.name, 'post_name': proj.post.name}



#######################################################################
############## BODY ###################################################
#######################################################################


    def computesum_rate(self, proj):
        code = ""    
        ids = {'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

        # Retrieve the psp code
        if Global.config['num_threads'] == 1 or proj.post.size <= Global.OMP_MIN_NB_NEURONS: # No need to copy the pre-synaptic firing rates
            if not 'psp' in  proj.synapse.description.keys(): # default
                psp = """proj%(id_proj)s.w[i][j] * pop%(id_pre)s.r[rk_pre];""" % ids
            else: # custom psp
                psp = (proj.synapse.description['psp']['cpp'] % ids)
            # Take delays into account if any
            if proj.max_delay > 1:
                if proj.uniform_delay == -1 : # Non-uniform delays
                    for var in proj.pre.delayed_variables:
                        if var in proj.pre.neuron_type.description['local']:
                            psp = psp.replace(
                                'pop%(id_pre)s.%(var)s[rk_pre]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[proj%(id_proj)s.delay[i][j]-1][rk_pre]'% dict({'var': var}.items() + ids.items())
                            )
                        else:
                            psp = psp.replace(
                                'pop%(id_pre)s.%(var)s[rk_pre]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[proj%(id_proj)s.delay[i][j]-1]'% dict({'var': var}.items() + ids.items())
                            )

                else: # Uniform delays
                    for var in proj.pre.delayed_variables:
                        if var in proj.pre.neuron_type.description['local']:
                            psp = psp.replace(
                                'pop%(id_pre)s.%(var)s[rk_pre]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[%(delay)s][rk_pre]' % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            )
                        else:
                            psp = psp.replace(
                                'pop%(id_pre)s.%(var)s[rk_pre]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[%(delay)s]' % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            )

            psp = psp.replace('[rk_pre]', '[proj%(id_proj)s.pre_rank[i][j]]'% ids)
            omp_code = ""
            pre_copy = ""

        else: # OMP: make a local copy of pre.r for each thread
            if not 'psp' in  proj.synapse.description.keys(): # default
                psp = """proj%(id_proj)s.w[i][j] * pop%(id_pre)s.r[rk_pre];""" % ids
            else: # custom psp
                psp = (proj.synapse.description['psp']['cpp'] % ids)
            # Take delays into account if any
            if proj.max_delay > 1:
                if proj.uniform_delay == -1 : # Non-uniform delays
                    for var in list(set(proj.synapse.description['dependencies']['pre'])):
                        if var in proj.pre.neuron_type.description['local']:
                            psp = psp.replace('pop%(id_pre)s.%(var)s[rk_pre]'% {'id_pre': proj.pre.id, 'var': var}, 'pop%(id_pre)s._delayed_%(var)s[proj%(id_proj)s.delay[i][j]-1][rk_pre]' % {'id_pre': proj.pre.id, 'id_proj' : proj.id, 'var': var})
                        else:
                            Global._error('The psp accesses a global variable with a non-uniform delay!')
                            Global._print(proj.synapse.description['psp']['eq'])
                            exit(0)
                    pre_copy = ""
                    omp_code = '#pragma omp parallel for private(sum) firstprivate(nb_post) schedule(dynamic)'% ids 
                else: # Uniform delays
                    pre_copy = ""; omp_code = "#pragma omp parallel for private(sum) firstprivate("
                    for var in list(set(proj.synapse.description['dependencies']['pre'])):
                        if var in proj.pre.neuron_type.description['local']:
                            pre_copy += """        std::vector<double> proj%(id_proj)s_pre_%(var)s = pop%(id_pre)s._delayed_%(var)s[%(delay)s];\n""" %{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            psp = psp.replace('pop%(id_pre)s.%(var)s[rk_pre]'% {'id_pre': proj.pre.id, 'var': var}, 'proj%(id_proj)s_pre_%(var)s[rk_pre]' % {'id_proj': proj.id, 'var': var})
                        else:
                            pre_copy += """        double proj%(id_proj)s_pre_%(var)s = pop%(id_pre)s._delayed_%(var)s[%(delay)s];\n""" % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            psp = psp.replace('pop%(id_pre)s.%(var)s[rk_pre]'% {'id_pre': proj.pre.id, 'var': var}, 'proj%(id_proj)s_pre_%(var)s' % {'id_proj': proj.id, 'var': var})

                        omp_code += 'proj%(id_proj)s_pre_%(var)s, ' % {'id_proj': proj.id, 'var': var}

                    omp_code += "nb_post) schedule(dynamic)"
            else:
                pre_copy = ""; omp_code = "#pragma omp parallel for private(sum) firstprivate("
                for var in list(set(proj.synapse.description['dependencies']['pre'])):
                    if var in proj.pre.neuron_type.description['local']:
                        pre_copy += """        std::vector<double> proj%(id_proj)s_pre_%(var)s = pop%(id_pre)s.%(var)s;\n""" %{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        psp = psp.replace('pop%(id_pre)s.%(var)s[rk_pre]'% {'id_pre': proj.pre.id, 'var': var}, 'proj%(id_proj)s_pre_%(var)s[rk_pre]' % {'id_proj': proj.id, 'var': var})
                    else:
                        pre_copy += """        double proj%(id_proj)s_pre_%(var)s = pop%(id_pre)s.%(var)s;\n""" % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        psp = psp.replace('pop%(id_pre)s.%(var)s[rk_pre]'% {'id_pre': proj.pre.id, 'var': var}, 'proj%(id_proj)s_pre_%(var)s' % {'id_proj': proj.id, 'var': var})

                    omp_code += 'proj%(id_proj)s_pre_%(var)s, ' % {'id_proj': proj.id, 'var': var}

                omp_code += "nb_post) schedule(dynamic)"
            # Modify the index
            psp = psp.replace('[rk_pre]', '[proj%(id_proj)s.pre_rank[i][j]]'% ids)
        
        # Generate the code depending on the operation
        if proj.synapse.operation == 'sum': # normal summation
            code+= """
%(pre_copy)s
        nb_post = proj%(id_proj)s.post_rank.size();
        %(omp_code)s
        for(int i = 0; i < nb_post; i++) {
            sum = 0.0;
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++) {
                sum += %(psp)s
            }
            pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
        }
"""%{ 'id_proj' : proj.id, 'target': proj.target, 
      'id_post': proj.post.id, 'id_pre': proj.pre.id, 
      'name_post': proj.post.name, 'name_pre': proj.pre.name, 
      'psp': psp, 
      'omp_code': omp_code,
      'pre_copy': pre_copy
    }

        elif proj.synapse.operation == 'max': # max pooling
            code+= """
        %(pre_copy)s
        nb_post = proj%(id_proj)s.post_rank.size();
        %(omp_code)s
        for(int i = 0; i < nb_post; i++){
            int j = 0;
            sum = %(psp)s ;
            for(int j = 1; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                if(%(psp)s > sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
        }
"""%{'id_proj' : proj.id, 'target': proj.target, 
     'id_post': proj.post.id, 'id_pre': proj.pre.id, 
     'psp': psp.replace(';', ''), 'omp_code': omp_code,
     'pre_copy': pre_copy}

        elif proj.synapse.operation == 'min': # max pooling
            code+= """
        %(pre_copy)s
        nb_post = proj%(id_proj)s.post_rank.size();
        %(omp_code)s
        for(int i = 0; i < nb_post; i++){
            int j= 0;
            sum = %(psp)s ;
            for(int j = 1; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                if(%(psp)s < sum){
                    sum = %(psp)s ;
                }
            }
            pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum;
        }
"""%{'id_proj' : proj.id, 'target': proj.target, 
    'id_post': proj.post.id, 'id_pre': proj.pre.id, 
    'psp': psp.replace(';', ''), 'omp_code': omp_code,
     'pre_copy': pre_copy}

        elif proj.synapse.operation == 'mean': # max pooling
            code+= """
        %(pre_copy)s
        nb_post = proj%(id_proj)s.post_rank.size();
        %(omp_code)s
        for(int i = 0; i < nb_post; i++){
            sum = 0.0 ;
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                sum += %(psp)s ;
            }
            pop%(id_post)s._sum_%(target)s[proj%(id_proj)s.post_rank[i]] += sum / (double)(proj%(id_proj)s.pre_rank[i].size());
        }
"""%{ 'id_proj' : proj.id, 'target': proj.target,
      'id_post': proj.post.id, 'id_pre': proj.pre.id,
      'psp': psp.replace(';', ''), 'omp_code': omp_code,
      'pre_copy': pre_copy }

        # Profiling code
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            pGen = ProfileGenerator(Global._populations, Global._projections)

        # annotate code
        if self._prof_gen:
            code = self._prof_gen.annotate_computesum_rate_omp(code)

        # finish the code
        code = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. operation = mean
    if (pop%(id_post)s._active){
%(code)s
    } // active
        """ % {'id_proj' : proj.id, 'target': proj.target,
               'name_post': proj.post.name, 'name_pre': proj.pre.name,
               'id_post': proj.post.id,
               'code': code,
               }

        return code

    def computesum_spiking(self, proj):

        ids = {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target} 

        # Analyse all elements of pre_spike
        pre_event = ""
        pre_event_list = []
        learning = ""
        psp = ""; psp_bounds = ""

        for eq in proj.synapse.description['pre_spike']:
            if eq['name'] == 'w':
                bounds = ""                
                for line in get_bounds(eq).splitlines():
                    bounds += ' ' * 24 + line % ids + '\n'
                learning = """                    if(proj%(id_proj)s._learning){
                        %(eq)s 
%(bounds)s
                    }
""" % {'id_proj' : proj.id, 'eq': eq['cpp'] % ids, 'bounds': bounds % ids}
            elif eq['name'] == 'g_target':
                psp = eq['cpp'].split('=')[1]
                for key, val in eq['bounds'].items():
                    try:
                        value = str(float(val))
                    except:
                        value = "proj%(id_proj)s.%(name)s%(locality)s" % {'id_proj' : proj.id, 'name': val, 'locality': '[i]' if val in proj.synapse.description['global'] else '[i][j]'}
                    psp_bounds += """
                if (pop%(id_post)s.g_%(target)s[proj%(id_proj)s.post_rank[i]] %(op)s %(val)s)
                    pop%(id_post)s.g_%(target)s[proj%(id_proj)s.post_rank[i]] = %(val)s;
""" % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target, 'op': "<" if key == 'min' else '>', 'val': value }
            else:
                pre_event_list.append(eq)

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
                // Exact integration of synaptic variables
                %(exact)s
""" % {'exact': var['cpp'].replace('(t)', '(t-1)') %{'id_proj' : proj.id}}
        if has_exact:
                event_based = False # to avoid the if not post.spike
                exact_code += """
                // Update the last event for the synapse 
                proj%(id_proj)s._last_event[i][j] = t;
""" % {'id_proj' : proj.id, 'exact': var['cpp']}

            
        # Other event-driven variables
        if len(pre_event_list) > 0 or learning != "": # There are other variables to update than g_target
            code = ""
            for eq in pre_event_list:
                code += ' ' * 20 + eq['cpp'] % ids + '\n'
                for line in get_bounds(eq).splitlines():
                    code += ' ' * 20 + line % ids + '\n'


            if event_based:
                pre_event += """
                // Event-based variables should not be updated when the postsynaptic neuron fires.
                if(pop%(id_post)s.last_spike[proj%(id_proj)s.post_rank[i]] != t-1){
                    // Pre-spike events
%(pre_event)s
                    // Plasticity of w can be disabled
%(learning)s
                }
"""% {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'pre_event': code, 'learning': learning}
            else:
                pre_event += """
                // Pre-spike events with exact integration should always be evaluated...
                {
                    // Pre-spike events
%(pre_event)s
                    // Plasticity of w can be disabled
%(learning)s
                }
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
        if Global.config['num_threads']>1:
            omp_code = """//#pragma omp parallel for firstprivate(nb_post, proj%(id_proj)s_inv_post) private(i, j)"""%{'id_proj' : proj.id} if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        code = """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s. event-based
    if (pop%(id_post)s._active){
        std::vector< std::pair<int, int> > proj%(id_proj)s_inv_post;
        // Iterate over all incoming spikes
        for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++){
            rk_j = %(pre_array)s[_idx_j];
            proj%(id_proj)s_inv_post = proj%(id_proj)s.inv_rank[rk_j];
            nb_post = proj%(id_proj)s_inv_post.size();
            // Iterate over connected post neurons
            %(omp_code)s
            for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                // Retrieve the correct indices
                i = proj%(id_proj)s_inv_post[_idx_i].first;
                j = proj%(id_proj)s_inv_post[_idx_i].second;
%(exact)s
                // Increase the post-synaptic conductance
                %(psp)s
                %(psp_bounds)s
%(pre_event)s
            }
        }
    } // active
"""%{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'name_post': proj.post.name, 'name_pre': proj.pre.name, 'pre_array': pre_array,
    'pre_event': pre_event, 'psp': psp_code , 'psp_bounds': psp_bounds, 'omp_code': omp_code,
    'exact': exact_code }

        # Not even-driven summation of psp
        if 'psp' in  proj.synapse.description.keys(): # not event-based
            if Global.config['num_threads']>1:
                omp_code = """#pragma omp parallel for private(sum)""" if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
            else:
                omp_code = ""
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

        # annotate code
        if self._prof_gen:
            code = self._prof_gen.annotate_computesum_spiking_omp(code)

        return code

    def postevent(self, proj):
        code = ""
        if proj.synapse.description['post_spike'] == []:
            return ""

        post_code = ""

        # Exact integration
        has_exact = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'exact':
                has_exact = True
                post_code += """
                // Exact integration
                %(exact)s
""" % {'exact': var['cpp'] %{'id_proj' : proj.id}}
        if has_exact:
            post_code += """
                // Update the last event for the synapse
                proj%(id_proj)s._last_event[i][j] = t;
""" % {'id_proj' : proj.id, 'exact': var['cpp']}

        # Gather the equations
        post_code += """
                // Post-spike events
"""
        for eq in proj.synapse.description['post_spike']:
            post_code += ' ' * 16 + eq['cpp'] %{'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id} + '\n'
            for line in get_bounds(eq).splitlines():
                post_code += ' ' * 16 + line % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id} + '\n'

        # Generate the code
        if post_code != "":
            if Global.config['num_threads']>1:
                omp_code = '#pragma omp parallel for private(j) firstprivate(i)' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
            else:
                omp_code = ""
            code += """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
    if(proj%(id_proj)s._learning && pop%(id_post)s._active){
        for(int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++){
            i = pop%(id_post)s.spiked[_idx_i];
            // Test if the post neuron has connections in this projection (PopulationView)
            if(std::find(proj%(id_proj)s.post_rank.begin(), proj%(id_proj)s.post_rank.end(), i) == proj%(id_proj)s.post_rank.end())
                continue;
            // Iterate over all synapse to this neuron
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

    def update_synapse(self, proj):
        code = ""
        from ..Utils import generate_equation_code

        # Global variables
        global_eq = generate_equation_code(proj.id, proj.synapse.description, 'global', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

        # Local variables
        local_eq =  generate_equation_code(proj.id, proj.synapse.description, 'local', 'proj') %{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id} 

        # Generate the code
        if local_eq.strip() != '' or global_eq.strip() != '' :
            if Global.config['num_threads']>1:
                omp_code = '#pragma omp parallel for private(rk_pre, rk_post)' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
            else:
                omp_code = ""

            code+= """
        %(omp_code)s
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            rk_post = proj%(id_proj)s.post_rank[i];
%(global)s
"""%{'id_proj' : proj.id, 'global': global_eq, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'name_post': proj.post.name, 'name_pre': proj.pre.name, 'omp_code': omp_code}

            if local_eq.strip() != "": 
                code+= """
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                rk_pre = proj%(id_proj)s.pre_rank[i][j];
                %(local)s
            }
"""%{'id_proj' : proj.id, 'local': local_eq}

            code += """
        }
"""

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1 : # Non-uniform delays
                for var in list(set(proj.synapse.description['dependencies']['pre'])):
                    if var in proj.pre.neuron_type.description['local']:
                        code = code.replace(
                            'pop%(id_pre)s.%(var)s[rk_pre]'%{'id_pre': proj.pre.id, 'var': var}, 
                            'pop%(id_pre)s._delayed_%(var)s[proj%(id_proj)s.delay[i][j]-1][rk_pre]'%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        )
                    else:
                        code = code.replace(
                            'pop%(id_pre)s.%(var)s[rk_pre]'%{'id_pre': proj.pre.id, 'var': var}, 
                            'pop%(id_pre)s._delayed_%(var)s[proj%(id_proj)s.delay[i][j]-1]'%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        )
                code = code.replace(
                    'pop%(id_pre)s.spiked['%{'id_pre': proj.pre.id},
                    'pop%(id_pre)s._delayed_spike[proj%(id_proj)s.delay[i][j]-1]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id}
                )
            else: # Uniform delays
                for var in list(set(proj.synapse.description['dependencies']['pre'])):
                    if var in proj.pre.neuron_type.description['local']:
                        code = code.replace(
                            'pop%(id_pre)s.%(var)s[rk_pre]'%{'id_pre': proj.pre.id, 'var': var}, 
                            'pop%(id_pre)s._delayed_%(var)s[%(delay)s][rk_pre]'%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var, 'delay': str(proj.uniform_delay-1)}
                        )
                    else:
                        code = code.replace(
                            'pop%(id_pre)s.%(var)s[rk_pre]'%{'id_pre': proj.pre.id, 'var': var}, 
                            'pop%(id_pre)s._delayed_%(var)s[%(delay)s]'%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var, 'delay': str(proj.uniform_delay-1)}
                        )
                code = code.replace(
                    'pop%(id_pre)s.spiked['%{'id_pre': proj.pre.id},
                    'pop%(id_pre)s._delayed_spike[%(delay)s]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
                )

        # profiling annotation
        if self._prof_gen:
            code = self._prof_gen.annotate_update_synapse_omp(code)

        # finish the code block
        if code != "":
            return """
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
    if(proj%(id_proj)s._learning && pop%(id_post)s._active){
%(code)s
    } // active
""" % { 'name_pre': proj.pre.name, 'name_post': proj.post.name, 'target': proj.target,
        'id_proj': proj.id, 'id_post': proj.post.id, 'code': code }
        else:
            return ""

    def init_random_distributions(self, proj):
        # Is it a specific population?
        if proj.generator['omp']['body_random_dist_init']:
            return proj.generator['omp']['body_random_dist_init'] %{'id_proj': proj.id}

        code = ""
        for rd in proj.synapse.description['random_distributions']:
            code += """    proj%(id_proj)s.%(rd_name)s = std::vector< std::vector<double> >(proj%(id_proj)s.post_rank.size(), std::vector<double>());
    for(int i=0; i<proj%(id_proj)s.post_rank.size(); i++){
        proj%(id_proj)s.%(rd_name)s[i] = std::vector<double>(proj%(id_proj)s.pre_rank[i].size(), 0.0);
    }
    proj%(id_proj)s.dist_%(rd_name)s = %(rd_init)s;
""" % {'id_proj': proj.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': proj.id}}
        return code

    def update_random_distributions(self, proj):
        # Is it a specific population?
        if proj.generator['omp']['body_random_dist_update']:
            return proj.generator['omp']['body_random_dist_update'] %{'id': pop.id}

        code = ""
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

    def init_projection(self, proj):

        # Is it a specific projection?
        if proj.generator['omp']['body_proj_init']:
            return proj.generator['omp']['body_proj_init']

        # Learning by default
        code = """
    /////////////////////////////////////////
    // proj%(id_proj)s: %(name_pre)s -> %(name_post)s with target %(target)s
    /////////////////////////////////////////
    proj%(id_proj)s._learning = true;
""" % { 'id_proj': proj.id, 'target': proj.target,
        'id_post': proj.post.id, 'id_pre': proj.pre.id,
        'name_post': proj.post.name, 'name_pre': proj.pre.name}

        # Initialize parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] == 'w':
                continue
            if var['name'] in proj.synapse.description['local']:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Local parameter %(name)s
    proj%(id)s.%(name)s = std::vector< std::vector<%(type)s> >(proj%(id)s.post_rank.size(), std::vector<%(type)s>());
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            else:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Global parameter %(name)s
    proj%(id)s.%(name)s = std::vector<%(type)s>(proj%(id)s.post_rank.size(), %(init)s);
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Initialize variables
        for var in proj.synapse.description['variables']:
            if var['name'] == 'w':
                code += """
    // Local variable %(name)s (only the record methods)
    proj%(id)s.record_%(name)s = std::vector<int>();
    proj%(id)s.record_period_%(name)s = std::vector<int>();
    proj%(id)s.record_offset_%(name)s = std::vector<long int>();
    proj%(id)s.recorded_%(name)s = std::vector< std::vector< std::vector< %(type)s > > > (proj%(id)s.post_rank.size());
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype']}
                continue
            if var['name'] in proj.synapse.description['local']:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Local variable %(name)s
    proj%(id)s.%(name)s = std::vector< std::vector<%(type)s> >(proj%(id)s.post_rank.size(), std::vector<%(type)s>());
    proj%(id)s.record_%(name)s = std::vector<int>();
    proj%(id)s.record_period_%(name)s = std::vector<int>();
    proj%(id)s.record_offset_%(name)s = std::vector<long int>();
    proj%(id)s.recorded_%(name)s = std::vector< std::vector< std::vector< %(type)s > > > (proj%(id)s.post_rank.size());
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            else:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Global variable %(name)s
    proj%(id)s.%(name)s = std::vector<%(type)s>(proj%(id)s.post_rank.size(), %(init)s);
    proj%(id)s.record_%(name)s = std::vector<int>();
    proj%(id)s.record_period_%(name)s = std::vector<int>();
    proj%(id)s.record_offset_%(name)s = std::vector<long int>();
    proj%(id)s.recorded_%(name)s = std::vector< std::vector< %(type)s > > (proj%(id)s.post_rank.size(), std::vector< %(type)s >());
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

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
        # Pruning
        if Global.config['structural_plasticity']:
            if 'pruning' in proj.synapse.description.keys():
                code +="""
    // Pruning
    proj%(id_proj)s._pruning = false;
    proj%(id_proj)s._pruning_period = 1;
    proj%(id_proj)s._pruning_offset = 0;
"""% {'id_proj': proj.id}
            if 'creating' in proj.synapse.description.keys():
                code +="""
    // Creating
    proj%(id_proj)s._creating = false;
    proj%(id_proj)s._creating_period = 1;
    proj%(id_proj)s._creating_offset = 0;
"""% {'id_proj': proj.id}

        return code

    def record(self, proj):
        code = ""
        for var in proj.synapse.description['variables']:
            code += """
    for(int i=0; i< proj%(id)s.record_%(name)s.size(); i++){
        int rk = proj%(id)s.record_%(name)s[i];
        if((t - proj%(id)s.record_offset_%(name)s[i]) %(modulo)s proj%(id)s.record_period_%(name)s[i] == 0)
            proj%(id)s.recorded_%(name)s[rk].push_back(proj%(id)s.%(name)s[rk]) ;
    }
""" % {'id': proj.id, 'name': var['name'], 'modulo': '%'}
        return code


#######################################################################
############## PYX ####################################################
#######################################################################

    def pyx_struct(self, proj):
        # Is it a specific projection?
        if proj.generator['omp']['pyx_proj_struct']:
            return proj.generator['omp']['pyx_proj_struct']

        # create the code for non-specific projections
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
            if 'creating' in proj.synapse.description.keys():
                code += """
        # Creating
        bool _creating
        int _creating_period
        long _creating_offset
"""

            # Retrieve the names of extra attributes   
            extra_args = ""
            for var in proj.synapse.description['parameters'] + proj.synapse.description['variables']:
                if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse.description['local']:
                    extra_args += ', ' + var['ctype'] + ' ' +  var['name'] 
            # Generate the code
            code += """
        # Structural plasticity
        int dendrite_index(int post, int pre)
        void addSynapse(int post, int pre, double weight, int _delay%(extra_args)s)
        void removeSynapse(int post, int pre)
""" % {'extra_args': extra_args}

        # Finalize the code
        return code % {'id_proj': proj.id}

    def pyx_wrapper(self, proj):

        # Is it a specific population?
        if proj.generator['omp']['pyx_proj_class']:
            return  proj.generator['omp']['pyx_proj_class'] 

        # Init
        code = """
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
            # Creating in the synapse
            if 'creating' in proj.synapse.description.keys():
                code += """
    # Creating
    def start_creating(self, int period, long offset):
        proj%(id)s._creating = True
        proj%(id)s._creating_period = period
        proj%(id)s._creating_offset = offset
    def stop_creating(self):
        proj%(id)s._creating = False
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
        proj%(id)s.removeSynapse(post_rank, proj%(id)s.dendrite_index(post_rank, pre_rank))
"""% {'id' : proj.id, 'extra_args': extra_args, 'extra_values': extra_values}

        return code



#######################################################################
############## Structural plasticity ##################################
#######################################################################

    def header_structural_plasticity(self, proj):
        code = ""
        # Pruning defined in the synapse
        if 'pruning' in proj.synapse.description.keys():
            code += """
    // Pruning
    bool _pruning;
    int _pruning_period;
    long int _pruning_offset;
"""
        # Creating defined in the synapse
        if 'creating' in proj.synapse.description.keys():
            code += """
    // Creating
    bool _creating;
    int _creating_period;
    long int _creating_offset;
"""
        # Retrieve the names of extra attributes   
        extra_args = ""
        add_code = ""
        remove_code = ""
        for var in proj.synapse.description['parameters'] + proj.synapse.description['variables']:
            if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse.description['local']:

                if not isinstance(proj.init[var['name']], (int, float, bool)):
                    init = var['init']
                else:
                    init = proj.init[var['name']]
                extra_args += ', ' + var['ctype'] + ' _' +  var['name'] +'='+str(init)
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
        int pre = pre_rank[post][idx];
        for(int i=0; i<inv_rank[pre].size(); i++){
            if(inv_rank[pre][i].second == idx){
                inv_rank[pre].erase(inv_rank[pre].begin() + i);
                break;
            }
        }
"""

        # Randomdistributions
        rd_addcode = ""; rd_removecode = ""
        for rd in proj.synapse.description['random_distributions']:
            rd_addcode += """
        %(name)s[post].insert(%(name)s[post].begin() + idx, 0.0);
""" % {'name': rd['name']}
            
            rd_removecode += """
        %(name)s[post].erase(%(name)s[post].begin() + idx);
""" % {'name': rd['name']}
            


        # Generate the code
        code += """
    // Structural plasticity
    int dendrite_index(int post, int pre){
        int idx = 0;
        for(int i=0; i<pre_rank[post].size(); i++){
            if(pre_rank[post][i] == pre){
                idx = i;
                break;
            }
        }
        return idx;
    }
    void addSynapse(int post, int pre, double weight, int _delay=0%(extra_args)s){
        // Find where to put the synapse
        int idx = pre_rank[post].size();
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
%(rd_add)s
    };
    void removeSynapse(int post, int idx){
        pre_rank[post].erase(pre_rank[post].begin() + idx);
        w[post].erase(w[post].begin() + idx);
%(remove_code)s  
%(spike_remove)s
%(rd_remove)s
    };
""" % { 'extra_args': extra_args, 'delay_code': delay_code, 
        'add_code': add_code, 'remove_code': remove_code,
        'spike_add': spiking_addcode, 'spike_remove': spiking_removecode,
        'rd_add': rd_addcode, 'rd_remove': rd_removecode
        }

        return code


    def creating(self, proj):
        creating_structure = proj.synapse.description['creating']

        # Random stuff
        proba = ""; proba_init = ""
        if 'proba' in creating_structure['bounds'].keys():
            val = creating_structure['bounds']['proba']
            proba += '&&(unif(rng)<' + val + ')'
            proba_init += "std::uniform_real_distribution<double> unif(0.0, 1.0);"
        if  creating_structure['rd']:
            proba_init += "\n        " +  creating_structure['rd']['template'] + ' rd(' + creating_structure['rd']['args'] + ');'

        # delays
        delay = ""
        if 'd' in creating_structure['bounds'].keys():
            d = int(creating_structure['bounds']['delay']/Global.config['dt'])
            if proj.max_delay > 1 and proj.uniform_delay == -1:
                if d > proj.max_delay:
                    Global._error('creating: you can not add a delay higher than the maximum of existing delays')
                    exit(0)
                delay = ", " + str(d)
            else:
                if d != proj.uniform_delay:
                    Global._error('creating: you can not add a delay different from the others if they were constant.')
                    exit(0)



        # OMP
        if Global.config['num_threads']>1:
            omp_code = '#pragma omp parallel for' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        creating = """
    // proj%(id_proj)s creating: %(eq)s
    if((proj%(id_proj)s._creating)&&((t - proj%(id_proj)s._creating_offset) %(modulo)s proj%(id_proj)s._creating_period == 0)){
        %(proba_init)s
        //%(omp_code)s
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            rk_post = proj%(id_proj)s.post_rank[i];
            for(int rk_pre = 0; rk_pre < pop%(id_pre)s.size; rk_pre++){
                if(%(condition)s){
                    // Check if the synapse exists
                    bool _exists = false;
                    for(int k=0; k<proj%(id_proj)s.pre_rank[i].size(); k++){
                        if(proj%(id_proj)s.pre_rank[i][k] == rk_pre){
                            _exists = true;
                            break;
                        }
                    }
                    if((!_exists)%(proba)s){
                        std::cout << "Creating synapse between " << rk_pre << " and " << rk_post << std::endl;
                        proj%(id_proj)s.addSynapse(i, rk_pre, %(weights)s%(delay)s);

                    }
                }
            }
        }
    }
""" % { 'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 
        'eq': creating_structure['eq'], 'modulo': '%', 
        'condition': creating_structure['cpp'] % {'id_proj' : proj.id, 'target': proj.target, 
        'id_post': proj.post.id, 'id_pre': proj.pre.id},
        'omp_code': omp_code,
        'weights': 0.0 if not 'w' in creating_structure['bounds'].keys() else creating_structure['bounds']['w'],
        'proba' : proba, 'proba_init': proba_init,
        'delay': delay
        }
        
        return creating

    def pruning(self, proj):
        pruning_structure = proj.synapse.description['pruning']


        proba = ""; proba_init = ""
        if 'proba' in pruning_structure['bounds'].keys():
            val = pruning_structure['bounds']['proba']
            proba = '&&(unif(rng)<' + val + ')'
            proba_init = "std::uniform_real_distribution<double> unif(0.0, 1.0);"
        if pruning_structure['rd']:
            proba_init += "\n        " +  pruning_structure['rd']['template'] + ' rd(' + pruning_structure['rd']['args'] + ');'

        if Global.config['num_threads']>1:
            omp_code = '#pragma omp parallel for' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        pruning = """
    // proj%(id_proj)s pruning: %(eq)s
    if((proj%(id_proj)s._pruning)&&((t - proj%(id_proj)s._pruning_offset) %(modulo)s proj%(id_proj)s._pruning_period == 0)){
        %(proba_init)s
        //%(omp_code)s
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            rk_post = proj%(id_proj)s.post_rank[i];
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                rk_pre = proj%(id_proj)s.pre_rank[i][j];
                if((%(condition)s)%(proba)s){
                    proj%(id_proj)s.removeSynapse(i, j);
                }
            }
        }
    }
""" % { 'id_proj' : proj.id, 'eq': pruning_structure['eq'], 'modulo': '%', 
        'condition': pruning_structure['cpp'] % {'id_proj' : proj.id, 'target': proj.target, 
        'id_post': proj.post.id, 'id_pre': proj.pre.id},
        'omp_code': omp_code,
        'proba' : proba, 'proba_init': proba_init
        }
        
        return pruning

######################################
### Code generation
######################################
def get_bounds(param):
    "Analyses the bounds of a variable and returns the corresponding code."
    from ...parser.SingleAnalysis import pattern_omp as pattern
    code = ""
    # Min-Max bounds
    for bound, val in param['bounds'].items():
        if bound == "init":
            continue

        code += """if(%(obj)s%(sep)s%(var)s%(index)s %(operator)s %(val)s)
    %(obj)s%(sep)s%(var)s%(index)s = %(val)s;
""" % {'obj': pattern['proj_prefix'],
       'sep': pattern['proj_sep'],
       'index': pattern['proj_index'],
       'var' : param['name'], 'val' : val, 'id': id, 
       'operator': '<' if bound=='min' else '>'
       }
    return code