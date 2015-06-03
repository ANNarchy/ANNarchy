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
from ANNarchy.core.PopulationView import PopulationView
import ProjectionTemplate as ProjTemplate

class ProjectionGenerator(object):

    def __init__(self):
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            self._prof_gen = ProfileGenerator(Global._network[0]['populations'], Global._network[0]['projections'])
        else:
            self._prof_gen = None

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, proj, annarchy_dir):
        """
        Generate and store the projection code in a single header file. The name is defined as
        proj%(id)s.hpp.

        Parameters:

            proj: Projection object
            annarchy_dir: working directory

        Returns:

            (str, str): include directive, pointer definition

        Templates:

            header_struct: basic template

        """
        # Is it a specific projection?
        if proj.generator['omp']['header_proj_struct']:
            final_code = proj.generator['omp']['header_proj_struct']
            init = ""
            update = ""

        else:
            # create the code for non-specific projections
            decl = ""

            # Spiking neurons have aditional data
            if proj.synapse.type == 'spike':
                # Inverse ranks
                decl += """
        std::map< int, std::vector< std::pair<int, int> > > inv_rank ;
    """

            # Exact integration
            has_exact = False
            for var in proj.synapse.description['variables']:
                if var['method'] == 'exact':
                    has_exact = True

            # Delays
            has_delay = ( proj.max_delay > 1 and proj.uniform_delay == -1)

            # Code for declarations and accessors
            accessor = ""
            # Parameters
            for var in proj.synapse.description['parameters']:
                decl += ProjTemplate.attribute_decl[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter' }
                accessor += ProjTemplate.attribute_acc[var['locality']]% {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter' }

            # Variables
            for var in proj.synapse.description['variables']:
                decl += ProjTemplate.attribute_decl[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable' }
                accessor += ProjTemplate.attribute_acc[var['locality']]% {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable' }

            # Arrays for the random numbers
            decl += """
    // Random numbers
"""
            for rd in proj.synapse.description['random_distributions']:
                decl += """    std::vector< std::vector<double> > %(rd_name)s;
    %(template)s dist_%(rd_name)s;
""" % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}

            # Local functions
            if len(proj.synapse.description['functions'])>0:
                decl += """
    // Local functions
"""
                for func in proj.synapse.description['functions']:
                    decl += ' '*4 + func['cpp'] + '\n'

            # Structural plasticity
            if Global.config['structural_plasticity']:
                decl += self.header_structural_plasticity(proj)

            # Definiton of synaptic equations, initialization
            init = self.init_projection(proj).replace("proj"+str(proj.id)+".", "") #TODO: adjust prefix in parser
            update = self.update_synapse(proj).replace("proj"+str(proj.id)+".", "") #TODO: adjust prefix in parser

            if proj.synapse.type == 'rate':
                psp_prefix = "int nb_post;\ndouble sum;"
                psp = self.computesum_rate(proj).replace("proj"+str(proj.id)+".", "") #TODO: adjust prefix in parser
            else:
                psp_prefix = "int nb_post, i, j, rk_j;\ndouble sum;"
                psp = self.computesum_spiking(proj).replace("proj"+str(proj.id)+".", "") #TODO: adjust prefix in parser

            final_code = ProjTemplate.header_struct % { 'id_proj': proj.id,
                                                        'pre_name': proj.pre.name,
                                                        'pre_id': proj.pre.id,
                                                        'post_name': proj.post.name,
                                                        'post_id': proj.post.id,
                                                        'target': proj.target,
                                                        'exact': ProjTemplate.exact_integ['header_struct'] % {'id': proj.id} if has_exact else "",
                                                        'delay': ProjTemplate.delay['header_struct'] % {'id': proj.id} if has_delay else "",
                                                        'decl': decl,
                                                        'accessor': accessor,
                                                        'init': init,
                                                        'psp_prefix': psp_prefix,
                                                        'psp': psp,
                                                        'update': update
                                                     }

        with open(annarchy_dir+'/generate/proj'+str(proj.id)+'.hpp', 'w') as ofile:
            ofile.write(final_code)

        proj_desc = {
            'include': """#include "proj%(id)s.hpp"\n""" % { 'id': proj.id },
            'extern': """extern ProjStruct%(id)s proj%(id)s;\n"""% { 'id': proj.id },
            'instance': """ProjStruct%(id)s proj%(id)s;\n"""% { 'id': proj.id },
            'update': "" if update=="" else """    proj%(id)s.update_synapse();\n""" % { 'id': proj.id }
        }

        return proj_desc

    def recorder_class(self, proj):
        """
        Generate the code for the recorder object.

        Templates:

            record
        """
        tpl_code = ProjTemplate.record
        init_code = ""
        recording_code = ""
        struct_code = ""

        for var in proj.synapse.description['variables']:
            struct_code += tpl_code[var['locality']]['struct'] % {'type' : var['ctype'], 'name': var['name']}
            init_code += tpl_code[var['locality']]['init'] % {'type' : var['ctype'], 'name': var['name']}
            recording_code += tpl_code[var['locality']]['recording'] % {'id': proj.id, 'type' : var['ctype'], 'name': var['name']}

        return tpl_code['struct'] % {'id': proj.id, 'init_code': init_code, 'recording_code': recording_code, 'struct_code': struct_code}

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
            if proj.max_delay > 1: # There is non-zero delay
                if isinstance(proj.pre, PopulationView):
                    delayed_variables = proj.pre.population.delayed_variables
                else:
                    delayed_variables = proj.pre.delayed_variables
                if proj.uniform_delay == -1 : # Non-uniform delays
                    for var in delayed_variables:
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
                    for var in delayed_variables:
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
            if proj.max_delay > 1: # there is a delay
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
            else: # No delay
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
            pGen = ProfileGenerator(Global._network[0]['populations'], Global._network[0]['projections'])

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
        omp_code = ""
        if Global.config['num_threads']>1:
            if proj.post.size > Global.OMP_MIN_NB_NEURONS and (len(pre_event_list) > 0 or learning != ""):
                omp_code = """#pragma omp parallel for firstprivate(nb_post, proj%(id_proj)s_inv_post) private(i, j)"""%{'id_proj' : proj.id}  
            

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
                continue
            if var['name'] in proj.synapse.description['local']:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Local variable %(name)s
    proj%(id)s.%(name)s = std::vector< std::vector<%(type)s> >(proj%(id)s.post_rank.size(), std::vector<%(type)s>());
""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            else:
                init = 0.0 if var['ctype'] == 'double' else 0
                code += """
    // Global variable %(name)s
    proj%(id)s.%(name)s = std::vector<%(type)s>(proj%(id)s.post_rank.size(), %(init)s);
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


#######################################################################
############## PYX ####################################################
#######################################################################

    def pyx_struct(self, proj):
        """
        The python extension wrapper needs a definition of the corresponding
        C object. The pyx_struct contains all methods, which should be accessible
        by the python extension wrapper.

        Templates:

            attribute_cpp_export: normal accessors for variables/parameters
            structural_plasticity: pruning, creating, calling method
            delay, exact_integ: variables accessed by the wrapper

        """
        # Is it a specific projection?
        if proj.generator['omp']['pyx_proj_struct']:
            return proj.generator['omp']['pyx_proj_struct']

        # Check for exact intgeration
        has_exact = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'exact':
                has_exact = True
                break;

        # Check if we need delay code
        has_delay = (proj.max_delay > 1 and proj.uniform_delay == -1)

        # Exact integration
        has_exact = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'exact':
                has_exact = True

        # Determine all export methods
        export = ""
        # Parameters
        for var in proj.synapse.description['parameters']:
            export += ProjTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter'}
        # Variables
        for var in proj.synapse.description['variables']:
            export += ProjTemplate.attribute_cpp_export[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable'}

        # Structural plasticity
        structural_plasticity = ""
        sp_tpl = ProjTemplate.structural_plasticity['pyx_struct']
        if Global.config['structural_plasticity']:
            # Pruning in the synapse
            if 'pruning' in proj.synapse.description.keys():
                structural_plasticity += sp_tpl['pruning']
            if 'creating' in proj.synapse.description.keys():
                structural_plasticity += sp_tpl['creating']

            # Retrieve the names of extra attributes   
            extra_args = ""
            for var in proj.synapse.description['parameters'] + proj.synapse.description['variables']:
                if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse.description['local']:
                    extra_args += ', ' + var['ctype'] + ' ' +  var['name'] 
            # Generate the code
            structural_plasticity += sp_tpl['func'] % {'extra_args': extra_args}

        return ProjTemplate.pyx_struct % { 'id_proj': proj.id,
                                           'exact': ProjTemplate.exact_integ['decl'] % {'id': proj.id} if has_exact else "",
                                           'delay': ProjTemplate.delay['decl'] % {'id': proj.id} if has_delay else "",
                                           'export': export,
                                           'structural_plasticity': structural_plasticity
                                          }

    def pyx_wrapper(self, proj):
        """
        Generates the python extension wrapper, which allows access from Python to the C module. There
        are three optional parts (structural plasticity, non-uniform delays and exact integration of
        synaptic events) which we need to handle seperatly. The rest of the variables/parameters is 
        handled by the standard accessors.

        Templates:

            attribute_pyx_wrapper: normal accessors for variables/parameters
            structural_plasticity: pruning, creating, calling method
            delay, exact_integ: __cinit__ code

        """
        # Is it a specific population?
        if proj.generator['omp']['pyx_proj_class']:
            return  proj.generator['omp']['pyx_proj_class'] 

        # Check for exact intgeration
        has_exact = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'exact':
                has_exact = True
                break;

        # Check if we need delay code
        has_delay = (proj.max_delay > 1 and proj.uniform_delay == -1)

        # Determine all accessor methods
        accessor = ""
        pyx_acc_tpl = ProjTemplate.attribute_pyx_wrapper
        for var in proj.synapse.description['parameters']:
            accessor += pyx_acc_tpl[var['locality']] % {'id' : proj.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'parameter'}
        for var in proj.synapse.description['variables']:
            accessor += pyx_acc_tpl[var['locality']] % {'id' : proj.id, 'name': var['name'], 'type': var['ctype'], 'attr_type': 'variable'}

        # Structural plasticity
        structural_plasticity = ""
        sp_tpl = ProjTemplate.structural_plasticity['pyx_wrapper']
        if Global.config['structural_plasticity']:
            # Pruning in the synapse
            if 'pruning' in proj.synapse.description.keys():
                structural_plasticity += sp_tpl['pruning'] % {'id' : proj.id}

            # Creating in the synapse
            if 'creating' in proj.synapse.description.keys():
                structural_plasticity += sp_tpl['creating'] % {'id' : proj.id}

            # Retrieve the names of extra attributes   
            extra_args = ""
            extra_values = ""
            for var in proj.synapse.description['parameters'] + proj.synapse.description['variables']:
                if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse.description['local']:
                    extra_args += ', ' + var['ctype'] + ' ' +  var['name']   
                    extra_values += ', ' +  var['name']       

            # Generate the code        
            structural_plasticity += sp_tpl['func'] % {'id' : proj.id, 'extra_args': extra_args, 'extra_values': extra_values}

        return ProjTemplate.pyx_wrapper % { 'id': proj.id,
                                            'exact_init': ProjTemplate.exact_integ['cinit'] % {'id': proj.id} if has_exact else "",
                                            'delay_init': ProjTemplate.delay['cinit'] % {'id': proj.id} if has_delay else "",
                                            'delay_acc': ProjTemplate.delay['pyx_wrapper_acc'] % {'id': proj.id} if has_delay else "",
                                            'accessor': accessor,
                                            'structural_plasticity': structural_plasticity
                                           }


#######################################################################
############## Recording ##############################################
#######################################################################

    def pyx_monitor_struct(self, proj):
        tpl_code = """
    # Projection %(id)s : Monitor
    cdef cppclass ProjRecorder%(id)s (Monitor):
        ProjRecorder%(id)s(vector[int], int, long) except +    
"""
        for var in proj.synapse.description['variables']:
            if var['name'] in proj.synapse.description['local']:
                tpl_code += """
        vector[vector[%(type)s]] %(name)s
        bool record_%(name)s""" % {'name': var['name'], 'type': var['ctype']}
            elif var['name'] in proj.synapse.description['global']:
                tpl_code += """
        vector[%(type)s] %(name)s
        bool record_%(name)s""" % {'name': var['name'], 'type': var['ctype']}


        return tpl_code % {'id' : proj.id}

    def pyx_monitor_wrapper(self, proj):
        tpl_code = """
# Projection Monitor wrapper
cdef class ProjRecorder%(id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, long offset):
        self.thisptr = new ProjRecorder%(id)s(ranks, period, offset)
"""

        for var in proj.synapse.description['variables']:
            tpl_code += """
    property %(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).%(name)s = val 
    property record_%(name)s:
        def __get__(self): return (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s
        def __set__(self, val): (<ProjRecorder%(id)s *>self.thisptr).record_%(name)s = val 
    def clear_%(name)s(self):
        (<ProjRecorder%(id)s *>self.thisptr).%(name)s.clear()""" % {'id' : proj.id, 'name': var['name']}

        return tpl_code % {'id' : proj.id}

#######################################################################
############## Structural plasticity ##################################
#######################################################################

    def header_structural_plasticity(self, proj):
        """
        Generate extension code for C header_struct: variable declaration, add and remove synapses.

        Templates:

            structural_plasticity: 'header_struct' field contains all relevant code templates

        """
        header_tpl = ProjTemplate.structural_plasticity['header_struct']

        code = ""
        # Pruning defined in the synapse
        if 'pruning' in proj.synapse.description.keys():
            code += header_tpl['pruning']

        # Creating defined in the synapse
        if 'creating' in proj.synapse.description.keys():
            code += header_tpl['creating']

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
        spiking_addcode = "" if proj.synapse.type == 'rate' else header_tpl['spiking_addcode']
        spiking_removecode = "" if proj.synapse.type == 'rate' else header_tpl['spiking_removecode']

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
        code += header_tpl['header'] % { 'extra_args': extra_args, 'delay_code': delay_code,
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