#===============================================================================
#
#     OpenMPGenerator.py
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
from .ProjectionGenerator import ProjectionGenerator, get_bounds
from .OpenMPTemplates import openmp_templates
from .Connectivity import OpenMPConnectivity

from ANNarchy.core import Global
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.generator.Utils import generate_equation_code, tabify

import re
from ANNarchy.generator.Projection import OpenMPTemplates

class OpenMPGenerator(ProjectionGenerator, OpenMPConnectivity):
    """
    Generate the header for a Population object to run either on single core
    or multi-cores with OpenMP.
    """
    _templates = openmp_templates

    def __init__(self, profile_generator, net_id):
        # The super here calls all the base classes, so first
        # ProjectionGenerator and afterwards OpenMPConnectivity
        super(OpenMPGenerator, self).__init__(profile_generator, net_id)

    def header_struct(self, proj, annarchy_dir):
        """
        Generate the projection header for a given projection. The resulting
        code will be stored in a file called proj<unique_id>.hpp in the
        directory indicated by annarchy_dir.

        This function will be called from CodeGenerator.

        Returns:

        * proj_desc: a dictionary with all call statements for the distinct
        operations (like compute_psp, update_synapse, etc.)
        """
        # configure Connectivity base class
        self.configure(proj)

        # Generate declarations and accessors for the variables
        decl, accessor = self._declaration_accessors(proj)

        # Initiliaze the projection
        init_parameters_variables = self._init_parameters_variables(proj)

        update_prefix, update_variables = self._update_synapse(proj)

        # Update the random distributions
        init_rng = self._init_random_distributions(proj)
        update_rng = self._update_random_distributions(proj)

        post_event_prefix, post_event = self._post_event(proj)

        # Compute sum is the trickiest part
        if proj.synapse_type.type == 'rate':
            psp_prefix, psp_code = self._computesum_rate(proj)
        else:
            psp_prefix, psp_code = self._computesum_spiking(proj)

        # Detect event-driven variables
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True

        # Detect non.uniform delays to eventually generate the code
        has_delay = (proj.max_delay > 1 and proj.uniform_delay == -1)

        # Connectivity matrix
        connectivity_matrix = self._connectivity(proj)

        # local functions
        decl['parameters_variables'] += self._local_functions(proj)

        # Profiling
        if self._prof_gen:
            include_profile = """#include "Profiling.h"\n"""
            declare_profile, init_profile = self._prof_gen.generate_init_projection(proj)
        else:
            include_profile = ""
            init_profile = ""
            declare_profile = ""

        # Additional info (overwritten)
        include_additional = ""
        struct_additional = ""
        init_additional = ""
        access_additional = ""
        if 'include_additional' in proj._specific_template.keys():
            include_additional = proj._specific_template['include_additional']
        if 'struct_additional' in proj._specific_template.keys():
            struct_additional = proj._specific_template['struct_additional']
        if 'init_additional' in proj._specific_template.keys():
            init_additional = proj._specific_template['init_additional']
        if 'access_additional' in proj._specific_template.keys():
            access_additional = proj._specific_template['access_additional']

        init_inverse = connectivity_matrix['init_inverse'] % {
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'post_size': proj.post.size # only needed if storage_format == "csr"
        }

        final_code = self._templates['projection_header'] % {
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'id_proj': proj.id,
            'name_pre': proj.pre.name,
            'name_post': proj.post.name,
            'target': proj.target,
            'include_additional': include_additional,
            'include_profile': include_profile,
            'struct_additional': struct_additional,
            'declare_connectivity_matrix': connectivity_matrix['declare'],
            'declare_inverse_connectivity_matrix': connectivity_matrix['declare_inverse'],
            'declare_delay': decl['delay'] if has_delay else "",
            'declare_event_driven': decl['event_driven'] if has_event_driven else "",
            'declare_rng': decl['rng'],
            'declare_parameters_variables': decl['parameters_variables'],
            'declare_additional': decl['additional'],
            'declare_profile': declare_profile,
            'init_connectivity_matrix': connectivity_matrix['init'],
            'init_inverse_connectivity_matrix': init_inverse,
            'init_event_driven': "",
            'init_rng': init_rng,
            'init_parameters_variables': init_parameters_variables,
            'init_additional': init_additional,
            'init_profile': init_profile,
            'psp_prefix': psp_prefix,
            'psp_code': psp_code,
            'update_rng': update_rng,
            'update_prefix': update_prefix,
            'update_variables': update_variables,
            'post_event_prefix': post_event_prefix,
            'post_event': post_event,
            'access_connectivity_matrix': connectivity_matrix['accessor'],
            'access_parameters_variables': accessor,
            'access_additional': access_additional,
        }

        # Store file
        with open(annarchy_dir+'/generate/net'+str(self._net_id)+'/proj'+str(proj.id)+'.hpp', 'w') as ofile:
            ofile.write(final_code)

        # Dictionary for inclusions in ANNarchy.cpp
        proj_desc = {
            'include': """#include "proj%(id)s.hpp"\n""" % {'id': proj.id},
            'extern': """extern ProjStruct%(id)s proj%(id)s;\n"""% {'id': proj.id},
            'instance': """ProjStruct%(id)s proj%(id)s;\n"""% {'id': proj.id},
            'init': """    proj%(id)s.init_projection();\n""" % {'id' : proj.id}
        }

        proj_desc['update'] = "" if update_variables == "" else """    proj%(id)s.update_synapse();\n""" % {'id': proj.id}
        proj_desc['rng_update'] = "" if update_rng == "" else """    proj%(id)s.update_rng();\n""" % {'id': proj.id}
        proj_desc['post_event'] = "" if post_event == "" else """    proj%(id)s.post_event();\n""" % {'id': proj.id}

        return proj_desc

    def creating(self, proj):
        creating_structure = proj.synapse_type.description['creating']

        # Random stuff
        proba = ""
        proba_init = ""
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

                delay = ", " + str(d)
            else:
                if d != proj.uniform_delay:
                    Global._error('creating: you can not add a delay different from the others if they were constant.')

        # OMP
        if Global.config['num_threads'] > 1:
            omp_code = '#pragma omp parallel for' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        creating_condition = creating_structure['cpp'] % {
            'id_proj' : proj.id, 'target': proj.target,
            'id_post': proj.post.id, 'id_pre': proj.pre.id,
            'post_prefix': 'pop%(id)s.' % {'id':proj.post.id}, 'post_index': '[rk_post]',
            'pre_prefix':  'pop%(id)s.' % {'id':proj.pre.id}, 'pre_index':'[rk_pre]'
        }
        creation_ids = {
            'id_proj' : proj.id, 'id_pre': proj.pre.id,
            'eq': creating_structure['eq'], 'modulo': '%',
            'condition': creating_condition,
            'omp_code': omp_code,
            'weights': 0.0 if not 'w' in creating_structure['bounds'].keys() else creating_structure['bounds']['w'],
            'proba' : proba, 'proba_init': proba_init,
            'delay': delay
        }
        creating = """
    // proj%(id_proj)s creating: %(eq)s
    if((proj%(id_proj)s._creating)&&((t - proj%(id_proj)s._creating_offset) %(modulo)s proj%(id_proj)s._creating_period == 0)){
        %(proba_init)s
        //%(omp_code)s
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            int rk_post = proj%(id_proj)s.post_rank[i];
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
                        //std::cout << "Creating synapse between " << rk_pre << " and " << rk_post << std::endl;
                        proj%(id_proj)s.addSynapse(i, rk_pre, %(weights)s%(delay)s);
                    }
                }
            }
        }
    }
""" % creation_ids

        return creating

    def pruning(self, proj):
        pruning_structure = proj.synapse_type.description['pruning']

        proba = ""
        proba_init = ""
        if 'proba' in pruning_structure['bounds'].keys():
            val = pruning_structure['bounds']['proba']
            proba = '&&(unif(rng)<' + val + ')'
            proba_init = "std::uniform_real_distribution<double> unif(0.0, 1.0);"
        if pruning_structure['rd']:
            proba_init += "\n        " +  pruning_structure['rd']['template'] + ' rd(' + pruning_structure['rd']['args'] + ');'

        if Global.config['num_threads'] > 1:
            omp_code = '#pragma omp parallel for' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        pruning_condition = pruning_structure['cpp'] % {
            'id_proj' : proj.id, 'target': proj.target,
            'id_post': proj.post.id, 'id_pre': proj.pre.id,
            'global_index': '',
            'semiglobal_index': '[i]',
            'local_index': '[i][j]'
        }

        # HACK:
        for dep in pruning_structure['dependencies']:
            pruning_condition = pruning_condition.replace(dep, 'proj'+str(proj.id)+'.'+dep)

        pruning_ids = {
            'id_proj' : proj.id,
            'eq': pruning_structure['eq'],
            'modulo': '%',
            'condition': pruning_condition,
            'omp_code': omp_code,
            'proba' : proba,
            'proba_init': proba_init
        }
        pruning = """
    // proj%(id_proj)s pruning: %(eq)s
    if((proj%(id_proj)s._pruning)&&((t - proj%(id_proj)s._pruning_offset) %(modulo)s proj%(id_proj)s._pruning_period == 0)){
        %(proba_init)s
        //%(omp_code)s
        for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
            int rk_post = proj%(id_proj)s.post_rank[i];
            for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
                int rk_pre = proj%(id_proj)s.pre_rank[i][j];
                if((%(condition)s)%(proba)s){
                    proj%(id_proj)s.removeSynapse(i, j);
                }
            }
        }
    }
""" % pruning_ids

        return pruning

    def _computesum_rate(self, proj):
        """
        Create the c++ code for post-synaptic potential computation.
        """
        # Default variables needed in psp_code
        psp_prefix = """
        int nb_post; double sum;"""
        if 'psp_prefix' in proj._specific_template.keys():
            psp_prefix = proj._specific_template['psp_prefix']

        # Specific projection
        if 'psp_code' in proj._specific_template.keys():
            psp_code = proj._specific_template['psp_code']
            if self._prof_gen:
                psp_code = self._prof_gen.annotate_computesum_rate_omp(proj, psp_code)

            return psp_prefix, psp_code

        # Choose the relevant summation template
        if proj._dense_matrix: # Dense connectivity
            template = OpenMPTemplates.dense_summation_operation
        else: # Default LiL
            template = OpenMPTemplates.lil_summation_operation

        # Dictionary of keywords to transform the parsed equations
        ids = {
            'id_proj' : proj.id,
            'target': proj.target,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,
            'local_index': "[i][j]",
            'semiglobal_index': '[i]',
            'global_index': '',
            'pre_index': '[pre_rank[i][j]]',
            'post_index': '[post_rank[i]]',
            'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
            'post_prefix': 'pop'+ str(proj.post.id) + '.',
            'delay_nu' : '[delay[i][j]-1]', # non-uniform delay
            'delay_u' : '[' + str(proj.uniform_delay-1) + ']' # uniform delay
        }

        # Special keywords based on the data structure
        if proj._dense_matrix: # Dense connectivity
            ids['pre_index'] = "[j]"
            ids['post_index'] = "[i]"

        # Retrieve the PSP
        if not 'psp' in  proj.synapse_type.description.keys(): # default
            psp = """%(preprefix)s.r%(pre_index)s * w%(local_index)s;"""
        else: # custom psp
            psp = (proj.synapse_type.description['psp']['cpp'])

        # Special case where w is a single value
        if proj._has_single_weight():
            psp = re.sub(
                r'([^\w]+)w%\(local_index\)s',
                r'\1w',
                ' ' + psp
            )

        # OpenMP
        with_openmp = Global.config['num_threads'] > 1 and proj.post.size > Global.OMP_MIN_NB_NEURONS

        # Dependencies
        dependencies = list(set(proj.synapse_type.description['dependencies']['pre']))

        # Delayed variables
        if isinstance(proj.pre, PopulationView):
            delayed_variables = proj.pre.population.delayed_variables
        else:
            delayed_variables = proj.pre.delayed_variables

        # Delays
        if proj.max_delay > 1: # There is non-zero delay
            if proj.uniform_delay == -1: # Non-uniform delays
                for var in delayed_variables:
                    if var in proj.pre.neuron_type.description['local']:
                        psp = psp.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_nu)s%(pre_index)s'
                        )
                    else:
                        Global._print(proj.synapse_type.description['psp']['eq'])
                        Global._error('The psp accesses a global variable with a non-uniform delay!')


            else: # Uniform delays
                for var in delayed_variables:
                    if var in proj.pre.neuron_type.description['local']:
                        psp = psp.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_u)s%(pre_index)s'
                        )
                    else:
                        psp = psp.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_u)s'
                        )

        # Generate OMP code and eventually a pre-copy
        omp_code = ""
        pre_copy = ""

        # OMP: make a local copy of local variables for each thread if the delays are constant
        if with_openmp:
            omp_schedule = "" if not 'psp_schedule' in proj._omp_config.keys() else proj._omp_config['psp_schedule']

            if proj.max_delay > 1: # there is a delay
                if proj.uniform_delay == -1: # Non-uniform delays: do nothing
                    omp_code = '#pragma omp parallel for private(sum) firstprivate(nb_post) %(schedule)s' % {'schedule': omp_schedule}

                else: # Uniform delays
                    omp_code = "#pragma omp parallel for private(sum) firstprivate("
                    for var in dependencies:
                        if var in proj.pre.neuron_type.description['local']:
                            pre_copy += "std::vector<double> _pre_" + var + " = %(pre_prefix)s_delayed_" + var + "%(delay_u)s;"
                            psp = psp.replace(
                                '%(pre_prefix)s_delayed_'+var+'%(delay_u)s%(pre_index)s',
                                '_pre_'+var+'%(pre_index)s'
                            )
                            omp_code += '_pre_%(var)s, ' % {'var': var}

                    omp_code += "nb_post) %(schedule)s" % {'schedule': omp_schedule}

            else: # No delay
                pre_copy = ""
                omp_code = "#pragma omp parallel for private(sum) firstprivate("
                for var in dependencies:
                    if var in proj.pre.neuron_type.description['local']:
                        pre_copy += "std::vector<double> _pre_" + var + " = %(pre_prefix)s" + var + ";"
                        psp = psp.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '_pre_'+var+'%(pre_index)s'
                        )
                        omp_code += '_pre_%(var)s, ' % {'var': var}

                omp_code += "nb_post) %(schedule)s" % {'schedule': omp_schedule}

        # Finalize the psp with the correct ids
        psp = psp % ids
        pre_copy = pre_copy % ids

        # Generate the code depending on the operation
        sum_code = template[proj.synapse_type.operation] % {
            'pre_copy': pre_copy,
            'omp_code': omp_code,
            'psp': psp.replace(';', ''),
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'target': proj.target,
        }

        # Finish the code
        final_code = """
        if (_transmission && pop%(id_post)s._active){
%(code)s
        } // active
        """ % {'id_post': proj.post.id,
               'code': tabify(sum_code, 3),
              }

        if self._prof_gen:
            final_code = self._prof_gen.annotate_computesum_rate(proj, final_code)

        return psp_prefix, final_code

    def _computesum_spiking(self, proj):
        """
        Generate codes for spike propagation and pre-spike part of event-driven equations.

        Returns:

            * psp_prefix: set of variables needed in the kernel, positioned at the begin of
                          Projection::compute_psp() method.
            * code: computation body

        Specific templates:

            * psp_prefix and psp_code
        """
        if 'psp_prefix' in proj._specific_template.keys() and 'psp_code' in proj._specific_template.keys():
            try:
                psp_prefix = proj._specific_template['psp_prefix']
                psp_code = proj._specific_template['psp_code']
            except KeyError:
                Global._error('psp_prefix as well as psp_code should be overwritten')
            return psp_prefix, psp_code

        # There was no
        if proj._storage_format == "lil":
            psp_prefix = """
        int nb_post, i, j, rk_j, rk_post, rk_pre;
        double sum;"""
        elif proj._storage_format == "csr":
            psp_prefix = ""
        else:
            raise NotImplementedError

        # Basic tags, dependent on storage format
        if proj._storage_format == "lil":
            ids = {
                'id_proj' : proj.id,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'target': proj.target,
                'local_index': "[i][j]",
                'semiglobal_index': '[i]',
                'global_index': ''
            }
        elif proj._storage_format == "csr":
            ids = {
                'id_proj' : proj.id,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'target': proj.target,
                'local_index': "[syn]",
                'semiglobal_index': '[_col_idx[syn]]',
                'global_index': ''
            }
        else:
            raise NotImplementedError

        # Determine the mode of synaptic transmission
        continous_transmission = False
        if 'psp' in  proj.synapse_type.description.keys(): # continous
            continous_transmission = True

        ####################################################
        # Event-driven summation of g_target
        ####################################################
        # Strings
        updated_variables_list = []
        g_target = ""
        g_target_code = ""

        # Analyse all elements of pre_spike
        for eq in proj.synapse_type.description['pre_spike']:
            # g_target is treated differently
            # Must be at the end of the equations
            if eq['name'] == 'g_target':
                # PSP form
                g_target = eq['cpp'].split('=')[1] % ids
                # Check targets
                if isinstance(proj.target, str):
                    targets = [proj.target]
                else:
                    targets = proj.target
                g_target_code = ""
                for target in targets:
                    if proj._storage_format == "lil":
                        acc = "post_rank[i]"
                    elif proj._storage_format == "csr":
                        acc = "_col_idx[syn]"
                    else:
                        raise NotImplementedError

                    target_dict = {
                        'id_post': proj.post.id,
                        'target': target,
                        'g_target': g_target,
                        'eq': eq['eq'],
                        'acc': acc,
                    }

                    if Global.config['num_threads'] == 1:
                        g_target_code += """
            // Increase the post-synaptic conductance %(eq)s
            pop%(id_post)s.g_%(target)s[%(acc)s] += %(g_target)s
""" % target_dict
                    else:
                        # access to post variable requires
                        # an atomic operation
                        g_target_code += """
            // Increase the post-synaptic conductance %(eq)s
            #pragma omp atomic
            pop%(id_post)s.g_%(target)s[%(acc)s] += %(g_target)s
""" % target_dict
                    # Determine bounds
                    for key, val in eq['bounds'].items():
                        if not key in ['min', 'max']:
                            continue
                        try:
                            value = str(float(val))
                        except: # TODO: more complex operations
                            value = val % ids

                        g_target_code += """
            if (pop%(id_post)s.g_%(target)s[post_rank[i]] %(op)s %(val)s)
                pop%(id_post)s.g_%(target)s[post_rank[i]] = %(val)s;
""" % {'id_post': proj.post.id, 'target': target, 'op': "<" if key == 'min' else '>', 'val': value}

            else:
                # process equations in pre_spike which
                # are not 'g_target'

                condition = ""
                # Check conditions to update the variable
                if eq['name'] == 'w': # Surround it by the learning flag
                    condition = "_plasticity" # Plasticity can be disabled

                if 'unless_post' in eq['flags']: # Flags avoids pre-spike evaluation when post fires at the same time
                    simultaneous = "pop%(id_pre)s.last_spike[pre_rank[i][j]] != pop%(id_post)s.last_spike[post_rank[i]]" % {'id_post': proj.post.id, 'id_pre': proj.pre.id}
                    if condition == "":
                        condition = simultaneous
                    else:
                        condition += "&&(" + simultaneous + ")"

                eq_dict = {
                    'eq': eq['eq'],
                    'cpp': eq['cpp'] % ids,
                    'bounds': get_bounds(eq) % ids,
                    'condition': condition
                }

                # Generate the code, either with or without coundition
                if condition != "":
                    updated_variables_list += """
// unless_post can prevent evaluation of presynaptic variables
if (%(condition)s) {
    // %(eq)s
    %(cpp)s
    %(bounds)s
}
""" % eq_dict
                else: # Normal synaptic variable
                    updated_variables_list += """
// %(eq)s
%(cpp)s
%(bounds)s""" % eq_dict


        # Generate the default post-conductance increase
        # default g_target += w
        if not continous_transmission and g_target == "":
            # Check targets
            if isinstance(proj.target, str):
                targets = [proj.target]
            else:
                targets = proj.target

            g_target_code = ""
            for target in targets:
                g_target_code += """
            // Increase the post-synaptic conductance g_target += w
            pop%(id_post)s.g_%(target)s[post_rank[i]] += w%(local_idx)s;
""" % ids

        # Special case where w is a single value
        if proj._has_single_weight():
            g_target_code = re.sub(
                r'([^\w]+)w\[i\]\[j\]',
                r'\1w',
                g_target_code
            )

        # Event-driven integration of synaptic variables
        has_exact = False
        event_driven_code = ''
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_exact = True
                event_driven_code += """
            // %(eq)s
            %(exact)s
""" % {'eq': var['eq'], 'exact': var['cpp'].replace('(t)', '(t-1)') % ids}
        if has_exact:
            event_driven_code += """
            // Update the last event for the synapse
            _last_event%(local_index)s = t;
""" % ids

        # Generate code for pre-spike variables
        pre_code = ""
        if len(updated_variables_list) > 0:
            for var in updated_variables_list:
                pre_code += var
            pre_code = tabify(pre_code, 3)
            # Special case where w is a single value
            if proj._has_single_weight():
                pre_code = re.sub(
                    r'([^\w]+)w\[i\]\[j\]',
                    r'\1w',
                    pre_code
                )

        # Default template is without variable delays

        #TODO: choose correct template based on connectivity
        if proj._storage_format == "lil":
            template = OpenMPTemplates.spiking_summation_fixed_delay
        elif proj._storage_format == "csr":
            template = OpenMPTemplates.spiking_summation_fixed_delay_csr
        #template = OpenMPTemplates.spiking_summation_fixed_delay_dense_matrix

        # Take delays into account if any
        pre_array = ""
        if proj.max_delay > 1:
            if proj.uniform_delay == -1: # Non-uniform delays
                Global._warning('Variable delays for spiking networks is experimental and slow...')
                template = OpenMPTemplates.spiking_summation_variable_delay
            else: # Uniform delays
                pre_array = "pop%(id_pre)s._delayed_spike[%(delay)s]" % {'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
        else:
            pre_array = "pop%(id_pre)s.spiked" % ids

        # No need for openmp if less than 100 post neurons
        omp_code = ""
        if Global.config['num_threads'] > 1:
            if proj._storage_format == "lil":
                if proj.post.size > Global.OMP_MIN_NB_NEURONS and len(updated_variables_list) > 0:
                    omp_code = """#pragma omp parallel for firstprivate(nb_post, inv_post) private(i, j)"""
            elif proj._storage_format == "csr":
                omp_code = """#pragma omp parallel for"""
            else:
                omp_code = ""

        # Generate the whole code block
        code = ""
        if g_target_code != "" or pre_code != "":
            code = template % {
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'pre_array': pre_array,
                'pre_event': pre_code,
                'g_target': g_target_code,
                'omp_code': omp_code,
                'event_driven': event_driven_code
            }

        # Add tabs
        code = tabify(code, 2)

        ####################################################
        # Not even-driven summation of psp: like rate-coded
        ####################################################
        if 'psp' in  proj.synapse_type.description.keys(): # not event-based
            # Compute it as if it were rate-coded
            psp_code = self._computesum_rate(proj)[1]
            # Change _sum_target into g_target
            psp_code = psp_code.replace( # for LIL
                'pop%(id_post)s._sum_%(target)s[post_rank[i]]' % {'id_post': proj.post.id, 'target': proj.target},
                'pop%(id_post)s.g_%(target)s[post_rank[i]]' % {'id_post': proj.post.id, 'target': proj.target}
            )
            psp_code = psp_code.replace( # for Dense
                'pop%(id_post)s._sum_%(target)s[i]' % {'id_post': proj.post.id, 'target': proj.target},
                'pop%(id_post)s.g_%(target)s[i]' % {'id_post': proj.post.id, 'target': proj.target}
            )
            # Add it to the main code
            code += """
        // PSP-based summation"""
            code += psp_code

        # Annotate code
        if self._prof_gen:
            code = self._prof_gen.annotate_computesum_spiking(proj, code)

        return psp_prefix, code

    def _header_structural_plasticity(self, proj):
        """
        Generate extension code for C header_struct: variable declaration, add and remove synapses.

        Templates:

            structural_plasticity: 'header_struct' field contains all relevant code templates

        """
        header_tpl = OpenMPTemplates.structural_plasticity['header_struct']

        code = ""
        # Pruning defined in the synapse
        if 'pruning' in proj.synapse_type.description.keys():
            code += header_tpl['pruning']

        # Creating defined in the synapse
        if 'creating' in proj.synapse_type.description.keys():
            code += header_tpl['creating']

        # Retrieve the names of extra attributes
        extra_args = ""
        add_code = ""
        remove_code = ""
        for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
            if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse_type.description['local']:

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

        # Spiking networks must update the inv_pre_rank array
        spiking_addcode = "" if proj.synapse_type.type == 'rate' else header_tpl['spiking_addcode']
        spiking_removecode = "" if proj.synapse_type.type == 'rate' else header_tpl['spiking_removecode']

        # Randomdistributions
        rd_addcode = ""
        rd_removecode = ""
        for rd in proj.synapse_type.description['random_distributions']:
            rd_addcode += """
        %(name)s[post].insert(%(name)s[post].begin() + idx, 0.0);
""" % {'name': rd['name']}

            rd_removecode += """
        %(name)s[post].erase(%(name)s[post].begin() + idx);
""" % {'name': rd['name']}

        # Generate the code
        code += header_tpl['header'] % {
            'extra_args': extra_args, 'delay_code': delay_code,
            'add_code': add_code, 'remove_code': remove_code,
            'spike_add': spiking_addcode, 'spike_remove': spiking_removecode,
            'rd_add': rd_addcode, 'rd_remove': rd_removecode
        }

        return code

    def _init_random_distributions(self, proj):
        # Is it a specific population?
        if 'init_rng' in proj._specific_template.keys():
            return proj._specific_template['init_rng']

        code = ""
        for rd in proj.synapse_type.description['random_distributions']:
            rd_init = rd['definition']% {'id': proj.id, 'float_prec': Global.config['precision']}
            code += """    %(rd_name)s = std::vector< std::vector<double> >(post_rank.size(), std::vector<double>());
    for(int i=0; i<post_rank.size(); i++){
        %(rd_name)s[i] = std::vector<double>(pre_rank[i].size(), 0.0);
    }
    dist_%(rd_name)s = %(rd_init)s;
""" % {'rd_name': rd['name'], 'rd_init': rd_init}
        return code

    def _local_functions(self, proj):
        " Local functions "
        local_func = ""
        if len(proj.synapse_type.description['functions']) > 0:
            local_func += """
    // Local functions
"""
            for func in proj.synapse_type.description['functions']:
                local_func += ' '*4 + func['cpp'] + '\n'

        return local_func

    def _post_event(self, proj):
        if proj.synapse_type.type == "rate":
            return "", ""

        if proj.synapse_type.description['post_spike'] == []:
            return "", ""

        if proj._storage_format == "lil":
            ids = {
                'id_proj' : proj.id,
                'target': proj.target,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'local_index': "[i][j]",
                'semiglobal_index': '[i]',
                'global_index': '',
                'pre_index': '[pre_rank[i][j]]',
                'post_index': '[post_rank[i]]',
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
            }

            post_event_prefix = """
        int i, j, rk_post, nb_pre;"""
        elif proj._storage_format == "csr":
            ids = {
                'id_proj' : proj.id,
                'target': proj.target,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'local_index': "[_inv_idx[j]]",
                'semiglobal_index': '[*it]',
                'global_index': '',
                'pre_index': '[]',
                'post_index': '[]',
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
            }

            post_event_prefix = """
        int rk_post;
        std::vector<int>::iterator it;
        """

        else:
            raise NotImplementedError

        # Event-driven integration
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True

        # Generate event-driven code
        event_driven_code = ""
        if has_event_driven:
            for var in proj.synapse_type.description['variables']:
                if var['method'] == 'event-driven':
                    event_driven_code += '// ' + var['eq'] + '\n'
                    event_driven_code += var['cpp'] % ids + '\n'
            event_driven_code += """
// Update the last event for the synapse
_last_event%(local_index)s = t;
""" % ids

            event_driven_code = tabify(event_driven_code, 3)

        # Gather the equations
        post_code = ""
        for eq in proj.synapse_type.description['post_spike']:
            post_code += '// ' + eq['eq'] + '\n'
            if eq['name'] == 'w':
                post_code += "if(_plasticity)\n"
            post_code += eq['cpp'] % ids + '\n'
            post_code += get_bounds(eq) % ids + '\n'
        post_code = tabify(post_code, 3)

        # OMP code
        if Global.config['num_threads'] > 1:
            omp_code = '#pragma omp parallel for private(j) firstprivate(i, nb_pre)' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        # Generate the code block
        if proj._storage_format == "lil":
            psp_lil = {
                'id_post': proj.post.id,
                'post_event': post_code,
                'event_driven': event_driven_code,
                'omp_code': omp_code
            }
            code = """
if(_transmission && pop%(id_post)s._active){
    for(int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++){
        // Rank of the postsynaptic neuron which fired
        rk_post = pop%(id_post)s.spiked[_idx_i];
        // Find its index in the projection
        i = inv_post_rank.at(rk_post);
        // Leave if the neuron is not part of the projection
        if (i==-1) continue;
        // Iterate over all synapse to this neuron
        nb_pre = pre_rank[i].size();
        %(omp_code)s
        for(j = 0; j < nb_pre; j++){
%(event_driven)s
%(post_event)s
        }
    }
}
""" % psp_lil
        elif proj._storage_format == "csr":
            psp_csr = {
                'id_post': proj.post.id,
                'post_event': post_code,
                'event_driven': event_driven_code,
                'omp_code': omp_code
            }
            code = """
if(_transmission && pop%(id_post)s._active){
    for(int _idx_i = 0; _idx_i < pop%(id_post)s.spiked.size(); _idx_i++){
        // Rank of the postsynaptic neuron which fired
        rk_post = pop%(id_post)s.spiked[_idx_i];
        // Find its index in the projection
        it = std::find(post_ranks.begin(), post_ranks.end(), rk_post);
        // Leave if the neuron is not part of the projection
        if (it==post_ranks.end())
            continue;
        // Iterate over all synapse to this neuron
        %(omp_code)s
        for(int j = _col_ptr[*it]; j < _col_ptr[(*it)+1]; j++){
%(event_driven)s
%(post_event)s
        }
    }
}
""" % psp_csr
        else:
            raise NotImplementedError

        return post_event_prefix, tabify(code, 2)

    def _update_random_distributions(self, proj):
        # Is it a specific population?
        if 'update_rng' in proj._specific_template.keys():
            return proj._specific_template['update_rng']

        code = ""
        if len(proj.synapse_type.description['random_distributions']) > 0:
            code += """
    // RD of proj%(id_proj)s
    for(int i = 0; i < post_rank.size(); i++){
        for(int j = 0; j < pre_rank[i].size(); j++){
"""% {'id_proj': proj.id}

            for rd in proj.synapse_type.description['random_distributions']:
                code += """
            %(rd_name)s[i][j] = dist_%(rd_name)s(rng);""" % {'rd_name': rd['name']}

            code += """
        }
    }
"""
        return code

    def _update_synapse(self, proj):
        prefix = """
        int rk_post, rk_pre;
        double _dt = dt * _update_period;"""

        # Dictionary of pre/suffixes
        ids = {
            'id_proj' : proj.id,
            'target': proj.target,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,
            'local_index': '[i][j]',
            'semiglobal_index': '[i]',
            'global_index': '',
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]',
            'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
            'post_prefix': 'pop'+ str(proj.post.id) + '.',
            'delay_nu' : '[delay[i][j]-1]', # non-uniform delay
            'delay_u' : '[' + str(proj.uniform_delay-1) + ']' # uniform delay
        }

        # Global variables
        global_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'global', 'proj', padding=2, wrap_w="_plasticity")

        # Semiglobal variables
        semiglobal_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'semiglobal', 'proj', padding=2, wrap_w="_plasticity")

        # Local variables
        local_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'local', 'proj', padding=3, wrap_w="_plasticity")

        # adjust dt dependent on the _update_period, this covers only
        # the switch statements
        global_eq = global_eq.replace(" dt*", " _dt*")
        local_eq = local_eq.replace(" dt*", " _dt*")

        # Skip generation if
        if local_eq.strip() == '' and global_eq.strip() == '':
            return "", ""

        # Special case where w is a single value
        if proj._has_single_weight():
            local_eq = re.sub(
                r'([^\w]+)w%\(local_index\)s',
                r'\1w',
                ' ' + local_eq
            )
            global_eq = re.sub(
                r'([^\w]+)w%\(local_index\)s',
                r'\1w',
                ' ' + global_eq
            )

        # OpenMP
        omp_code = ""
        if Global.config['num_threads'] > 1 and proj.post.size > Global.OMP_MIN_NB_NEURONS:
            omp_code = '#pragma omp parallel for private(rk_pre, rk_post) schedule(dynamic)'

        # Dependencies
        dependencies = list(set(proj.synapse_type.description['dependencies']['pre']))

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1: # Non-uniform delays
                for var in dependencies:
                    if var in proj.pre.neuron_type.description['local']:
                        local_eq = local_eq.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_nu)s%(pre_index)s'
                        )
                    else:
                        local_eq = local_eq.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_nu)s'
                        )
            else: # Uniform delays
                for var in dependencies:
                    if var in proj.pre.neuron_type.description['local']:
                        local_eq = local_eq.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_u)s%(pre_index)s'
                        )
                    else:
                        local_eq = local_eq.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_u)s'
                        )

        # Choose the template
        if proj._dense_matrix: # Dense matrix
            template = OpenMPTemplates.dense_update_variables
        else: # Default: LIL
            template = OpenMPTemplates.lil_update_variables

        # Fill the code template
        if local_eq.strip() != "": # local synapses are updated
            code = template['local'] % {
                'global': global_eq % ids,
                'semiglobal': semiglobal_eq % ids,
                'local': local_eq % ids,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'omp_code': omp_code
            }
        else: # Only global variables
            code = template['global'] % {
                'global': global_eq % ids,
                'semiglobal': semiglobal_eq % ids,
                'id_post': proj.post.id,
                'omp_code': omp_code
            }

        if self._prof_gen:
            code = self._prof_gen.annotate_update_synapse(proj, code)

        # Return the code block
        return prefix, tabify(code, 2)
