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
import Template.ProjectionTemplate as ProjTemplate
from .Utils import generate_equation_code, tabify

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
        # Dictionary for inclusions in ANNarchy.cpp
        proj_desc = {
            'include': """#include "proj%(id)s.hpp"\n""" % { 'id': proj.id },
            'extern': """extern ProjStruct%(id)s proj%(id)s;\n"""% { 'id': proj.id },
            'instance': """ProjStruct%(id)s proj%(id)s;\n"""% { 'id': proj.id },
            'init': """    proj%(id)s.init_projection();\n""" % {'id' : proj.id}
        }

        # Generate declarations and accessors for the variables
        decl, accessor = self.declaration_accessors(proj)

        # Initiliaze the projection
        init_parameters_variables = self.init_parameters_variables(proj)

        # Update the variables
        update_prefix, update_variables = self.update_synapse(proj)

        # Update the random distributions
        init_rng = self.init_random_distributions(proj)
        update_rng = self.update_random_distributions(proj)

        # Spiking networks may have a post_spike arguement
        post_event_prefix, post_event = self.postevent(proj)

        # Compute sum is the trickiest part
        if proj.synapse.type == 'rate': 
            if Global.config['paradigm'] == "openmp":
                psp_prefix, psp_code = self.computesum_rate_openmp(proj)
            else:
                psp_header, psp_body, psp_call = self.computesum_rate_cuda(proj)
        else:
            if Global.config['paradigm'] == "openmp":
                psp_prefix, psp_code = self.computesum_spiking(proj)
            else:
                Global._error("Spiking networks are not supported on CUDA yet ...")
                exit(0)

        # Detect event-driven variables
        has_event_driven = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True

        # Detect non.uniform delays to eventually generate the code
        has_delay = ( proj.max_delay > 1 and proj.uniform_delay == -1)

        # Connectivity matrix
        connectivity_matrix = self.connectivity(proj) 

        # Additional info (overwritten)
        include_additional = ""; struct_additional = ""; init_additional = ""; access_additional = ""
        if 'include_additional' in proj._specific_template.keys():
            include_additional = proj._specific_template['include_additional']
        if 'struct_additional' in proj._specific_template.keys():
            struct_additional = proj._specific_template['struct_additional']
        if 'init_additional' in proj._specific_template.keys():
            init_additional = proj._specific_template['init_additional']
        if 'access_additional' in proj._specific_template.keys():
            access_additional = proj._specific_template['access_additional']
        
        final_code = ProjTemplate.header_struct % {
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'id_proj': proj.id,
            'name_pre': proj.pre.name,
            'name_post': proj.post.name,
            'target': proj.target,
            'include_additional': include_additional,
            'struct_additional': struct_additional,
            'declare_connectivity_matrix': connectivity_matrix['declare'],
            'declare_inverse_connectivity_matrix': connectivity_matrix['declare_inverse'],
            'declare_delay': decl['delay'] if has_delay else "",
            'declare_event_driven': decl['event_driven']if has_event_driven else "",
            'declare_rng': decl['rng'],
            'declare_parameters_variables': decl['parameters_variables'],
            'declare_cuda_stream': decl['cuda_stream'],
            'declare_additional': decl['additional'],
            'init_connectivity_matrix': connectivity_matrix['init'],
            'init_inverse_connectivity_matrix': connectivity_matrix['init_inverse'] % {'id_pre': proj.pre.id, 'id_post': proj.post.id},
            'init_event_driven': "",
            'init_rng': init_rng,
            'init_parameters_variables': init_parameters_variables,
            'init_additional': init_additional,
            'psp_prefix': psp_prefix if Global.config['paradigm'] == "openmp" else "",
            'psp_code': psp_code if Global.config['paradigm'] == "openmp" else "",
            'update_rng': update_rng,
            'update_prefix': update_prefix,
            'update_variables': update_variables,
            'post_event_prefix': post_event_prefix,
            'post_event': post_event,
            'access_connectivity_matrix': connectivity_matrix['accessor'],
            'access_parameters_variables': accessor,
            'access_additional': access_additional,
            'cuda_flattening': ProjTemplate.cuda_flattening if Global.config['paradigm'] == "cuda" else ""
        }
        
        # Store file    
        with open(annarchy_dir+'/generate/proj'+str(proj.id)+'.hpp', 'w') as ofile:
            ofile.write(final_code)

        #
        # Dependent on the chosen paradigm, different codes need to
        # added in ANNarchy.c* file:
        #
        # * openmp: contains calls of filled *.hpp functions
        # * cuda: we return the generated calls to the main CodeGenerator
        if Global.config['paradigm'] == "openmp":
            proj_desc['update'] = "" if update_variables=="" else """    proj%(id)s.update_synapse();\n""" % { 'id': proj.id }
            proj_desc['rng_update'] = "" if update_rng=="" else """    proj%(id)s.update_rng();\n""" % { 'id': proj.id }
            proj_desc['post_event'] = "" if post_event == "" else """    proj%(id)s.post_event();\n""" % { 'id': proj.id }

        else:
            proj_desc['psp_header'] = psp_header
            proj_desc['psp_body'] = psp_body
            proj_desc['psp_call'] = psp_call

            host_device_transfer, device_host_transfer = self._cuda_memory_transfers(proj)

            proj_desc['host_to_device'] = host_device_transfer
            proj_desc['device_to_host'] = device_host_transfer

        return proj_desc

    def connectivity(self, proj):
        """
        Create codes for connectivity, comprising usually of post_rank and pre_rank.
        In case of spiking models they are extended by an inv_rank data field. The
        extension SharedProjection as well as SpecificProjection members overwrite
        the _specific_template member variable of the Projection object, to 
        replace directly the default codes.
        
        Returns:
            
            a dictionary containing the following fields: *declare*, *init*, 
            *accessor*, *declare_inverse*, *init_inverse*

        TODO:

            maybe we should make a forced linkage between the declare, accessor and init fields.
            Currently it could be, that one overwrite only the declare field and leave the others
            untouched which causes errors. 
        """
        declare_connectivity_matrix = ""
        init_connectivity_matrix = ""
        access_connectivity_matrix = ""
        declare_inverse_connectivity_matrix = ""
        init_inverse_connectivity_matrix = ""
        
        # Retrieve the template
        connectivity_matrix_tpl = ProjTemplate.connectivity_matrix_omp if Global.config['paradigm']=="openmp" else ProjTemplate.connectivity_matrix_cuda
        
        # Connectivity
        declare_connectivity_matrix = connectivity_matrix_tpl['declare'] 
        access_connectivity_matrix = connectivity_matrix_tpl['accessor']
        init_connectivity_matrix = connectivity_matrix_tpl['init']

        # Spiking model require inverted ranks 
        if proj.synapse.type == "spike":
            inv_connectivity_matrix_tpl = ProjTemplate.inverse_connectivity_matrix if Global.config['paradigm']=="openmp" else {}
            declare_inverse_connectivity_matrix = inv_connectivity_matrix_tpl['declare'] 
            init_inverse_connectivity_matrix = inv_connectivity_matrix_tpl['init']

        # Specific projections can overwrite
        if 'declare_connectivity_matrix' in proj._specific_template.keys():
            declare_connectivity_matrix = proj._specific_template['declare_connectivity_matrix'] 
        if 'access_connectivity_matrix' in proj._specific_template.keys():
            access_connectivity_matrix = proj._specific_template['access_connectivity_matrix']
        if 'declare_inverse_connectivity_matrix' in proj._specific_template.keys():
            declare_inverse_connectivity_matrix = proj._specific_template['declare_inverse_connectivity_matrix'] 
        if 'init_connectivity_matrix' in proj._specific_template.keys():
            init_connectivity_matrix = proj._specific_template['init_connectivity_matrix'] 
        if 'init_inverse_connectivity_matrix' in proj._specific_template.keys():
            init_inverse_connectivity_matrix = proj._specific_template['init_inverse_connectivity_matrix'] 

        
        return {
            'declare' : declare_connectivity_matrix,
            'init' : init_connectivity_matrix,
            'accessor' : access_connectivity_matrix,
            'declare_inverse': declare_inverse_connectivity_matrix,
            'init_inverse': init_inverse_connectivity_matrix
        }
        
    def declaration_accessors(self, proj):
        """
        Generate declaration and accessor code for variables/parameters of the projection.
        
        Returns:
            (dict, str): first return value contain declaration code and last one the accessor code.
            
            The declaration dictionary has the following fields: 
                delay, event_driven, rng, parameters_variables, additional, cuda_stream
        """
        # create the code for non-specific projections
        declare_delay = ""
        declare_event_driven = ""
        declare_rng = ""
        declare_parameters_variables = ""
        declare_additional = ""

        # choose templates dependend on the paradigm
        decl_template = ProjTemplate.attribute_decl[Global.config['paradigm']]
        acc_template = ProjTemplate.attribute_acc[Global.config['paradigm']]

        # Delays
        declare_delay = ProjTemplate.delay['header_struct']

        # Code for declarations and accessors
        accessor = ""
        # Parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            declare_parameters_variables += decl_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter' }
            accessor += acc_template[var['locality']]% {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'parameter' }

        # Variables
        for var in proj.synapse.description['variables']:
            if var['name'] == 'w': # Already defined by the connectivity matrix
                continue
            declare_parameters_variables += decl_template[var['locality']] % {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable' }
            accessor += acc_template[var['locality']]% {'type' : var['ctype'], 'name': var['name'], 'attr_type': 'variable' }

        # If no psp is defined, it's event-driven
        has_event_driven = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break
        if has_event_driven:
            declare_event_driven = ProjTemplate.event_driven['header_struct']

        # Arrays for the random numbers
        if len(proj.synapse.description['random_distributions']) > 0:
            declare_rng += """
    // Random numbers
"""
            for rd in proj.synapse.description['random_distributions']:
                declare_rng += """    std::vector< std::vector<double> > %(rd_name)s;
    %(template)s dist_%(rd_name)s;
""" % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}

        # Local functions
        if len(proj.synapse.description['functions'])>0:
            declare_parameters_variables += """
    // Local functions"""
            for func in proj.synapse.description['functions']:
                declare_parameters_variables += ' '*4 + func['cpp'] + '\n'

        # Structural plasticity
        if Global.config['structural_plasticity']:
            declare_parameters_variables += self.header_structural_plasticity(proj)

        # Specific projections can overwrite
        if 'declare_parameters_variables' in proj._specific_template.keys():
            declare_parameters_variables  = proj._specific_template['declare_parameters_variables']
        if 'declare_rng' in proj._specific_template.keys():
            declare_rng = proj._specific_template['declare_rng']
        if 'declare_event_driven' in proj._specific_template.keys():
            declare_event_driven = proj._specific_template['declare_event_driven']
        if 'declare_delay' in proj._specific_template.keys():
            declare_delay = proj._specific_template['declare_delay']
        if 'declare_additional' in proj._specific_template.keys():
            declare_additional = proj._specific_template['declare_additional']
        if 'access_parameters_variables' in proj._specific_template.keys():
            accessor = proj._specific_template['access_parameters_variables']

        # Finalize the declarations
        declaration = {
            'delay': declare_delay,
            'event_driven': declare_event_driven,
            'rng': declare_rng,
            'parameters_variables': declare_parameters_variables,
            'additional': declare_additional,
            'cuda_stream': ProjTemplate.cuda_stream if Global.config['paradigm']=="cuda" else ""
        }
        
        return declaration, accessor

#######################################################################
############## Initialize projection OMP ##############################
#######################################################################
    def init_parameters_variables(self, proj):

        # Is it a specific projection?
        if 'init_parameters_variables' in proj._specific_template.keys():
            return proj._specific_template['init_parameters_variables']

        # Learning by default
        code = ""

        # In case of GPUs we need to pre-allocate data fields
        # on the GPU, the weight variable w must be added additionally
        if Global.config['paradigm'] == "cuda":
            code += ProjTemplate.cuda_proj_base_data % {'id': proj.id}

        # choose initialization templates based on chosen paradigm
        attr_init_tpl = ProjTemplate.attribute_cpp_init[Global.config['paradigm']]

        # Initialize parameters
        for var in proj.synapse.description['parameters']:
            if var['name'] == 'w':
                continue
            init = 0.0 if var['ctype'] == 'double' else 0
            code += attr_init_tpl[var['locality']] % { 'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init, 'attr_type': 'parameter' }

        # Initialize variables
        for var in proj.synapse.description['variables']:
            if var['name'] == 'w':
                continue
            init = 0.0 if var['ctype'] == 'double' else 0
            code += attr_init_tpl[var['locality']] % { 'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init, 'attr_type': 'variable' }

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
############## Compute sum rate-coded OMP #############################
#######################################################################
    def computesum_rate_openmp(self, proj):
        code = ""    
        ids = {
                'id_proj' : proj.id, 
                'target': proj.target, 
                'id_post': proj.post.id, 
                'id_pre': proj.pre.id, 
                'local_index': "[i][j]", 
                'global_index': '[i]',
                'pre_index': '[pre_rank[i][j]]',
                'post_index': '[post_rank[i]]',
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
        }

        # Default variables needed in psp_code
        psp_prefix = """
        int nb_post; double sum;"""
        if 'psp_prefix' in proj._specific_template.keys():
            psp_prefix = proj._specific_template['psp_prefix']

        # Specific projection
        if 'psp_code' in proj._specific_template.keys():
            return psp_prefix, proj._specific_template['psp_code']


        # Retrieve the psp code
        if Global.config['num_threads'] < 2 or proj.post.size <= Global.OMP_MIN_NB_NEURONS: # No need to copy the pre-synaptic firing rates
            if not 'psp' in  proj.synapse.description.keys(): # default
                psp = """w[i][j] * pop%(id_pre)s.r[pre_rank[i][j]];""" % ids
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
                                'pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[delay[i][j]-1][pre_rank[i][j]]'% dict({'var': var}.items() + ids.items())
                            )
                        else:
                            psp = psp.replace(
                                'pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[delay[i][j]-1]'% dict({'var': var}.items() + ids.items())
                            )

                else: # Uniform delays
                    for var in delayed_variables:
                        if var in proj.pre.neuron_type.description['local']:
                            psp = psp.replace(
                                'pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[%(delay)s][pre_rank[i][j]]' % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            )
                        else:
                            psp = psp.replace(
                                'pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% dict({'var': var}.items() + ids.items()), 
                                'pop%(id_pre)s._delayed_%(var)s[%(delay)s]' % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            )

            omp_code = ""
            pre_copy = ""

        else: # OMP: make a local copy of pre.r for each thread
            if not 'psp' in  proj.synapse.description.keys(): # default
                psp = """w[i][j] * pop%(id_pre)s.r[pre_rank[i][j]];""" % ids
            else: # custom psp
                psp = (proj.synapse.description['psp']['cpp'] % ids)
            # Take delays into account if any
            if proj.max_delay > 1: # there is a delay
                if proj.uniform_delay == -1 : # Non-uniform delays
                    for var in list(set(proj.synapse.description['dependencies']['pre'])):
                        if var in proj.pre.neuron_type.description['local']:
                            psp = psp.replace(  
                                'pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% {'id_pre': proj.pre.id, 'var': var}, 
                                'pop%(id_pre)s._delayed_%(var)s[delay[i][j]-1][pre_rank[i][j]]' % {'id_pre': proj.pre.id, 'id_proj' : proj.id, 'var': var})
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
                            pre_copy += """std::vector<double> _pre_%(var)s = pop%(id_pre)s._delayed_%(var)s[%(delay)s];\n""" %{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            psp = psp.replace('pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% {'id_pre': proj.pre.id, 'var': var}, '_pre_%(var)s[pre_rank[i][j]]' % {'id_proj': proj.id, 'var': var})
                        else:
                            pre_copy += """double _pre_%(var)s = pop%(id_pre)s._delayed_%(var)s[%(delay)s];\n""" % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1), 'var': var}
                            psp = psp.replace('pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% {'id_pre': proj.pre.id, 'var': var}, '_pre_%(var)s' % {'id_proj': proj.id, 'var': var})

                        omp_code += '_pre_%(var)s, ' % {'id_proj': proj.id, 'var': var}

                    omp_code += "nb_post) schedule(dynamic)"
            else: # No delay
                pre_copy = ""; omp_code = "#pragma omp parallel for private(sum) firstprivate("
                for var in list(set(proj.synapse.description['dependencies']['pre'])):
                    if var in proj.pre.neuron_type.description['local']:
                        pre_copy += """        std::vector<double> _pre_%(var)s = pop%(id_pre)s.%(var)s;\n""" %{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        psp = psp.replace('pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% {'id_pre': proj.pre.id, 'var': var}, '_pre_%(var)s[pre_rank[i][j]]' % {'id_proj': proj.id, 'var': var})
                    else:
                        pre_copy += """        double _pre_%(var)s = pop%(id_pre)s.%(var)s;\n""" % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        psp = psp.replace('pop%(id_pre)s.%(var)s[pre_rank[i][j]]'% {'id_pre': proj.pre.id, 'var': var}, '_pre_%(var)s' % {'id_proj': proj.id, 'var': var})

                    omp_code += '_pre_%(var)s, ' % {'id_proj': proj.id, 'var': var}

                omp_code += "nb_post) schedule(dynamic)"
        
        # Generate the code depending on the operation
        sum_code = ProjTemplate.summation_operation[proj.synapse.operation] %{
            'pre_copy': pre_copy,
            'omp_code': omp_code,
            'psp': psp.replace(';', ''), 
            'id_post': proj.post.id,
            'target': proj.target,
        }

        # Finish the code
        code = """
        if (pop%(id_post)s._active){
%(code)s
        } // active
        """ % {'id_post': proj.post.id,
               'code': tabify(sum_code, 3),
               }

        return psp_prefix, code

#######################################################################
############## Compute sum rate-coded CUDA ############################
#######################################################################
    def computesum_rate_cuda(self, proj):
        """
        returns all data needed for compute postsynaptic sum kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
        code = ""

        psp = proj.synapse.description['psp']['cpp'] % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id}
        psp = psp.replace('[rk_pre]', '[rank_pre[j]]')

        body_code = ProjTemplate.cuda_psp_kernel % {
                                   'id': proj.id,
                                   'pre': proj.pre.id,
                                   'post': proj.post.id,
                                   'target': proj.target,
                                   'psp': psp
                                  }

        header_code = """__global__ void cuPop%(pre)s_Pop%(post)s_%(target)s_psp( int* pre_rank, int* nb_synapses, int *offsets, double *pre_r, double* w, double *sum_%(target)s );
""" % { 'pre': proj.pre.id,
        'post': proj.post.id,
        'target': proj.target,
      }

        call_code = ProjTemplate.cuda_psp_kernel_call % {
                                        'id': proj.id,
                                        'pre': proj.pre.id,
                                        'post': proj.post.id,
                                        'target': proj.target,
                                      }

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1:
                Global._error("only uniform delays are supported on GPUs.")
                exit(0)
            else:
                call_code = call_code.replace("gpu_r", "gpu_delayed_r["+str(proj.max_delay-1)+"]")

        return header_code, body_code, call_code


#######################################################################
############## Compute sum spiking OMP ################################
#######################################################################
    def computesum_spiking(self, proj):
        # Needed variables
        psp_prefix = """
        int nb_post, i, j, rk_j, rk_post, rk_pre;
        double sum;"""
        if 'psp_prefix' in proj._specific_template.keys():
            psp_prefix = proj._specific_template['psp_prefix']

        # Specific projection
        if 'psp_code' in proj._specific_template.keys():
            return psp_prefix, proj._specific_template['psp_code']

        # Basic tags
        ids = {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target, 'local_index': "[i][j]", 'global_index': '[i]'} 

        # Determine the mode of synaptic transmission
        continous_transmission = False
        if 'psp' in  proj.synapse.description.keys(): # continous
            continous_transmission = True

        ####################################################
        # Event-driven summation of g_target
        ####################################################
        # Strings
        updated_variables_list = []
        g_target = ""; g_target_bounds = ""
        g_target_code = ""    

        # Analyse all elements of pre_spike
        for eq in proj.synapse.description['pre_spike']:
            # g_target is treated differently
            # Must be at the end of the equations
            if eq['name'] == 'g_target': 
                g_target = eq['cpp'].split('=')[1] % ids
                g_target_code = """
            // Increase the post-synaptic conductance %(eq)s
            pop%(id_post)s.g_%(target)s[post_rank[i]] += %(g_target)s ;
""" % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target, 'g_target': g_target, 'eq': eq['eq']}
                # Determine bounds
                for key, val in eq['bounds'].items():
                    if not key in ['min', 'max']:
                        continue
                    try:
                        value = str(float(val))
                    except: # TODO: more complex operations
                        value = "%(name)s%(locality)s" % {'id_proj' : proj.id, 'name': val, 'locality': '[i]' if val in proj.synapse.description['global'] else '[i][j]'}

                    g_target += """
if (pop%(id_post)s.g_%(target)s[post_rank[i]] %(op)s %(val)s)
    pop%(id_post)s.g_%(target)s[post_rank[i]] = %(val)s;
""" % {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'target': proj.target, 'op': "<" if key == 'min' else '>', 'val': value }

            elif eq['name'] == 'w': # Surround it by the learning flag
                updated_variables_list += """
// Plasticity of w can be disabled
if(_learning){
    // %(eq)s
    %(cpp)s 
    %(bounds)s
}
""" % {'eq': eq['eq'], 'cpp': eq['cpp'] % ids, 'bounds': get_bounds(eq) % ids}

            else: # Normal synaptic variable
                updated_variables_list += """
// %(eq)s
%(cpp)s 
%(bounds)s""" % {'eq': eq['eq'], 'cpp': eq['cpp'] % ids, 'bounds': get_bounds(eq) % ids}


        # Generate the default post-conductance increase
        # default g_target += w
        if not continous_transmission and g_target == "": 
            g_target_code = """
            // Increase the post-synaptic conductance g_target += w
            pop%(id_post)s.g_%(target)s[post_rank[i]] += w[i][j];
""" % ids


        # Event-driven integration of synaptic variables
        has_exact = False
        event_driven_code = ''
        for var in proj.synapse.description['variables']:
            if var['method'] == 'event-driven':
                has_exact = True
                event_driven_code += """
            // %(eq)s
            %(exact)s
""" % {'eq': var['eq'], 'exact': var['cpp'].replace('(t)', '(t-1)') %{'id_proj' : proj.id, 'local_index': "[i][j]", 'global_index': '[i]'}}
        if has_exact:
                event_driven_code += """
            // Update the last event for the synapse 
            _last_event[i][j] = t;
""" % {'id_proj' : proj.id, 'exact': var['cpp']}

            
        # Generate code for pre-spike variables
        pre_event = ""
        if len(updated_variables_list) > 0: 
            code = ""
            for var in updated_variables_list:
                code += var
            code = tabify(code, 4)
            if not has_exact:
                pre_event += """
            // Event-based variables should not be updated when the postsynaptic neuron fires.
            if(pop%(id_post)s.last_spike[post_rank[i]] != t-1){
%(pre_event)s
            }
"""% {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'pre_event': code}
            else:
                pre_event += """
            // Pre-spike events with exact integration should always be evaluated...
            {
%(pre_event)s
            }
"""% {'id_proj' : proj.id, 'id_post': proj.post.id, 'id_pre': proj.pre.id, 'pre_event': code}

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1 : # Non-uniform delays
                Global._error('Non-uniform delays are not yet possible for spiking networks.')
                exit(0)
            else: # Uniform delays
                pre_array = "pop%(id_pre)s._delayed_spike[%(delay)s]" % {'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
        else:
            pre_array = "pop%(id_pre)s.spiked" % ids

        # No need for openmp if less than 100 post neurons
        omp_code = ""
        if Global.config['num_threads']>1:
            if proj.post.size > Global.OMP_MIN_NB_NEURONS and len(updated_variables_list) > 0:
                omp_code = """#pragma omp parallel for firstprivate(nb_post, inv_post) private(i, j)"""%{'id_proj' : proj.id}  
        
        # Generate the whole code block
        code = ""    
        if g_target_code != "" or pre_event != "": 
            code = """
// Event-based summation
if (pop%(id_post)s._active){
    std::vector< std::pair<int, int> > inv_post;
    // Iterate over all incoming spikes
    for(int _idx_j = 0; _idx_j < %(pre_array)s.size(); _idx_j++){
        rk_j = %(pre_array)s[_idx_j];
        inv_post = inv_pre_rank[rk_j];
        nb_post = inv_post.size();
        // Iterate over connected post neurons
        %(omp_code)s
        for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
            // Retrieve the correct indices
            i = inv_post[_idx_i].first;
            j = inv_post[_idx_i].second;
%(event_driven)s
%(g_target)s
%(pre_event)s
        }
    }
} // active
"""%{   'id_pre': proj.pre.id, 
        'id_post': proj.post.id, 
        'pre_array': pre_array,
        'pre_event': pre_event, 
        'g_target': g_target_code, 
        'omp_code': omp_code,
        'event_driven': event_driven_code 
    }

        # Add tabs
        code = tabify(code, 2)

        ####################################################
        # Not even-driven summation of psp: like rate-coded
        ####################################################
        if 'psp' in  proj.synapse.description.keys(): # not event-based
            # Compute it as if it were rate-coded
            psp_code = self.computesum_rate_openmp(proj)[1]
            # Change _sum_target into g_target
            psp_code = psp_code.replace(
                'pop%(id_post)s._sum_%(target)s[post_rank[i]]' % {'id_post': proj.post.id, 'target': proj.target},
                'pop%(id_post)s.g_%(target)s[post_rank[i]]' % {'id_post': proj.post.id, 'target': proj.target}
            )
            # Add it to the main code
            code += """
        // PSP-based summation"""
            code += psp_code

        # Annotate code
        if self._prof_gen:
            code = self._prof_gen.annotate_computesum_spiking_omp(code)

        return psp_prefix, code

#######################################################################
############## Post-synaptic event spiking OMP ########################
#######################################################################
    def postevent(self, proj):
        if proj.synapse.type == "rate":
            return "",""

        if proj.synapse.description['post_spike'] == []:
            return "",""


        ids = {
                'id_proj' : proj.id, 
                'target': proj.target, 
                'id_post': proj.post.id, 
                'id_pre': proj.pre.id, 
                'local_index': "[i][j]", 
                'global_index': '[i]',
                'pre_index': '[pre_rank[i][j]]',
                'post_index': '[post_rank[i]]',
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
        }

        post_event_prefix = """
        int i, j, rk_post, nb_pre;"""

        # Event-driven integration
        has_event = False
        for var in proj.synapse.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True

        # Generate event-driven code
        event_driven_code = ""
        if has_event_driven:
            for var in proj.synapse.description['variables']:
                if var['method'] == 'event-driven':
                    event_driven_code += '// ' + var['eq'] + '\n'
                    event_driven_code += var['cpp'] % ids + '\n'
            event_driven_code += """
// Update the last event for the synapse
_last_event[i][j] = t;
""" 
            event_driven_code = tabify(event_driven_code, 3)

        # Gather the equations
        post_code = ""
        for eq in proj.synapse.description['post_spike']:
            post_code += '// ' + eq['eq'] + '\n'
            post_code += eq['cpp'] % ids + '\n'
            post_code += get_bounds(eq) % ids + '\n'
        post_code = tabify(post_code, 3)

        # OMP code
        if Global.config['num_threads']>1:
            omp_code = '#pragma omp parallel for private(j) firstprivate(i, nb_pre)' if proj.post.size > Global.OMP_MIN_NB_NEURONS else ''
        else:
            omp_code = ""

        # Generate the code block
        code = """
if(_learning && pop%(id_post)s._active){
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
"""%{'id_post': proj.post.id, 
    'post_event': post_code, 
    'event_driven': event_driven_code,
    'omp_code': omp_code}

        return post_event_prefix, tabify(code, 2)


#######################################################################
############## Synaptic variables OMP #################################
#######################################################################
    def update_synapse(self, proj):
        prefix = """
        int rk_post, rk_pre;"""

        # Global variables
        global_eq = generate_equation_code(proj.id, proj.synapse.description, 'global', 'proj', padding=2) %{
                'id_proj' : proj.id, 
                'target': proj.target, 
                'id_post': proj.post.id, 
                'id_pre': proj.pre.id, 
                'local_index': '[i][j]', 
                'global_index': '[i]',
                'pre_index': '[rk_pre]',
                'post_index': '[rk_post]',
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
        }

        # Local variables
        local_eq =  generate_equation_code(proj.id, proj.synapse.description, 'local', 'proj', padding=3) %{
                'id_proj' : proj.id, 
                'target': proj.target, 
                'id_post': proj.post.id, 
                'id_pre': proj.pre.id, 
                'local_index': "[i][j]", 
                'global_index': '[i]',
                'pre_index': '[rk_pre]',
                'post_index': '[rk_post]',
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
        } 

        # Skip generation if 
        if local_eq.strip() == '' and global_eq.strip() == '' :
            return "", ""

        # OpenMP
        omp_code = ""
        if Global.config['num_threads']>1 and proj.post.size > Global.OMP_MIN_NB_NEURONS:
            omp_code = '#pragma omp parallel for private(rk_pre, rk_post)'

        # Code template  
        if local_eq.strip() != "": # local synapses are updated
            code = """
if(_learning && pop%(id_post)s._active){
    %(omp_code)s
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i];
    %(global)s
        for(int j = 0; j < pre_rank[i].size(); j++){
            rk_pre = pre_rank[i][j];
    %(local)s
        }
    }
}
"""%{   'global': global_eq, 
        'local': local_eq,
        'id_post': proj.post.id, 
        'omp_code': omp_code
    }
        else: # Only global variables
            code = """
if(_learning && pop%(id_post)s._active){
    %(omp_code)s
    for(int i = 0; i < post_rank.size(); i++){
        rk_post = post_rank[i];
    %(global)s
    }
}
"""%{   'global': global_eq, 
        'id_post': proj.post.id, 
        'omp_code': omp_code
    }

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1 : # Non-uniform delays
                for var in list(set(proj.synapse.description['dependencies']['pre'])):
                    if var in proj.pre.neuron_type.description['local']:
                        code = code.replace(
                            'pop%(id_pre)s.%(var)s[rk_pre]'%{'id_pre': proj.pre.id, 'var': var}, 
                            'pop%(id_pre)s._delayed_%(var)s[delay[i][j]-1][rk_pre]'%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        )
                    else:
                        code = code.replace(
                            'pop%(id_pre)s.%(var)s[rk_pre]'%{'id_pre': proj.pre.id, 'var': var}, 
                            'pop%(id_pre)s._delayed_%(var)s[delay[i][j]-1]'%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'var': var}
                        )
                # Access to spike arrays should also be delayed? TODO check
                code = code.replace(
                    'pop%(id_pre)s.spiked['%{'id_pre': proj.pre.id},
                    'pop%(id_pre)s._delayed_spike[delay[i][j]-1]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id}
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
                # Access to spike arrays should also be delayed? TODO check
                code = code.replace(
                    'pop%(id_pre)s.spiked['%{'id_pre': proj.pre.id},
                    'pop%(id_pre)s._delayed_spike[%(delay)s]['%{'id_proj' : proj.id, 'id_pre': proj.pre.id, 'delay': str(proj.uniform_delay-1)}
                )

        # profiling annotation
        if self._prof_gen:
            code = self._prof_gen.annotate_update_synapse_omp(code)

        # Return the code block
        return prefix, tabify(code, 2)


#######################################################################
############## Random Distributions OMP ###############################
#######################################################################
    def init_random_distributions(self, proj):
        # Is it a specific population?
        if 'init_rng' in proj._specific_template.keys():
            return proj._specific_template['init_rng']

        code = ""
        for rd in proj.synapse.description['random_distributions']:
            code += """    %(rd_name)s = std::vector< std::vector<double> >(post_rank.size(), std::vector<double>());
    for(int i=0; i<post_rank.size(); i++){
        %(rd_name)s[i] = std::vector<double>(pre_rank[i].size(), 0.0);
    }
    dist_%(rd_name)s = %(rd_init)s;
""" % {'id_proj': proj.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': proj.id}}
        return code

    def update_random_distributions(self, proj):
        # Is it a specific population?
        if 'update_rng' in proj._specific_template.keys():
            return proj._specific_template['update_rng']

        code = ""
        if len(proj.synapse.description['random_distributions']) > 0:
            code += """
    // RD of proj%(id_proj)s
    for(int i = 0; i < post_rank.size(); i++){
        for(int j = 0; j < pre_rank[i].size(); j++){
"""% {'id_proj': proj.id}

            for rd in proj.synapse.description['random_distributions']:
                code += """
            %(rd_name)s[i][j] = dist_%(rd_name)s(rng);""" % {'id_proj': proj.id, 'rd_name': rd['name']}

            code += """
        }
    }
"""
        return code


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
        
        # Spiking networks must update the inv_pre_rank array
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

#######################################################################
############## CUDA stuff #############################################
#######################################################################
    def _cuda_memory_transfers(self, proj):
        host_device_transfer = ""
        device_host_transfer = ""

        # transfer of variables
        host_device_transfer += """\n    // host to device transfers for proj%(id)s\n""" % { 'id': proj.id }
        for attr in proj.synapse.description['parameters']+proj.synapse.description['variables']:
            if attr['name'] in proj.synapse.description['local']:
                host_device_transfer += """
        // %(name)s: local
        if ( proj%(id)s.%(name)s_dirty )
        {
            std::vector<double> flat_proj%(id)s_%(name)s = flattenArray<double>(proj%(id)s.%(name)s);
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

######################################
### Code generation
######################################
def get_bounds(param):
    "Analyses the bounds of a variable and returns the corresponding code."
    code = ""
    # Min-Max bounds
    for bound, val in param['bounds'].items():
        if bound == "init":
            continue

        code += """if(%(var)s%(index)s %(operator)s %(val)s)
    %(var)s%(index)s = %(val)s;
""" % {'index': "[i][j]" ,
       'var' : param['name'], 
       'val' : val, 
       'operator': '<' if bound=='min' else '>'
       }
    return code
