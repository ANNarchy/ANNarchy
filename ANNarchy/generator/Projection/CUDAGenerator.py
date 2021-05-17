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
"""
The CUDAGenerator is responsible for the complete code generation process
in ANNarchy to support CUDA devices. Generate the header for a Population
object to run either on a Nvidia GPU using Nvidia SDK > 5.0 and CC > 2.0
"""
import re

from .ProjectionGenerator import ProjectionGenerator, get_bounds
from .CUDATemplates import cuda_templates
from .Connectivity import CUDAConnectivity

from ANNarchy.core import Global
from ANNarchy.core.Population import Population
from ANNarchy.generator.Utils import generate_equation_code, tabify, check_and_apply_pow_fix

from ANNarchy.generator.Population.PopulationGenerator import PopulationGenerator

class CUDAGenerator(ProjectionGenerator, CUDAConnectivity):
    """
    As stated in module description, inherits from ProjectionGenerator
    and implements abstract functions.
    """
    _templates = cuda_templates

    def __init__(self, profile_generator, net_id):
        # The super here calls all the base classes, so first
        # ProjectionGenerator and afterwards CUDAConnectivity
        super(CUDAGenerator, self).__init__(profile_generator, net_id)

    def header_struct(self, proj, annarchy_dir):
        """
        Generate the codes for the pop[id].hpp file. This file contains
        the c-style structure with all data members and equation codes (in
        case of openMP).
        """
        # configure Connectivity base class
        self.configure(proj)

        # Generate declarations and accessors for the variables
        decl, accessor = self._declaration_accessors(proj)

        # concurrent streams
        decl['cuda_stream'] = cuda_templates['cuda_stream']

        # Initiliaze the projection
        init_parameters_variables = self._init_parameters_variables(proj)

        variables_body, variables_header, variables_call = self._update_synapse(proj)

        # Update the random distributions
        init_rng = self._init_random_distributions(proj)

        # Post event
        post_event_body, post_event_header, post_event_call = self._post_event(proj)

        # Compute sum is the trickiest part
        psp_header, psp_body, psp_call = self._computesum_rate(proj) if proj.synapse_type.type == 'rate' else self._computesum_spiking(proj)

        # Detect event-driven variables
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True

        # Detect delays to eventually generate the code
        has_delay = proj.max_delay > 1

        # Connectivity matrix
        connectivity_matrix = self._connectivity(proj)

        # Memory transfers
        host_device_transfer, device_host_transfer = self._memory_transfers(proj)

        # Memory management
        determine_size_in_bytes = self._determine_size_in_bytes(proj)
        clear_container = self._clear_container(proj)

        # Local functions
        host_local_func, device_local_func = self._local_functions(proj)
        decl['parameters_variables'] += host_local_func

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
        cuda_flattening = ""
        if 'include_additional' in proj._specific_template.keys():
            include_additional = proj._specific_template['include_additional']
        if 'struct_additional' in proj._specific_template.keys():
            struct_additional = proj._specific_template['struct_additional']
        if 'init_additional' in proj._specific_template.keys():
            init_additional = proj._specific_template['init_additional']
        if 'access_additional' in proj._specific_template.keys():
            access_additional = proj._specific_template['access_additional']

        inverse_connectivity_matrix = connectivity_matrix['init_inverse'] % {
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'post_size': proj.post.size # only needed by CSR
        }

        if 'cuda_flattening' in proj._specific_template.keys():
            cuda_flattening = proj._specific_template['cuda_flattening']
        else:
            if proj._storage_format == "lil":
                cuda_flattening = self._templates['flattening'] % {
                    'id_post':proj.post.id
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
            'declare_delay': decl['declare_delay'] if has_delay else "",
            'declare_event_driven': decl['event_driven'] if has_event_driven else "",
            'declare_rng': decl['rng'],
            'declare_parameters_variables': decl['parameters_variables'],
            'declare_cuda_stream': decl['cuda_stream'],
            'declare_additional': decl['additional'],
            'declare_profile': declare_profile,
            'init_connectivity_matrix': connectivity_matrix['init'] % {'float_prec': Global.config['precision']},
            'init_inverse_connectivity_matrix': inverse_connectivity_matrix,
            'init_event_driven': "",
            'init_rng': init_rng,
            'init_parameters_variables': init_parameters_variables,
            'init_additional': init_additional,
            'init_profile': init_profile,
            'access_connectivity_matrix': connectivity_matrix['accessor'],
            'access_parameters_variables': accessor,
            'access_additional': access_additional,
            'host_to_device': host_device_transfer,
            'device_to_host': device_host_transfer,
            'cuda_flattening': cuda_flattening,
            'determine_size': determine_size_in_bytes,
            'clear_container': clear_container
        }

        # Store the file in generate ( will be compared with files contained
        # in build/ later on )
        with open(annarchy_dir+'/generate/net'+str(self._net_id)+'/proj'+str(proj.id)+'.hpp', 'w') as ofile:
            ofile.write(final_code)

        # Build dictionary for inclusions in ANNarchy.cu
        proj_desc = {
            'include': """#include "proj%(id)s.hpp"\n""" % {'id': proj.id},
            'extern': """extern ProjStruct%(id)s proj%(id)s;\n"""% {'id': proj.id},
            'instance': """ProjStruct%(id)s proj%(id)s;\n"""% {'id': proj.id},
            'init': """    proj%(id)s.init_projection();\n""" % {'id' : proj.id}
        }

        proj_desc['psp_header'] = psp_header
        proj_desc['psp_body'] = psp_body
        proj_desc['psp_call'] = psp_call
        proj_desc['custom_func'] = device_local_func

        proj_desc['update_synapse_header'] = variables_header
        proj_desc['update_synapse_body'] = variables_body
        proj_desc['update_synapse_call'] = variables_call

        proj_desc['postevent_header'] = post_event_header
        proj_desc['postevent_body'] = post_event_body
        proj_desc['postevent_call'] = post_event_call

        proj_desc['host_to_device'] = tabify("proj%(id)s.host_to_device();" % {'id':proj.id}, 1)+"\n"
        proj_desc['device_to_host'] = tabify("proj%(id)s.device_to_host();" % {'id':proj.id}, 1)+"\n"

        return proj_desc

    def _clear_container(self, proj):
        """
        Override default implementation. We need host and device allocations to be destroyed.
        """
        host_code = super(CUDAGenerator, self)._clear_container(proj)

        device_code = "\n/* Free device allocations */\n\n"

        # Connectivity ( weights + indices )
        device_code += "// Connectivity\n"
        device_code += self._templates['connectivity_matrix']['clear']

        # Attributes
        device_code += "// Parameters \n"
        for attr in proj.synapse_type.description['parameters']:
            device_code += """cudaFree(gpu_%(name)s);\n""" % {'name': attr['name']}
        device_code += "// Variables \n"
        for attr in proj.synapse_type.description['variables']:
            device_code += """cudaFree(gpu_%(name)s);\n""" % {'name': attr['name']}

        return host_code + tabify(device_code, 2)

    def _computesum_rate(self, proj):
        """
        returns all data needed for compute postsynaptic sum kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
        # Default variables needed in psp_code
        psp_prefix = """
        int nb_post; %(float_prec)s sum;""" % {'float_prec': Global.config['precision']}
        if 'psp_prefix' in proj._specific_template.keys():
            psp_prefix = proj._specific_template['psp_prefix']

        # Specific projection
        if 'psp_header' in proj._specific_template.keys() and \
            'psp_body' in proj._specific_template.keys() and \
            'psp_call' in proj._specific_template.keys():
            return proj._specific_template['psp_header'], proj._specific_template['psp_body'], proj._specific_template['psp_call']

        # Dictionary of keywords to transform the parsed equations
        ids = {
            'id_proj' : proj.id,
            'target': proj.target,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,
            'local_index': "[j]",
            'semiglobal_index': '[i]',
            'global_index': '[0]',
            'pre_index': '[rank_pre[j]]',
            'post_index': '[post_rank[i]]',
            'pre_prefix': 'pre_',
            'post_prefix': 'post_',
            'delay_nu' : '[delay[j]-1]', # non-uniform delay
            'delay_u' : '[' + str(proj.uniform_delay-1) + ']' # uniform delay
        }

        #
        # Retrieve the PSP
        add_args_header = ""
        add_args_call = ""
        if not 'psp' in  proj.synapse_type.description.keys(): # default
            psp = """%(preprefix)s.r%(pre_index)s * w%(local_index)s;"""
        else: # custom psp
            psp = (proj.synapse_type.description['psp']['cpp'])

            # update dependencies
            for dep in proj.synapse_type.description['psp']['dependencies']:
                if dep == "w":
                    continue

                _, attr = self._get_attr_and_type(proj, dep)
                attr_ids = {
                    'id_proj': proj.id,
                    'type': attr['ctype'],
                    'name': attr['name']
                }
                add_args_header += ", %(type)s* %(name)s" % attr_ids
                add_args_call += ", proj%(id_proj)s.gpu_%(name)s" % attr_ids

        # Special case where w is a single value
        if proj._has_single_weight():
            psp = re.sub(
                r'([^\w]+)w%\(local_index\)s',
                r'\1w',
                ' ' + psp
            )

        # connectivity, yet only CSR
        conn_header = "int* rank_pre, int *row_ptr, %(float_prec)s *pre_r, %(float_prec)s* w" % {'float_prec': Global.config['precision']}
        conn_call = "proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, pop%(id_pre)s.gpu_r, proj%(id_proj)s.gpu_w " % {'id_proj': proj.id, 'id_pre': proj.pre.id}

        #
        # finish the kernel etc.
        psp = proj.synapse_type.description['psp']['cpp'] % ids
        operation = proj.synapse_type.operation

        body_code = self._templates['rate_psp']['body'][operation] % {
            'float_prec': Global.config['precision'],
            'id_proj': proj.id,
            'conn_args': conn_header,
            'target_arg': "sum_"+proj.target,
            'add_args': add_args_header,
            'psp': psp,
            'thread_init': self._templates['rate_psp']['thread_init'][Global.config['precision']][operation]
        }
        header_code = self._templates['rate_psp']['header'] % {
            'float_prec': Global.config['precision'],
            'id': proj.id,
            'conn_args': conn_header,
            'target_arg': "sum_"+proj.target,
            'add_args': add_args_header
        }
        call_code = self._templates['rate_psp']['call'] % {
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'conn_args': conn_call,
            'target': proj.target,
            'target_arg': ", pop%(id_post)s.gpu__sum_%(target)s" % {'id_post': proj.post.id, 'target': proj.target},
            'add_args': add_args_call,
            'float_prec': Global.config['precision']
        }

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1:
                Global._error("Only uniform delays are supported on GPUs.")

            else:
                # TODO: replace by regex
                call_code = call_code.replace("gpu_r,", "gpu_delayed_r[proj"+str(proj.id)+".delay-1],")

        # Profiling
        if self._prof_gen:
            call_code = self._prof_gen.annotate_computesum_rate(proj, call_code)

        return header_code, body_code, call_code

    def _computesum_spiking(self, proj):
        """
        Generate code for the spike propagation. As ANNarchy supports a set of
        different data structures, this method split in up into several sub
        functions.

        In contrast to _computsum_rate() the spike propagation kernel need
        to implement the signal transmission (event- as well as continous)
        and also the equations filled in the 'pre-spike' field of synapse
        desctiption.
        """
        # Specific template ?
        if 'header' in proj._specific_template.keys() and \
           'body' in proj._specific_template.keys() and \
           'call' in proj._specific_template.keys():
            try:
                header = proj._specific_template['header']
                body = proj._specific_template['body']
                call = proj._specific_template['call']
            except KeyError:
                Global._error('header,spike_count body and call should be overwritten')
            return header, body, call

        # some variables needed for the final templates
        psp_code = ""
        kernel_args = ""
        kernel_args_call = ""

        pre_spike_code = ""
        kernel_deps = []

        if proj.max_delay > 1 and proj.uniform_delay == -1:
            Global._error("Non-uniform delays are not supported yet on GPUs.")

        # some basic definitions
        ids = {
            # identifiers
            'id_proj' : proj.id,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,

            # common for all equations
            'local_index': "[syn_idx]",
            'semiglobal_index': '[post_rank]',
            'global_index': '[0]',
            'float_prec': Global.config['precision'],

            # psp specific
            'pre_prefix': 'pre_',
            'post_prefix': 'post_',
            'pre_index': '[col_idx[syn_idx]]',
            'post_index': '[post_rank]'
        }

        #
        # All statements in the 'pre_spike' field of synapse description
        #
        for var in proj.synapse_type.description['pre_spike']:
            if var['name'] == "g_target":   # synaptic transmission
                # compute psp
                psp_code += "%(float_prec)s tmp = %(psp)s\n" % {
                    'psp': var['cpp'].split('=')[1] % ids,
                    'float_prec': Global.config['precision']
                }
                # apply to all targets
                target_list = proj.target if isinstance(proj.target, list) else [proj.target]
                for target in sorted(list(set(target_list))):
                    psp_code += "atomicAdd(&g_%(target)s[post_rank], tmp);\n" % {'target': target}

            else:
                condition = ""
                # Check conditions to update the variable
                if var['name'] == 'w': # Surround it by the learning flag
                    condition = "plasticity"

                # Flags avoids pre-spike evaluation when post fires at the same time
                if 'unless_post' in var['flags']:
                    simultaneous = "pop%(id_pre)s_last_spike[_pr] != pop%(id_post)s_last_spike[%(semiglobal_index)s]" % ids
                    if condition == "":
                        condition = simultaneous
                    else:
                        condition += "&&(" + simultaneous + ")"

                eq_dict = {
                    'eq': var['eq'],
                    'cpp': var['cpp'] % ids,
                    'bounds': get_bounds(var) % ids,
                    'condition': condition,
                }

                # Generate the code
                if condition != "":
                    pre_spike_code += """
// unless_post can prevent evaluation of presynaptic variables
if(%(condition)s){
// %(eq)s
%(cpp)s
%(bounds)s
}
""" % eq_dict
                else: # Normal synaptic variable
                    pre_spike_code += """
// %(eq)s
%(cpp)s
%(bounds)s""" % eq_dict

            # Update the dependencies
            kernel_deps.append(var['name']) # right side
            for dep in var['dependencies']: # left side
                kernel_deps.append(dep)

        #
        # Event-driven integration of synaptic variables
        #
        has_exact = False
        event_driven_code = ''
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_exact = True
                event_dict = {
                    'eq': var['eq'],
                    'exact': var['cpp'].replace('(t)', '(t-1)') % ids
                }
                event_driven_code += """
    // %(eq)s
    %(exact)s
""" % event_dict
                # add the dependencies to kernel dependencies
                for dep in var['dependencies']:
                    kernel_deps.append(dep)

        # Does an event-driven variable occur?
        if has_exact:
            event_driven_code += """
    // Update the last event for the synapse
    _last_event%(local_index)s = t;
""" % ids

            # event-driven requires access to last event variable
            kernel_args += ", long* _last_event"
            kernel_args_call += ", proj%(id_proj)s._gpu_last_event"  % ids

        # Add pre- and post-synaptic population dependencies
        pre_post_deps = list(set(proj.synapse_type.description['dependencies']['pre'] + proj.synapse_type.description['dependencies']['post']))
        pre_post_args = self._gen_kernel_args(proj, pre_post_deps, pre_post_deps)
        kernel_args += pre_post_args[0]
        kernel_args_call += pre_post_args[1]

        # Add synaptic variables to kernel arguments
        kernel_deps = list(set(kernel_deps)) # sort out doublings
        for dep in kernel_deps:
            if dep == "w" or dep == "g_target":
                # already contained
                continue

            _, attr = self._get_attr_and_type(proj, dep)
            attr_ids = {
                'id_proj': proj.id,
                'name': attr['name'],
                'type': attr['ctype']
            }
            kernel_args += ", %(type)s* %(name)s" % attr_ids
            kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % attr_ids

        #
        # Finally, fill the templates
        #
        if 'psp' in  proj.synapse_type.description.keys(): # not event-based
            # transfrom psp equation
            psp_code = proj.synapse_type.description['psp']['cpp']

            # update dependencies
            for dep in proj.synapse_type.description['psp']['dependencies']:
                if dep == "w":
                    continue

                _, attr = self._get_attr_and_type(proj, dep)
                attr_ids = {
                    'id_proj': proj.id,
                    'type': attr['ctype'],
                    'name': attr['name']
                }
                kernel_args += ", %(type)s* %(name)s" % attr_ids
                kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % attr_ids

            psp_code = proj.synapse_type.description['psp']['cpp'] % ids

            # select the correct template
            template = self._templates['spike_transmission']['continous'][proj._storage_order]

            call = ""
            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                call += template['call'] % {
                    'id_proj': proj.id,
                    'id_pre': proj.pre.id,
                    'id_post': proj.post.id,
                    'target_arg': ', pop%(id_post)s.gpu_g_%(target)s' % {'id_post': proj.post.id, 'target': target},
                    'target': target,
                    'kernel_args': kernel_args_call,
                    'float_prec': Global.config['precision']
                }
            body = template['body'] % {
                'id_proj': proj.id,
                'target_arg': proj.target,
                'kernel_args':  kernel_args,
                'psp': psp_code,
                'pre_code': tabify(pre_spike_code, 3),
                'float_prec': Global.config['precision']
            }
            header = template['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args,
                'target_arg': 'g_'+proj.target,
                'float_prec': Global.config['precision']
            }

        else: # event-based
            # select the correct template
            template = self._templates['spike_transmission']['event_driven'][proj._storage_order]

            # Connectivity description
            if proj._storage_order == "post_to_pre":
                conn_header = "int* col_ptr, int* row_idx, int* inv_idx, %(float_prec)s *w" % ids
                conn_call = "proj%(id_proj)s.gpu_col_ptr, proj%(id_proj)s.gpu_row_idx, proj%(id_proj)s.gpu_inv_idx, proj%(id_proj)s.gpu_w" % ids
            else:
                conn_call = "proj%(id_proj)s._gpu_row_ptr, proj%(id_proj)s._gpu_col_idx, proj%(id_proj)s.gpu_w" % ids
                conn_body = "int* row_ptr, int* col_idx, %(float_prec)s* w" % ids
                conn_header = "int* row_ptr, int* col_idx, %(float_prec)s *w" % ids

            # Population sizes
            pre_size = proj.pre.size if isinstance(proj.pre, Population) else proj.pre.population.size
            post_size = proj.post.size if isinstance(proj.post, Population) else proj.post.population.size

            targets_call = ""
            targets_header = ""

            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                targets_call += ", pop%(id_post)s.gpu_g_"+target
                targets_header += (", %(float_prec)s* g_"+target) % {'float_prec': Global.config['precision']}

            call = template['call'] % {
                'id_proj': proj.id,
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'target': target_list[0],
                'kernel_args': kernel_args_call  % {'id_post': proj.post.id, 'target': target},
                'conn_args': conn_call + targets_call % {'id_post': proj.post.id}
            }
            body = template['body'] % {
                'id': proj.id,
                'float_prec': Global.config['precision'],
                'conn_arg': conn_header + targets_header,
                'kernel_args': kernel_args,
                'event_driven': tabify(event_driven_code, 2),
                'psp': tabify(psp_code, 4),
                'pre_event': tabify(pre_spike_code, 4),
                'pre_size': pre_size,
                'post_size': post_size,
            }
            header = template['header'] % {
                'id': proj.id,
                'float_prec': Global.config['precision'],
                'conn_header': conn_header + targets_header,
                'kernel_args': kernel_args
            }

        return header, body, call


    def _declaration_accessors(self, proj):
        """
        Extend basic declaration statements by CUDA streams.
        """
        declaration, accessor = ProjectionGenerator._declaration_accessors(self, proj)

        declaration['cuda_stream'] = cuda_templates['cuda_stream']
        return declaration, accessor

    @staticmethod
    def _select_deps(proj, locality):
        """
        Dependencies of synaptic equations consist of several components:

        * access to pre- or post-population
        * variables / parameters of the projection
        * pre- or post-spike event

        Return:

        * syn_deps      list of all dependencies
        * neur_deps     list of dependencies part of neurons
        """
        syn_deps = []

        # Access to pre- or postsynaptic neurons
        neur_deps = list(set(proj.synapse_type.description['dependencies']['pre'] +
                             proj.synapse_type.description['dependencies']['post']))
        for dep in neur_deps:
            syn_deps.append(dep)

        # Variables
        for var in proj.synapse_type.description['variables']:
            if var['eq'] == '':
                continue # nothing to do here

            if var['locality'] == locality:
                syn_deps.append(var['name'])
                for dep in var['dependencies']:
                    syn_deps.append(dep)

        # Random distributions
        for rd in proj.synapse_type.description['random_distributions']:
            for dep in rd['dependencies']:
                syn_deps += dep

        syn_deps = list(set(syn_deps))
        return syn_deps, neur_deps

    @staticmethod
    def _gen_kernel_args(proj, pop_deps, deps):
        """
        The header and function definitions as well as the call statement need
        to be extended with the additional variables.
        """
        kernel_args = ""
        kernel_args_call = ""

        for dep in deps:
            if dep in pop_deps:

                if dep in proj.synapse_type.description['dependencies']['pre']:
                    attr_type, attr_dict = PopulationGenerator._get_attr_and_type(proj.pre, dep)
                    ids = {
                        'type': attr_dict['ctype'],
                        'name': attr_dict['name'],
                        'id': proj.pre.id
                    }
                    kernel_args += ", %(type)s* pre_%(name)s" % ids
                    kernel_args_call += ", pop%(id)s.gpu_%(name)s" % ids

                if dep in proj.synapse_type.description['dependencies']['post']:
                    attr_type, attr_dict = PopulationGenerator._get_attr_and_type(proj.post, dep)
                    ids = {
                        'type': attr_dict['ctype'],
                        'name': attr_dict['name'],
                        'id': proj.post.id
                    }
                    kernel_args += ", %(type)s* post_%(name)s" % ids
                    kernel_args_call += ", pop%(id)s.gpu_%(name)s" % ids

            else:
                attr_type, attr_dict = ProjectionGenerator._get_attr_and_type(proj, dep)

                if attr_type == "var" or attr_type == "par":
                    ids = {
                        'id_proj': proj.id,
                        'type': attr_dict['ctype'],
                        'name': attr_dict['name']
                    }
                    kernel_args += ", %(type)s* %(name)s" % ids
                    kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % ids

                elif attr_type == "rand":
                    ids = {
                        'id_proj': proj.id,
                        'type': 'curandState',
                        'name': attr_dict['name']
                    }
                    kernel_args += ", %(type)s* state_%(name)s" % ids
                    kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % ids

        #
        # global operations related to pre- and post-synaptic operations
        for glop in proj.synapse_type.description['pre_global_operations']:
            attr = PopulationGenerator._get_attr(proj.pre, glop['variable'])
            ids = {
                'id': proj.pre.id,
                'name': glop['variable'],
                'type': attr['ctype'],
                'func': glop['function']
            }
            kernel_args += ", %(type)s pre__%(func)s_%(name)s" % ids
            kernel_args_call += ", pop%(id)s._%(func)s_%(name)s" % ids

        for glop in proj.synapse_type.description['post_global_operations']:
            attr = PopulationGenerator._get_attr(proj.post, glop['variable'])
            ids = {
                'id': proj.post.id,
                'name': glop['variable'],
                'type': attr['ctype'],
                'func': glop['function']
            }
            kernel_args += ", %(type)s post__%(func)s_%(name)s" % ids
            kernel_args_call += ", pop%(id)s._%(func)s_%(name)s" % ids

        #
        # event-driven spike synapses require the access to last_spike member
        # of pre- and post-synaptic populations.
        if proj.synapse_type.type == "spike":
            kernel_args = ", long int* pre_last_spike, long int* post_last_spike" + kernel_args
            kernel_args_call = ", pop%(id_pre)s.gpu_last_spike, pop%(id_post)s.gpu_last_spike" % {'id_pre': proj.pre.id, 'id_post': proj.post.id} + kernel_args_call

        return kernel_args, kernel_args_call

    def _header_structural_plasticity(self, proj):
        Global._error("Structural Plasticity is not supported on GPUs yet.")

    def _local_functions(self, proj):
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
        if len(proj.synapse_type.description['functions']) == 0:
            return "", ""

        host_code = ""
        device_code = ""
        for func in proj.synapse_type.description['functions']:
            cpp_func = func['cpp'] + '\n'

            host_code += cpp_func
            # TODO: improve code
            if (Global.config["precision"]=="float"):
                device_code += cpp_func.replace('float' + func['name'], '__device__ float proj%(id)s_%(func)s' % {'id': proj.id, 'func': func['name']})
            else:
                device_code += cpp_func.replace('double '+ func['name'], '__device__ double proj%(id)s_%(func)s' % {'id': proj.id, 'func':func['name']})

        return host_code, check_and_apply_pow_fix(device_code)

    def _replace_local_funcs(self, proj, glob_eqs, semiglobal_eqs, loc_eqs):
        """
        As the local functions can be occur repeatadly in the same file,
        there are modified with proj[id]_ to unique them. Now we need
        to adjust the call accordingly.
        """
        for func in proj.synapse_type.description['functions']:
            search_term = r"%(name)s\([^\(]*\)" % {'name': func['name']}

            func_occur = re.findall(search_term, glob_eqs)
            for term in func_occur:
                glob_eqs = loc_eqs.replace(term, term.replace(func['name'], 'proj'+str(proj.id)+'_'+func['name']))

            func_occur = re.findall(search_term, semiglobal_eqs)
            for term in func_occur:
                semiglobal_eqs = loc_eqs.replace(term, term.replace(func['name'], 'proj'+str(proj.id)+'_'+func['name']))

            func_occur = re.findall(search_term, loc_eqs)
            for term in func_occur:
                loc_eqs = loc_eqs.replace(term, term.replace(func['name'], 'proj'+str(proj.id)+'_'+func['name']))

        return glob_eqs, semiglobal_eqs, loc_eqs

    def _replace_random(self, loc_eqs, glob_eqs, random_distributions):
        """
        This method replace the variables rand_%(id)s in the parsed equations
        by the corresponding curand... term.
        """
        # double precision methods have a postfix
        prec_extension = "" if Global.config['precision'] == "float" else "_double"

        loc_pre = ""
        semi_pre = ""
        glob_pre = ""

        for dist in random_distributions:
            if dist['dist'] == "Uniform":
                dist_ids = {
                    'postfix': prec_extension,
                    'rd': dist['name'],
                    'min': dist['args'].split(',')[0],
                    'max': dist['args'].split(',')[1]
                }

                if dist["locality"] == "local":
                    term = """( curand_uniform%(postfix)s( &state_%(rd)s[j] ) * (%(max)s - %(min)s) + %(min)s )""" % dist_ids
                    loc_pre += "%(prec)s %(name)s = %(term)s;" % {'prec': Global.config['precision'], 'name': dist['name'], 'term': term}

                    # suppress local index
                    loc_eqs = loc_eqs.replace(dist['name']+"[j]", dist['name'])
                else:
                    # HD (17th May 2021): this path can not be reached as the parser rejects equations like:
                    # dw/dt = -w * Uniform(0,.1) : init=1, midpoint
                    raise NotImplementedError

            elif dist['dist'] == "Normal":
                dist_ids = {
                    'postfix': prec_extension, 'rd': dist['name'],
                    'mean': dist['args'].split(",")[0],
                    'sigma': dist['args'].split(",")[1]
                }

                if dist["locality"] == "local":
                    term = """( curand_normal%(postfix)s( &state_%(rd)s[j] ) * %(sigma)s + %(mean)s )""" % dist_ids
                    loc_pre += "%(prec)s %(name)s = %(term)s;" % {'prec': Global.config['precision'], 'name': dist['name'], 'term': term}

                    # suppress local index
                    loc_eqs = loc_eqs.replace(dist['name']+"[j]", dist['name'])
                else:
                    # HD (17th May 2021): this path can not be reached as the parser rejects equations like:
                    # dw/dt = -w * Uniform(0,.1) : init=1, midpoint
                    raise NotImplementedError

            elif dist['dist'] == "LogNormal":
                dist_ids = {
                    'postfix': prec_extension, 'rd': dist['name'],
                    'mean': dist['args'].split(',')[0],
                    'std_dev': dist['args'].split(',')[1]
                }

                if dist["locality"] == "local":
                    term = """( curand_log_normal%(postfix)s( &state_%(rd)s[j], %(mean)s, %(std_dev)s) )""" % dist_ids
                    loc_pre += "%(prec)s %(name)s = %(term)s;" % {'prec': Global.config['precision'], 'name': dist['name'], 'term': term}

                    # suppress local index
                    loc_eqs = loc_eqs.replace(dist['name']+"[j]", dist['name'])
                else:
                    # HD (17th May 2021): this path can not be reached as the parser rejects equations like:
                    # dw/dt = -w * Uniform(0,.1) : init=1, midpoint
                    raise NotImplementedError

            else:
                Global._error("Unsupported random distribution on GPUs: " + dist['dist'])

        # check which equation blocks we need to extend
        if len(loc_pre) > 0:
            loc_eqs = tabify(loc_pre, 2) + "\n" + loc_eqs
        if len(glob_pre) > 0:
            glob_eqs = tabify(glob_pre, 1) + "\n" + glob_eqs

        return loc_eqs, glob_eqs

    def _post_event(self, proj):
        """
        Post-synaptic event kernel for CUDA devices
        """
        if proj.synapse_type.type == "rate":
            return "", "", ""

        if proj.synapse_type.description['post_spike'] == []:
            return "", "", ""

        if proj._storage_format == "lil":
            ids = {
                'id_proj' : proj.id,
                'target': proj.target,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'local_index': "[j]",
                'semiglobal_index': '[i]',
                'global_index': '[0]',
                'pre_index': '[pre_rank[j]]',
                'post_index': '[post_rank[i]]',
                'pre_prefix': 'pre_',
                'post_prefix': ''
            }
        elif proj._storage_format == "csr":
            ids = {
                'id_proj' : proj.id,
                'target': proj.target,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'local_index': "[col_idx[j]]",
                'semiglobal_index': '[i]',
                'global_index': '',
                'pre_index': '[pre_rank[j]]',
                'post_index': '[post_rank[i]]',
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
            }
        else:
            raise NotImplementedError

        add_args_header = ""
        add_args_call = ""

        # Event-driven integration
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True

        # Generate event-driven code
        event_driven_code = ""
        event_deps = []
        if has_event_driven:
            # event-driven rely on last pre-synaptic event
            add_args_header += ", long* _last_event"
            add_args_call += ", proj%(id_proj)s._gpu_last_event" % {'id_proj': proj.id}

            for var in proj.synapse_type.description['variables']:
                if var['method'] == 'event-driven':
                    event_driven_code += '// ' + var['eq'] + '\n'
                    event_driven_code += var['cpp'] % ids + '\n'

                    for deps in var['dependencies']:
                        event_deps.append(deps)
            event_driven_code += """
// Update the last event for the synapse
_last_event%(local_index)s = t;
""" % {'local_index' : '[j]'}

        # Gather the equations
        post_code = ""
        post_deps = []
        for post_eq in proj.synapse_type.description['post_spike']:
            post_code += '// ' + post_eq['eq'] + '\n'
            if post_eq['name'] == 'w':
                post_code += "if(plasticity)\n"
            post_code += post_eq['cpp'] % ids + '\n'
            post_code += get_bounds(post_eq) % ids + '\n'

            # add dependencies, only right side!
            for deps in post_eq['dependencies']:
                post_deps.append(deps)
            # left side of equations is not part of dependencies
            post_deps.append(post_eq['name'])
        post_code = tabify(post_code, 2)

        # Create add_args for event-driven eqs and post_event
        kernel_deps = list(set(post_deps+event_deps)) # variables can occur in several eqs
        for dep in kernel_deps:
            if dep == "w":
                continue

            _, attr = self._get_attr_and_type(proj, dep)
            add_args_header += ', %(type)s* %(name)s' % {'type': attr['ctype'], 'name': attr['name']}
            add_args_call += ', proj%(id)s.gpu_%(name)s' % {'id': proj.id, 'name': attr['name']}

        if proj._storage_format == "lil":
            conn_header = "int* row_ptr, int* pre_ranks,"
            conn_call = ", proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank"
            templates = self._templates['post_event']['post_to_pre']
        elif proj._storage_format == "csr":
            conn_header = "int* row_ptr, int *col_idx, "
            conn_call = ", proj%(id_proj)s._gpu_row_ptr,  proj%(id_proj)s._gpu_col_idx"
            templates = self._templates['post_event']['pre_to_post']
        else:
            raise NotImplementedError


        postevent_header = templates['header'] % {
            'id_proj': proj.id,
            'conn_args': conn_header,
            'add_args': add_args_header,
            'float_prec': Global.config['precision']
        }

        postevent_body = templates['body'] % {
            'id_proj': proj.id,
            'conn_args': conn_header,
            'add_args': add_args_header,
            'event_driven': tabify(event_driven_code, 2),
            'post_code': post_code,
            'float_prec': Global.config['precision']
        }

        postevent_call = templates['call'] % {
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'target': proj.target[0] if isinstance(proj.target, list) else proj.target,
            'conn_args': conn_call % ids,
            'add_args': add_args_call
        }

        return postevent_body, postevent_header, postevent_call

    def _init_random_distributions(self, proj):
        # Random numbers
        code = ""
        if len(proj.synapse_type.description['random_distributions']) > 0:
            code += """
        // Random numbers"""
            for dist in proj.synapse_type.description['random_distributions']:
                rng_ids = {
                    'id': proj.id,
                    'rd_name': dist['name'],
                }
                code += self._templates['rng'][dist['locality']]['init'] % rng_ids

        return code

    def _memory_transfers(self, proj):
        """
        Generate source code for transfer variables and parameters.
        """
        if 'host_device_transfer' in  proj._specific_template.keys() and \
            'device_host_transfer' in proj._specific_template.keys():
            return proj._specific_template['host_device_transfer'], proj._specific_template['device_host_transfer']

        host_device_transfer = ""
        device_host_transfer = ""

        #
        # Host -> Device
        #
        proc_attr = []
        for attr in proj.synapse_type.description['parameters']+proj.synapse_type.description['variables']:
            if attr['name'] in proc_attr:
                continue

            host_device_transfer += self._templates['host_to_device'][attr['locality']] % {
                'id': proj.id,
                'name': attr['name'],
                'type': attr['ctype']
            }

            proc_attr.append(attr['name'])

        #
        # Device -> Host
        #
        proc_attr = []
        for attr in proj.synapse_type.description['parameters']+proj.synapse_type.description['variables']:
            if attr['name'] in proc_attr:
                continue

            device_host_transfer += self._templates['device_to_host'][attr['locality']] % {
                'id': proj.id, 'name': attr['name'], 'type': attr['ctype']
            }

            proc_attr.append(attr['name'])

        return host_device_transfer, device_host_transfer

    def _process_equations(self, proj, equations, ids, locality):
        """
        Process the equation block and create equation blocks and
        corresponding kernel argument list and call statement.

        .. Note:

        This function is a helper function and should be called by
        _update_synapse() only.
        """
        # Process equations and determine dependencies
        syn_deps, neur_deps = self._select_deps(proj, locality)
        kernel_args, kernel_args_call = self._gen_kernel_args(proj, neur_deps, syn_deps)

        # Add pre_rank identifier if needed
        if len(neur_deps) > 0:
            if locality == "semiglobal":
                equations = "\t\tint rk_pre = pre_rank%(semiglobal_index)s;\n" + equations
            elif locality == "local":
                equations = "\t\tint rk_pre = pre_rank%(local_index)s;\n" + equations

        # Fill code template with ids
        equations = equations % ids

        # Gather pre-loop declaration (dt/tau for ODEs)
        pre_loop = ""
        for var in proj.synapse_type.description['variables']:
            if var['locality'] == locality:
                if 'pre_loop' in var.keys() and len(var['pre_loop']) > 0:
                    pre_loop += Global.config['precision'] + ' ' + var['pre_loop']['name'] + ' = ' + var['pre_loop']['value'] + ';\n'
            else:
                continue
        if pre_loop.strip() != '':
            pre_loop = """
    // Updating the step sizes
""" + pre_loop % ids

        return equations, pre_loop, kernel_args, kernel_args_call

    def _update_synapse(self, proj):
        """
        Generate the device codes for synaptic equations. As the parallel
        evaluation of local and global equations within one kernel would require
        a __syncthread() call, we split up the implementation into two seperate
        parts.

        Return:

        * a tuple contain three strings ( body, call, header )
        """
        # Global variables
        global_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'global', 'proj', padding=1, wrap_w="plasticity")

        # Semiglobal variables
        semiglobal_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'semiglobal', 'proj', padding=2, wrap_w="plasticity")

        # Local variables
        local_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'local', 'proj', padding=2, wrap_w="plasticity")

        # Something to do?
        if global_eq.strip() == '' and semiglobal_eq.strip() == '' and local_eq.strip() == '':
            return "", "", ""

        # Dictionary of pre/suffixes
        ids = {
            'id_proj' : proj.id,
            'target': proj.target,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,
            'local_index': '[j]',
            'semiglobal_index': '[rk_post]',
            'global_index': '[0]',
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]',
            'pre_prefix': 'pre_',
            'post_prefix': 'post_',
            'delay_nu' : '[delay[i][j]-1]', # non-uniform delay
            'delay_u' : '[' + str(proj.uniform_delay-1) + ']', # uniform delay
        }

        body = ""
        header = ""
        local_call = ""
        global_call = ""
        semiglobal_call = ""

        #
        # Fill code templates for global, semiglobal and local equations
        #
        if global_eq.strip() != '':
            global_eq, global_pre_code, kernel_args_global, kernel_args_call_global = self._process_equations( proj, global_eq, ids, 'global' )

        if semiglobal_eq.strip() != '':
            semiglobal_eq, semiglobal_pre_code, kernel_args_semiglobal, kernel_args_call_semiglobal =  self._process_equations( proj, semiglobal_eq, ids, 'semiglobal' )

        if local_eq.strip() != '':
            local_eq, local_pre_code, kernel_args_local, kernel_args_call_local =  self._process_equations( proj, local_eq, ids, 'local' )

        #
        # replace local function calls
        if len(proj.synapse_type.description['functions']) > 0:
            global_eq, semiglobal_eq, local_eq = self._replace_local_funcs(proj, global_eq, semiglobal_eq, local_eq)

        # replace the random distributions
        local_eq, global_eq = self._replace_random(local_eq, global_eq, proj.synapse_type.description['random_distributions'])

        if global_eq.strip() != '':
            body += self._templates['synapse_update']['global']['body'] % {
                'id': proj.id,
                'kernel_args': kernel_args_global,
                'global_eqs': global_eq,
                'pre': proj.pre.id,
                'post': proj.post.id,
                'pre_loop':  global_pre_code,
                'float_prec': Global.config['precision']
            }

            header += self._templates['synapse_update']['global']['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args_global,
                'float_prec': Global.config['precision']
            }

            global_call = self._templates['synapse_update']['global']['call'] % {
                'id_proj': proj.id,
                'post': proj.post.id,
                'pre': proj.pre.id,
                'target': proj.target[0] if isinstance(proj.target, list) else proj.target,
                'kernel_args_call': kernel_args_call_global,
                'float_prec': Global.config['precision']
            }

        if semiglobal_eq.strip() != '':
            body += self._templates['synapse_update']['semiglobal']['body'] % {
                'id': proj.id,
                'kernel_args': kernel_args_semiglobal,
                'semiglobal_eqs': semiglobal_eq,
                'pre': proj.pre.id,
                'post': proj.post.id,
                'pre_loop': semiglobal_pre_code,
                'float_prec': Global.config['precision']
            }

            header += self._templates['synapse_update']['semiglobal']['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args_semiglobal,
                'float_prec': Global.config['precision']
            }

            semiglobal_call = self._templates['synapse_update']['semiglobal']['call'] % {
                'id_proj': proj.id,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'target': proj.target[0] if isinstance(proj.target, list) else proj.target,
                'kernel_args_call': kernel_args_call_semiglobal,
                'float_prec': Global.config['precision']
            }

        if local_eq.strip() != '':
            body += self._templates['synapse_update']['local']['body'] % {
                'id': proj.id,
                'kernel_args': kernel_args_local,
                'local_eqs': local_eq,
                'pre': proj.pre.id,
                'post': proj.post.id,
                'pre_loop': local_pre_code,
                'float_prec': Global.config['precision']
            }

            header += self._templates['synapse_update']['local']['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args_local,
                'float_prec': Global.config['precision']
            }

            local_call = self._templates['synapse_update']['local']['call'] % {
                'id_proj': proj.id,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'target': proj.target[0] if isinstance(proj.target, list) else proj.target,
                'kernel_args_call': kernel_args_call_local,
                'float_prec': Global.config['precision']
            }

        call = self._templates['synapse_update']['call'] % {
            'id_proj': proj.id,
            'post': proj.post.id,
            'pre': proj.pre.id,
            'target': proj.target,
            'global_call': global_call,
            'semiglobal_call': semiglobal_call,
            'local_call': local_call,
            'float_prec': Global.config['precision']
        }

        # Profiling
        if self._prof_gen:
            call = self._prof_gen.annotate_update_synapse(proj, call)

        return body, header, call
