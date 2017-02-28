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

        # Detect non.uniform delays to eventually generate the code
        has_delay = (proj.max_delay > 1 and proj.uniform_delay == -1)

        # Connectivity matrix
        connectivity_matrix = self._connectivity(proj)

        # Memory transfers
        host_device_transfer, device_host_transfer = self._memory_transfers(proj)

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

        if proj._storage_format == "lil":
            cuda_flattening = self._templates['flattening'] % {
                'id_post':proj.post.id
            }
        else:
            cuda_flattening = ""

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
            'declare_delay': decl['delspike_countay'] if has_delay else "",
            'declare_event_driven': decl['event_driven'] if has_event_driven else "",
            'declare_rng': decl['rng'],
            'declare_parameters_variables': decl['parameters_variables'],
            'declare_cuda_stream': decl['cuda_stream'],
            'declare_additional': decl['additional'],
            'declare_profile': declare_profile,
            'init_connectivity_matrix': connectivity_matrix['init'],
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
            'cuda_flattening': cuda_flattening
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

    def _computesum_rate(self, proj):
        """
        returns all data needed for compute postsynaptic sum kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
        # Default variables needed in psp_code
        psp_prefix = """
        int nb_post; double sum;"""
        if 'psp_prefix' in proj._specific_template.keys():
            psp_prefix = proj._specific_template['psp_prefix']

        # Specific projection
        if 'psp_code' in proj._specific_template.keys():
            return psp_prefix, proj._specific_template['psp_code']

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

            # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #
            # Some of the features currently not work
            # psp = w*pre.r + X
            #
            # ATTENTION: an idea is implemented in computesum_spiking_cuda

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

        body_code = self._templates['rate_psp']['body'] % {
            'float_prec': Global.config['precision'],
            'id_proj': proj.id,
            'conn_args': conn_header,
            'target_arg': "sum_"+proj.target,
            'add_args': add_args_header,
            'psp': psp
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
            'add_args': add_args_call
        }

        # Take delays into account if any
        if proj.max_delay > 1:
            if proj.uniform_delay == -1:
                Global._error("Only uniform delays are supported on GPUs.")

            else:
                # TODO: replace by regex
                call_code = call_code.replace("gpu_r,", "gpu_delayed_r["+str(proj.max_delay-1)+"],")

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
                psp_dict = {
                    'psp': var['cpp'].split('=')[1] % ids,
                    'float_prec': Global.config['precision']
                }
                psp_code += "%(float_prec)s tmp = %(psp)s\natomicAdd(&g_target[post_rank], tmp);" % psp_dict

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

            attr = self._get_attr(proj, dep)
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

                attr = self._get_attr(proj, dep)
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

            # connectivity
            conn_body = "int* row_ptr, int *col_idx, double* w"
            conn_header = "int* row_ptr, int *col_idx, double *w"
            conn_call = "proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_w" % ids

            call = ""
            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                call += template['call'] % {
                    'id_proj': proj.id,
                    'id_pre': proj.pre.id,
                    'id_post': proj.post.id,
                    'target_arg': ', pop%(id_post)s.gpu_g_%(target)s' % {'id_post': proj.post.id, 'target': target},
                    'target': target,
                    'conn_args': conn_call,
                    'kernel_args': kernel_args_call,
                    'float_prec': Global.config['precision']
                }
            body = template['body'] % {
                'id_proj': proj.id,
                'conn_args': conn_body,
                'target_arg': proj.target,
                'kernel_args':  kernel_args,
                'psp': psp_code,
                'pre_code': tabify(pre_spike_code, 1),
                'float_prec': Global.config['precision']
            }
            header = template['header'] % {
                'id': proj.id,
                'conn_args': conn_header,
                'kernel_args': kernel_args,
                'target_arg': 'g_'+proj.target,
                'float_prec': Global.config['precision']
            }

        else: # event-based
            # select the correct template
            template = self._templates['spike_transmission']['event_driven'][proj._storage_order]

            # Connectivity description
            conn_call = "proj%(id_proj)s.gpu_col_ptr, proj%(id_proj)s.gpu_row_idx, proj%(id_proj)s.gpu_inv_idx, proj%(id_proj)s.gpu_w" % ids
            conn_body = "int* col_ptr, int* row_idx, int* inv_idx, %(float_prec)s* w, %(float_prec)s* g_target" % ids
            conn_header = "int* col_ptr, int* row_idx, int* inv_idx, %(float_prec)s *w, %(float_prec)s* g_target" % ids

            # Population sizes
            pre_size = proj.pre.size if isinstance(proj.pre, Population) else proj.pre.population.size
            post_size = proj.post.size if isinstance(proj.post, Population) else proj.post.population.size

            call = ""
            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                call += template['call'] % {
                    'id_proj': proj.id,
                    'id_pre': proj.pre.id,
                    'id_post': proj.post.id,
                    'target': target,
                    'kernel_args': kernel_args_call  % {'id_post': proj.post.id, 'target': target},
                    'conn_args': conn_call + ", pop%(id_post)s.gpu_g_%(target)s" % {'id_post': proj.post.id, 'target': target}
                }
            body = template['body'] % {
                'id': proj.id,
                'float_prec': Global.config['precision'],
                'conn_arg': conn_body,
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
                'conn_header': conn_header,
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

        * pop_deps     list of dependencies part of populations
        * deps         list of all dependencies
        """
        deps = []

        # access pre- or postsynaptic neurons
        pop_deps = list(set(proj.synapse_type.description['dependencies']['pre'] +
                            proj.synapse_type.description['dependencies']['post']))
        for dep in pop_deps:
            deps.append(dep)

        for var in proj.synapse_type.description['variables']:
            if var['eq'] == '':
                continue # nothing to do here

            if var['locality'] == locality:
                deps.append(var['name'])
                for dep in var['dependencies']:
                    deps.append(dep)

        deps = list(set(deps))
        return pop_deps, deps

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
                    attr = PopulationGenerator._get_attr(proj.pre, dep)
                    ids = {'type': attr['ctype'], 'id': proj.pre.id, 'name': dep}
                    kernel_args += ", %(type)s* pre_%(name)s" % ids
                    kernel_args_call += ", pop%(id)s.gpu_%(name)s" % ids
                if dep in proj.synapse_type.description['dependencies']['post']:
                    attr = PopulationGenerator._get_attr(proj.post, dep)
                    ids = {'type': attr['ctype'], 'id': proj.post.id, 'name': dep}
                    kernel_args += ", %(type)s* post_%(name)s" % ids
                    kernel_args_call += ", pop%(id)s.gpu_%(name)s" % ids

            else:
                attr_type, attr_dict = ProjectionGenerator._get_attr_and_type(proj, dep)

                ids = {
                    'id_proj': proj.id,
                    'type': 'curandState'  if attr_type == "rand" else attr_dict['ctype'],
                    'name': attr_dict['name']
                }
                kernel_args += ", %(type)s* %(name)s" % ids
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
            device_code += cpp_func.replace('double '+func['name'], '__device__ double proj%(id)s_%(func)s'%{'id': proj.id, 'func':func['name']})

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

        for dist in random_distributions:
            if dist['dist'] == "Uniform":
                dist_ids = {
                    'postfix': prec_extension,
                    'rd': dist['name'], 'min': dist['args'].split(',')[0],
                    'max': dist['args'].split(',')[1]
                }
                term = """( curand_uniform%(postfix)s( &%(rd)s[j] ) * (%(max)s - %(min)s) + %(min)s )""" % dist_ids
                loc_eqs = loc_eqs.replace(dist['name']+"[j]", term)

                term = """( curand_uniform%(postfix)s( &%(rd)s[0] ) * (%(max)s - %(min)s) + %(min)s )""" % dist_ids
                glob_eqs = glob_eqs.replace(dist['name']+"[0]", term)
            elif dist['dist'] == "Normal":
                dist_ids = {
                    'postfix': prec_extension, 'rd': dist['name'],
                    'mean': dist['args'].split(",")[0],
                    'sigma': dist['args'].split(",")[1]
                }
                term = """( curand_normal%(postfix)s( &%(rd)s[j] ) * %(sigma)s + %(mean)s )""" % dist_ids
                loc_eqs = loc_eqs.replace(dist['name']+"[j]", term)

                term = """( curand_normal%(postfix)s( &%(rd)s[0] ) * %(sigma)s + %(mean)s )""" % dist_ids
                glob_eqs = glob_eqs.replace(dist['name']+"[0]", term)
            elif dist['dist'] == "LogNormal":
                dist_ids = {
                    'postfix': prec_extension, 'rd': dist['name'],
                    'mean': dist['args'].split(',')[0],
                    'std_dev': dist['args'].split(',')[1]
                }
                term = """( curand_log_normal%(postfix)s( &%(rd)s[j], %(mean)s, %(std_dev)s) )""" % dist_ids
                loc_eqs = loc_eqs.replace(dist['name']+"[j]", term)

                term = """( curand_log_normal%(postfix)s( &%(rd)s[0], %(mean)s, %(std_dev)s) )""" % dist_ids
                glob_eqs = glob_eqs.replace(dist['name']+"[0]", term)
            else:
                Global._error("Unsupported random distribution on GPUs: " + dist['dist'])

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
                'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
                'post_prefix': 'pop'+ str(proj.post.id) + '.'
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

            attr = self._get_attr(proj, dep)
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

        postevent_call = ""
        target_list = proj.target if isinstance(proj.target, list) else [proj.target]
        for target in target_list:
            postevent_call += templates['call'] % {
                'id_proj': proj.id,
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'target': target,
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
                # in principal only important for openmp
                rng_def = {
                    'id': proj.id,
                    'float_prec': Global.config['precision']
                }
                # RNG declaration, only for openmp
                rng_ids = {
                    'id': proj.id,
                    'rd_name': dist['name'],
                    'type': dist['ctype'],
                    'rd_init': dist['definition'] % rng_def
                }
                code += self._templates['rng'][dist['locality']]['init'] % rng_ids

        return code

    def _memory_transfers(self, proj):
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
        global_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'global', 'proj', padding=2, wrap_w="plasticity")

        # Semiglobal variables
        semiglobal_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'semiglobal', 'proj', padding=2, wrap_w="plasticity")

        # Local variables
        local_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'local', 'proj', padding=2, wrap_w="plasticity")

        # Something to do?
        if global_eq.strip() == '' and local_eq.strip() == '':
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
            'delay_u' : '[' + str(proj.uniform_delay-1) + ']' # uniform delay
        }

        body = ""
        header = ""
        local_call = ""
        global_call = ""
        semiglobal_call = ""

        # fill code templates
        global_pop_deps, global_pop = self._select_deps(proj, 'global')
        kernel_args_global, kernel_args_call_global = self._gen_kernel_args(proj, global_pop_deps, global_pop)
        global_eq = global_eq % ids

        semiglobal_pop_deps, semiglobal_pop = self._select_deps(proj, 'semiglobal')
        kernel_args_semiglobal, kernel_args_call_semiglobal = self._gen_kernel_args(proj, semiglobal_pop_deps, semiglobal_pop)
        semiglobal_eq = semiglobal_eq % ids

        local_pop_deps, local_pop = self._select_deps(proj, 'local')
        kernel_args_local, kernel_args_call_local = self._gen_kernel_args(proj, local_pop_deps, local_pop)
        local_eq = local_eq % ids

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
                'target': proj.target,
                'pre': proj.pre.id,
                'post': proj.post.id,
            }

            header += self._templates['synapse_update']['global']['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args_global,
            }

            global_call = ""
            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                global_call += self._templates['synapse_update']['global']['call'] % {
                    'id_proj': proj.id,
                    'post': proj.post.id,
                    'pre': proj.pre.id,
                    'target': target,
                    'kernel_args_call': kernel_args_call_global
                }

        if semiglobal_eq.strip() != '':
            body += self._templates['synapse_update']['semiglobal']['body'] % {
                'id': proj.id,
                'kernel_args': kernel_args_semiglobal,
                'semiglobal_eqs': semiglobal_eq,
                'target': proj.target,
                'pre': proj.pre.id,
                'post': proj.post.id,
            }

            header += self._templates['synapse_update']['semiglobal']['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args_semiglobal
            }

            semiglobal_call = ""
            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                semiglobal_call += self._templates['synapse_update']['semiglobal']['call'] % {
                    'id_proj': proj.id,
                    'id_post': proj.post.id,
                    'id_pre': proj.pre.id,
                    'target': target,
                    'kernel_args_call': kernel_args_call_semiglobal
                }

        if local_eq.strip() != '':
            body += self._templates['synapse_update']['local']['body'] % {
                'id': proj.id,
                'kernel_args': kernel_args_local,
                'local_eqs': local_eq,
                'target': proj.target,
                'pre': proj.pre.id,
                'post': proj.post.id,
            }

            header += self._templates['synapse_update']['local']['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args_local
            }

            local_call = ""
            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                local_call += self._templates['synapse_update']['local']['call'] % {
                    'id_proj': proj.id,
                    'id_post': proj.post.id,
                    'id_pre': proj.pre.id,
                    'target': target,
                    'kernel_args_call': kernel_args_call_local
                }

        call = self._templates['synapse_update']['call'] % {
            'id_proj': proj.id,
            'post': proj.post.id,
            'pre': proj.pre.id,
            'target': proj.target,
            'global_call': global_call,
            'semiglobal_call': semiglobal_call,
            'local_call': local_call
        }

        # Profiling
        if self._prof_gen:
            call = self._prof_gen.annotate_update_synapse(proj, call)

        return body, header, call
