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
from .ProjectionGenerator import ProjectionGenerator, get_bounds
from .CUDATemplates import cuda_templates
from .Connectivity import CUDAConnectivity

from ANNarchy.core import Global
from ANNarchy.core.Population import Population
from ANNarchy.generator.Utils import generate_equation_code, tabify

import re

class CUDAGenerator(ProjectionGenerator, CUDAConnectivity):
    """
    Generate the header for a Population object to run either on a Nvidia
    GPU using Nvidia SDK > 5.0 and CC > 2.0
    """
    _templates = cuda_templates

    def __init__(self, profile_generator, net_id):
        # The super here calls all the base classes, so first ProjectionGenerator
        # and afterwards CUDAConnectivity
        super(CUDAGenerator, self).__init__(profile_generator, net_id)

    def header_struct(self, proj, annarchy_dir):
        """
        """
        # configure Connectivity base class
        self.configure(proj)
        
        # Generate declarations and accessors for the variables
        decl, accessor = self._declaration_accessors(proj)

        # concurrent streams
        decl['cuda_stream'] = cuda_templates['cuda_stream']

        # Initiliaze the projection
        init_parameters_variables = self._init_parameters_variables(proj)

        update_variables_body, update_variables_header, update_variables_call = self._update_synapse(proj)

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
            'declare_delay': decl['delay'] if has_delay else "",
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

        proj_desc['psp_header'] = psp_header
        proj_desc['psp_body'] = psp_body
        proj_desc['psp_call'] = psp_call

        proj_desc['update_synapse_header'] = update_variables_header
        proj_desc['update_synapse_body'] = update_variables_body
        proj_desc['update_synapse_call'] = update_variables_call

        proj_desc['postevent_header'] = post_event_header
        proj_desc['postevent_body'] = post_event_body
        proj_desc['postevent_call'] = post_event_call

        proj_desc['host_to_device'] = tabify("proj%(id)s.host_to_device();" % {'id':proj.id}, 1)+"\n"
        proj_desc['device_to_host'] = tabify("proj%(id)s.device_to_host();" % {'id':proj.id}, 1)+"\n"

        return proj_desc

    def _declaration_accessors(self, proj):
        declaration, accessor = ProjectionGenerator._declaration_accessors(self, proj)

        declaration['cuda_stream'] = cuda_templates['cuda_stream']
        return declaration, accessor

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
                'global_index': '[i]',
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
        conn_header = "int* rank_pre, int *row_ptr, double *pre_r, double* w"
        conn_call = "proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, pop%(id_pre)s.gpu_r, proj%(id_proj)s.gpu_w " % {'id_proj': proj.id, 'id_pre': proj.pre.id}

        #
        # finish the kernel etc.
        psp = proj.synapse_type.description['psp']['cpp'] % ids

        body_code = self._templates['computesum_rate']['body'] % {
            'id_proj': proj.id,
            'conn_args': conn_header,
            'target_arg': "sum_"+proj.target,
            'add_args': add_args_header,
            'psp': psp
        }

        header_code = self._templates['computesum_rate']['header'] % {
            'id': proj.id,
            'conn_args': conn_header,
            'target_arg': "sum_"+proj.target,
            'add_args': add_args_header
        }

        call_code = self._templates['computesum_rate']['call'] % {
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

        return header_code, body_code, call_code

    def _computesum_spiking(self, proj):
        """
        Generate code for the spike propagation.
        """
        if 'header' in proj._specific_template.keys() and \
           'body' in proj._specific_template.keys() and \
           'call' in proj._specific_template.keys():
            try:
                header = proj._specific_template['header']
                body = proj._specific_template['body']
                call = proj._specific_template['call']
            except KeyError:
                Global._error('header, body and call should be overwritten')
            return header, body, call

        updated_variables_list = []
        deps = [] # list of dependencies for this kernel

        kernel_args = ""
        kernel_args_call = ""
        eq_code = ""

        # Basic tags
        if proj._storage_format == "lil":
            ids = {
                'id_proj' : proj.id,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'target': proj.target,
                'local_index': "[indices[syn_idx]]",
                'global_index': '[post_ranks[syn_idx]]'
            }
        elif proj._storage_format == "csr":
            ids = {
                'id_proj' : proj.id,
                'id_post': proj.post.id,
                'id_pre': proj.pre.id,
                'target': proj.target,
                'local_index': "[syn_idx]",
                'global_index': '[post_ranks[syn_idx]]'
            }
        else:
            raise NotImplementedError

        for eq in proj.synapse_type.description['pre_spike']:

            if eq['name'] == "g_target":   # synaptic transmission
                code = eq['cpp'].split('=')[1] % ids
                # HD (04.08.2016):
                # The temporary variable (_tmp_) is not absolutely essential,
                # but it might be better, as the psp term can be complex.
                eq_code += """
            double _tmp_ = %(psp)s
            atomicAdd(&g_target[post_ranks[syn_idx]], _tmp_);""" % {'psp': code }

                # Determine bounds
                for key, val in eq['bounds'].items():
                    if not key in ['min', 'max']:
                        continue
                    try:
                        value = str(float(val))
                    except: # TODO: more complex operations
                        value = val % ids
                        items = re.findall("[A-Za-z_]+\%\(", val)
                        for item in items:
                            deps.append(item.replace("%(",""))

                    eq_code += """
        if ( g_target[post_ranks[syn_idx]] %(op)s %(val)s )
             g_target[post_ranks[syn_idx]] = %(val)s;
""" % {'id_post': proj.post.id, 'op': "<" if key == 'min' else '>', 'val': value}

                kernel_args_call += ", pop%(id_post)s.gpu_g_%(target)s" % {'id_post': proj.post.id, 'target': proj.target}
                kernel_args += ", " + eq['ctype'] + "* " + eq['name']
            else:
                condition = ""
                # Check conditions to update the variable
                if eq['name'] == 'w': # Surround it by the learning flag
                    condition = "plasticity"
                if 'unless_post' in eq['flags']: # Flags avoids pre-spike evaluation when post fires at the same time
                    simultaneous = "pop%(id_pre)s.last_spike[pre_rank[i][j]] != pop%(id_post)s.last_spike[post_rank[i]]" % {'id_post': proj.post.id, 'id_pre': proj.pre.id}
                    if condition == "":
                        condition = simultaneous
                    else:
                        condition += "&&(" + simultaneous + ")"
                # Generate the code
                if condition != "":
                    updated_variables_list += """
// unless_post can prevent evaluation of presynaptic variables
if(%(condition)s){
    // %(eq)s
    %(cpp)s
    %(bounds)s
}
""" % {'eq': eq['eq'], 'cpp': eq['cpp'] % ids, 'bounds': get_bounds(eq) % ids, 'condition': condition}

                else: # Normal synaptic variable
                    updated_variables_list += """
// %(eq)s
%(cpp)s
%(bounds)s""" % {'eq': eq['eq'], 'cpp': eq['cpp'] % ids, 'bounds': get_bounds(eq) % ids}

        #######################################################
        # Event-driven integration of synaptic variables
        #######################################################
        has_exact = False
        event_driven_code = ''
        for var in proj.synapse_type.description['variables']:

            if var['method'] == 'event-driven':
                has_exact = True
                event_driven_code += """
        // %(eq)s
        %(exact)s
""" % {'eq': var['eq'], 'exact': var['cpp'].replace('(t)', '(t-1)') %{'id_proj' : proj.id, 'local_index': "[indices[syn_idx]]", 'global_index': '[post_ranks[syn_idx]]'}}

                # add to kernel dependencies
                for dep in var['dependencies']:
                    deps.append(dep)

        if has_exact:
            event_driven_code += """
        // Update the last event for the synapse
        _last_event%(local_idx)s = t;
""" % {'local_idx': '[indices[syn_idx]]'}

            # event-driven requires last event
            kernel_args += ", long* _last_event"
            kernel_args_call += ", proj%(id_proj)s._gpu_last_event"  % {'id_proj': proj.id}

        # Generate code for pre-spike variables
        pre_code = ""
        if len(updated_variables_list) > 0:
            for var in updated_variables_list:
                pre_code += var
            pre_code = tabify(pre_code, 3)

        for pre_deps in proj.synapse_type.description['pre_spike']+proj.synapse_type.description['post_spike']:
            # right side
            deps.append(pre_deps['name'])
            # left side
            for var in pre_deps['dependencies']:
                deps.append(var)
        deps = list(set(deps))

        # remove pre defined variables
        if 'w' in deps:
            deps.remove('w')
        if 'g_target' in deps:
            deps.remove('g_target')

        for var in deps:
            attr = self._get_attr(proj, var)

            kernel_args += ", "+ attr['ctype']+ "* " + attr['name']
            kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % {'id_proj': proj.id, 'name': attr['name']}

        if proj._storage_format == "lil":
            conn_call = "proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_w" % {'id_proj': proj.id}
            conn_body = "int* row_ptr, int* pre_ranks, double* w"
            conn_header = "int* row_ptr, int* pre_ranks, double *w"
            prefix = ""
            row_desc = ""
        elif proj._storage_format == "csr":
            conn_call = "proj%(id_proj)s._gpu_row_ptr, proj%(id_proj)s._gpu_col_idx, proj%(id_proj)s._gpu_inv_idx, proj%(id_proj)s.gpu_w" % {'id_proj': proj.id}
            conn_body = "int* row_ptr, int* post_ranks, int* indices, double* w"
            conn_header = "int* row_ptr, int* post_ranks, int* indices, double *w"
            prefix = "syn_idx = row_ptr[pre_idx]+threadIdx.x;"
            row_desc = "syn_idx < row_ptr[pre_idx+1]"
        else:
            raise NotImplementedError

        call = self._templates['computesum_spiking']['call'] % {
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'target': proj.target,
            'kernel_args': kernel_args_call,
            'conn_args': conn_call,
            'stream_id': proj.id
        }

        pre_size = proj.pre.size if isinstance(proj.pre, Population) else proj.pre.population.size
        post_size = proj.post.size if isinstance(proj.post, Population) else proj.post.population.size
        body = self._templates['computesum_spiking']['body'] % {
            'id': proj.id,
            'conn_arg': conn_body,
            'prefix': prefix,
            'row_desc': row_desc,
            'kernel_args': kernel_args,
            'event_driven': event_driven_code,
            'psp':  eq_code,
            'pre_event': pre_code,
            'pre_size': pre_size,
            'post_size': post_size,
        }

        header = self._templates['computesum_spiking']['header'] % {
            'id': proj.id,
            'conn_header': conn_header,
            'kernel_args': kernel_args
        }

        ####################################################
        # Not even-driven summation of psp: like rate-coded
        ####################################################
        if 'psp' in  proj.synapse_type.description.keys(): # not event-based
            # transfrom psp equation
            psp = proj.synapse_type.description['psp']['cpp']

            add_args_header = ""
            add_args_call = ""
            #
            # on which other variables the psp rely?
            tmp_deps = re.findall(r"post_prefix\)s[A-Za-z]*\%\(", psp)
            for post_dep in tmp_deps:
                var = post_dep.split(")s")[1].split("%(")[0]

                add_args_header += ", double* post_%(var)s" % {'var': var}
                add_args_call += ", pop%(id)s.gpu_%(var)s" % {'id': proj.post.id, 'var': var}

            tmp_deps = re.findall(r"pre_prefix\)s[A-Za-z]*\%\(", psp)
            for pre_dep in tmp_deps:
                var = pre_dep.split(")s")[1].split("%(")[0]

                add_args_header += ", double* pre_%(var)s" % {'var': var}
                add_args_call += ", pop%(id)s.gpu_%(var)s" % {'id': proj.post.id, 'var': var}

            psp = proj.synapse_type.description['psp']['cpp'] % {
                'local_index': '[rank_pre[j]]',
                'pre_prefix': 'pre_',
                'pre_index': '[rank_pre[j]]',
                'post_prefix': 'post_',
                'post_index': '[blockIdx.x]'
            }

            # connectivity
            conn_body = "int *rank_pre, int* row_ptr, double* w"
            conn_header = "int *rank_pre, int* row_ptr, double *w"
            conn_call = "proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_w" % {'id_proj': proj.id}

            # build up kernel
            body = self._templates['computesum_rate']['body'] % {
                'id_proj': proj.id,
                'conn_args': conn_body,
                'target_arg': 'g_'+proj.target,
                'add_args':  add_args_header,
                'psp': psp,
            }

            header = self._templates['computesum_rate']['header'] % {
                'id': proj.id,
                'conn_args': conn_header,
                'add_args': add_args_header,
                'target_arg': 'g_'+proj.target,
            }

            call = self._templates['computesum_rate']['call'] % {
                'id_proj': proj.id,
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'target_arg': ', pop%(id_post)s.gpu_g_%(target)s' % {'id_post': proj.post.id, 'target': proj.target},
                'target': proj.target,
                'conn_args': conn_call,
                'add_args': add_args_call
            }

        # Profiling
        if self._prof_gen:
            call = self._prof_gen.annotate_computesum_spiking(proj, call)

        return header, body, call

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
                'global_index': '[i]',
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
                'global_index': '[i]',
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
        for eq in proj.synapse_type.description['post_spike']:
            post_code += '// ' + eq['eq'] + '\n'
            if eq['name'] == 'w':
                post_code += "if(plasticity)\n"
            post_code += eq['cpp'] % ids + '\n'
            post_code += get_bounds(eq) % ids + '\n'

            # add dependencies, only right side!
            for deps in eq['dependencies']:
                post_deps.append(deps)
            # left side of equations is not part of dependencies
            post_deps.append(eq['name'])
        post_code = tabify(post_code, 2)

        # Create add_args for event-driven eqs and post_event
        kernel_deps = list(set(post_deps+event_deps)) # variables can occur in several eqs
        for dep in kernel_deps:
            if dep == "w":
                continue

            attr = self._get_attr(proj, dep)
            add_args_header += ', %(type)s* %(name)s' % {'type': attr['ctype'], 'name': attr['name']}
            add_args_call +=  ', proj%(id)s.gpu_%(name)s' % {'id': proj.id, 'name': attr['name']}

        if proj._storage_format == "lil":
            conn_header = "int* row_ptr, int* pre_ranks,"
            conn_call = ", proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank"
        elif proj._storage_format == "csr":
            conn_header = "int* row_ptr, int *col_idx, "
            conn_call = ", proj%(id_proj)s._gpu_row_ptr,  proj%(id_proj)s._gpu_col_idx"
        else:
            raise NotImplementedError

        postevent_header = self._templates['post_event']['header'] % {
            'id_proj': proj.id,
            'conn_args': conn_header,
            'add_args': add_args_header
        }

        postevent_body = self._templates['post_event']['body'] % {
            'id_proj': proj.id,
            'conn_args': conn_header,
            'add_args': add_args_header,
            'event_driven': event_driven_code,
            'post_code': post_code
        }

        postevent_call = self._templates['post_event']['call'] % {
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'target': proj.target,
            'conn_args': conn_call % ids,
            'add_args': add_args_call
        }

        return postevent_body, postevent_header, postevent_call

    def _init_random_distributions(self, proj):
        if proj.synapse_type.description['random_distributions'] != []:
            raise NotImplementedError

        return ""

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
        # Global variables
        global_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'global', 'proj', padding=2, wrap_w="plasticity")

        # Local variables
        local_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'local', 'proj', padding=3, wrap_w="plasticity")

        if global_eq.strip() == '' and local_eq.strip() == '':
            return "", "", ""

        if proj.synapse_type.type == "rate":
            default_args = "int* pre_rank, int *row_ptr, double dt"
            default_args_call = "proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, dt" % {'id_proj': proj.id}
        else:
            default_args = "int *pre_rank, int *row_ptr, double dt"
            default_args_call = "proj%(id_proj)s.gpu_pre_rank, proj%(id_proj)s.gpu_row_ptr, dt" % {'id_proj': proj.id}

        deps = []
        # dependencies consists of several componentes:
        # - acces to pre- or post-population
        # - variables / parameters of the projection
        # - pre- or post-spike event
        pop_deps = list(set(proj.synapse_type.description['dependencies']['pre'] +
                            proj.synapse_type.description['dependencies']['post']))

        for attr in ( proj.synapse_type.description['variables'] + proj.synapse_type.description['parameters'] ):
            deps.append(attr['name'])

        for dep in pop_deps:
            deps.append(dep)

        deps = list(set(deps))

        kernel_args = ""
        kernel_args_call = ""
        for dep in deps:
            if dep in pop_deps:
                # Attention: a variable can occur in pre and post,
                # consequently the two independent if cases
                if proj.pre.id != proj.post.id:
                    if dep in proj.synapse_type.description['dependencies']['pre']:
                        # TODO: type dependency !!!!!!!!!!!!!!!!
                        kernel_args += ", double* pop%(id)s_%(name)s" % {'id': proj.pre.id, 'name': dep}
                        kernel_args_call += ", pop%(id)s.gpu_%(name)s" % {'id': proj.pre.id, 'name': dep}

                    if dep in proj.synapse_type.description['dependencies']['post']:
                        # TODO: type dependency !!!!!!!!!!!!!!!!
                        kernel_args += ", double* pop%(id)s_%(name)s" % {'id': proj.post.id, 'name': dep}
                        kernel_args_call += ", pop%(id)s.gpu_%(name)s" % {'id': proj.post.id, 'name': dep}
                else:
                    if dep in proj.synapse_type.description['dependencies']['pre']:
                        # TODO: type dependency !!!!!!!!!!!!!!!!
                        kernel_args += ", double* pop%(id)s_%(name)s" % {'id': proj.pre.id, 'name': dep}
                        kernel_args_call += ", pop%(id)s.gpu_%(name)s" % {'id': proj.pre.id, 'name': dep}

                    else:
                        # TODO: type dependency !!!!!!!!!!!!!!!!
                        kernel_args += ", double* pop%(id)s_%(name)s" % {'id': proj.post.id, 'name': dep}
                        kernel_args_call += ", pop%(id)s.gpu_%(name)s" % {'id': proj.post.id, 'name': dep}

            else:
                attr = self._get_attr(proj, dep)

                kernel_args += ", %(type)s* %(name)s" % {'type': attr['ctype'], 'name': attr['name']}
                kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % {'id_proj': proj.id, 'type': attr['ctype'], 'name': attr['name']}

        # Dictionary of pre/suffixes
        ids = {
            'id_proj' : proj.id,
            'target': proj.target,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,
            'local_index': '[j]',
            'global_index': '[rk_post]',
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]',
            'pre_prefix': 'pop'+ str(proj.pre.id) + '_',
            'post_prefix': 'pop'+ str(proj.post.id) + '_',
            'delay_nu' : '[delay[i][j]-1]', # non-uniform delay
            'delay_u' : '[' + str(proj.uniform_delay-1) + ']' # uniform delay
        }

        body = self._templates['synapse_update']['body'] % {
            'id': proj.id,
            'default_args': default_args,
            'kernel_args': kernel_args,
            'global_eqs': global_eq % ids,
            'local_eqs': local_eq % ids,
            'target': proj.target,
            'pre': proj.pre.id,
            'post': proj.post.id,
        }

        header = self._templates['synapse_update']['header'] % {
            'id': proj.id,
            'default_args': default_args,
            'kernel_args': kernel_args
        }

                # generate code
        call = self._templates['synapse_update']['call'] % {
            'id_proj': proj.id,
            'post': proj.post.id,
            'pre': proj.pre.id,
            'target': proj.target,
            'default_args_call': default_args_call,
            'kernel_args_call': kernel_args_call
        }

        return body, header, call
