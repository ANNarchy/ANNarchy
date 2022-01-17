#===============================================================================
#
#     CUDAGenerator.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2016-2021  Julien Vitay <julien.vitay@gmail.com>,
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
import ANNarchy

from copy import deepcopy

from ANNarchy.core import Global
from ANNarchy.core.Population import Population
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.generator.Utils import generate_equation_code, tabify, check_and_apply_pow_fix, determine_idx_type_for_projection

from ANNarchy.generator.Population.PopulationGenerator import PopulationGenerator

from ANNarchy.generator.Projection.ProjectionGenerator import ProjectionGenerator, get_bounds
from ANNarchy.generator.Projection.CUDA import *

class CUDAGenerator(ProjectionGenerator):
    """
    As stated in module description, inherits from ProjectionGenerator
    and implements abstract functions.
    """
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
        # Initial state
        self._templates = deepcopy(BaseTemplates.cuda_templates)
        self._template_ids = {}

        # Select the C++ connectivity template
        sparse_matrix_include, sparse_matrix_format, sparse_matrix_args, single_matrix = self._select_sparse_matrix_format(proj)

        # configure Connectivity base class
        self._configure_template_ids(proj)

        # Initialize launch configuration
        init_launch_config = self._generate_launch_config(proj)

        # Generate declarations and accessors for the variables
        decl, accessor = self._declaration_accessors(proj, single_matrix)

        # concurrent streams
        decl['cuda_stream'] = BaseTemplates.cuda_stream

        # Initiliaze the projection
        init_weights, init_delays, init_parameters_variables = self._init_parameters_variables(proj, single_matrix)

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

        # Connectivity template
        if 'declare_connectivity_matrix' not in proj._specific_template.keys():
            connector_call = self._connectivity_init(proj, sparse_matrix_format, sparse_matrix_args) % {
                'sparse_format': sparse_matrix_format,
                'init_weights': init_weights,
                'init_delays': init_delays,
                'rng_idx': "[0]" if single_matrix else "",
                'add_args': "",
                'num_threads': "",
                'float_prec': Global.config["precision"],
                'idx_type': determine_idx_type_for_projection(proj)[0]
            }
            declare_connectivity_matrix = ""
            access_connectivity_matrix = ""
        else:
            sparse_matrix_format = "SpecificConnectivity"
            sparse_matrix_args = ""
            connector_call = ""
            declare_connectivity_matrix = proj._specific_template['declare_connectivity_matrix']
            access_connectivity_matrix = proj._specific_template['access_connectivity_matrix']

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

        if 'include_additional' in proj._specific_template.keys():
            include_additional = proj._specific_template['include_additional']
        if 'struct_additional' in proj._specific_template.keys():
            struct_additional = proj._specific_template['struct_additional']
        if 'init_additional' in proj._specific_template.keys():
            init_additional = proj._specific_template['init_additional']
        if 'access_additional' in proj._specific_template.keys():
            access_additional = proj._specific_template['access_additional']

        final_code = self._templates['projection_header'] % {
            # version tag
            'annarchy_version': ANNarchy.__release__,
            # fill code templates
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'id_proj': proj.id,
            'name_pre': proj.pre.name,
            'name_post': proj.post.name,
            'target': proj.target,
            'float_prec': Global.config['precision'],
            'sparse_matrix_include': sparse_matrix_include,
            'sparse_format': sparse_matrix_format,
            'sparse_format_args': sparse_matrix_args,
            'include_additional': include_additional,
            'include_profile': include_profile,
            'struct_additional': struct_additional,
            'declare_connectivity_matrix': declare_connectivity_matrix,
            'access_connectivity_matrix': access_connectivity_matrix,
            'declare_delay': decl['declare_delay'] if has_delay else "",
            'declare_event_driven': decl['event_driven'] if has_event_driven else "",
            'declare_rng': decl['rng'],
            'declare_parameters_variables': decl['parameters_variables'],
            'declare_cuda_stream': decl['cuda_stream'],
            'declare_additional': decl['additional'],
            'declare_profile': declare_profile,
            'connector_call': connector_call,
            'init_weights': init_weights,
            'init_event_driven': "",
            'init_rng': init_rng,
            'init_launch_config': init_launch_config,            
            'init_parameters_variables': init_parameters_variables,
            'init_additional': init_additional,
            'init_profile': init_profile,
            'access_parameters_variables': accessor,
            'access_additional': access_additional,
            'host_to_device': host_device_transfer,
            'device_to_host': device_host_transfer,
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

    def _configure_template_ids(self, proj):
        """
        Assign the correct template dictionary based on projection
        storage format.
        """
        if proj._storage_format == "csr":
            self._templates.update(CSR_CUDA.conn_templates)
            if proj._storage_order == "post_to_pre":
                self._template_ids.update(CSR_CUDA.conn_ids)
                self._template_ids.update({
                    # TODO: as this value is hardcoded, this will lead to a recompile if the delay is modified
                    #       -> do we want this ? we could also use [this->delay-1]
                    'delay_u' : '[' + str(proj.uniform_delay-1) + ']' # uniform delay
                })
            else:
                raise NotImplementedError

        elif proj._storage_format == "bsr":
            self._templates.update(BSR_CUDA.conn_templates)
            self._template_ids.update(BSR_CUDA.conn_ids)

        elif proj._storage_format == "coo":
            self._templates.update(COO_CUDA.conn_templates)
            if proj._storage_order == "post_to_pre":
                self._template_ids.update(COO_CUDA.conn_ids)
            else:
                raise NotImplementedError

        elif proj._storage_format == "ellr":
            self._templates.update(ELLR_CUDA.conn_templates)
            if proj._storage_order == "post_to_pre":
                self._template_ids.update(ELLR_CUDA.conn_ids)
            else:
                raise NotImplementedError

        elif proj._storage_format == "ell":
            self._templates.update(ELL_CUDA.conn_templates)
            if proj._storage_order == "post_to_pre":
                self._template_ids.update(ELL_CUDA.conn_ids)
            else:
                raise NotImplementedError

        elif proj._storage_format == "hyb":
            self._templates.update(HYB_CUDA.conn_templates)
            # Indices must be set locally for each part

        elif proj._storage_format == "dense":
            self._templates.update(Dense_CUDA.conn_templates)
            if proj._storage_order == "post_to_pre":
                self._template_ids.update(Dense_CUDA.conn_ids)
            else:
                raise NotImplementedError

        else:
            raise Global.InvalidConfiguration("   The storage_format="+str(proj._storage_format)+" is not available on CUDA devices")

    def _generate_launch_config(self, proj):
        """
        TODO: multiple targets???
        """
        code = self._templates['launch_config'] % {
            'id_proj': proj.id
        }
        return code

    def _computesum_rate(self, proj):
        """
        returns all data needed for compute postsynaptic sum kernels:

        header:  kernel prototypes
        body:    kernel implementation
        call:    kernel call
        """
        # Specific projection
        if 'psp_header' in proj._specific_template.keys() and \
            'psp_body' in proj._specific_template.keys() and \
            'psp_call' in proj._specific_template.keys():
            return proj._specific_template['psp_header'], proj._specific_template['psp_body'], proj._specific_template['psp_call']

        # Dictionary of keywords to transform the parsed equations
        ids = deepcopy(self._template_ids)

        # Some adjustments to spare single used local variables
        if proj._storage_format == "ellr":
            ids['post_index'] = "[rank_post[i]]"
            ids['pre_index'] = "[rank_pre[j*post_size+i]]"
        elif proj._storage_format == "bsr":
            ids['pre_prefix'] = "loc_pre_"
            ids['pre_index'] = "[col]"

        # Dependencies
        dependencies = list(set(proj.synapse_type.description['dependencies']['pre']))

        #
        # Retrieve the PSP
        add_args_header = ""
        add_args_call = ""
        if not 'psp' in  proj.synapse_type.description.keys(): # default
            psp = """%(preprefix)s.r%(pre_index)s * w%(local_index)s;"""

            add_args_header += "const %(float_prec)s* __restrict__ pre_r, const %(float_prec)s* __restrict__ w" % {'float_prec': Global.config['precision']}
            add_args_call = "pop%(id_pre)s.gpu_r, proj%(id_proj)s.gpu_w " % {'id_proj': proj.id, 'id_pre': proj.pre.id}

        else: # custom psp
            psp = (proj.synapse_type.description['psp']['cpp'])

            # update dependencies
            for dep in proj.synapse_type.description['psp']['dependencies']:
                _, attr = self._get_attr_and_type(proj, dep)
                attr_ids = {
                    'id_proj': proj.id,
                    'type': attr['ctype'],
                    'name': attr['name']
                }
                add_args_header += ", const %(type)s* __restrict__ %(name)s" % attr_ids
                add_args_call += ", proj%(id_proj)s.gpu_%(name)s" % attr_ids

            for dep in list(set(proj.synapse_type.description['dependencies']['pre'])):
                _, attr = PopulationGenerator._get_attr_and_type(proj.pre, dep)
                attr_ids = {
                    'id_pre': proj.pre.id,
                    'type': attr['ctype'],
                    'name': attr['name']
                }
                add_args_header += ", %(type)s* pre_%(name)s" % attr_ids
                add_args_call += ", pop%(id_pre)s.gpu_%(name)s" % attr_ids

        # Special case where w is a single value
        if proj._has_single_weight():
            psp = re.sub(
                r'([^\w]+)w%\(local_index\)s',
                r'\1w',
                ' ' + psp
            )
        
        # Allow the use of global variables in psp (issue60)
        for var in dependencies:
            if var in proj.pre.neuron_type.description['global']:
                psp = psp.replace("%(pre_prefix)s"+var+"%(pre_index)s", "%(pre_prefix)s"+var+"%(global_index)s")

        # connectivity, yet only CSR
        conn_header = ""
        conn_call = ""

        #
        # finish the kernel etc.
        operation = proj.synapse_type.operation

        if proj._storage_format != "hyb":
            idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)

            id_dict = {
                'id_proj': proj.id,
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'idx_type': idx_type,
                'size_type': size_type
            }
            # connectivity
            conn_header = self._templates['conn_header'] % id_dict
            conn_call = self._templates['conn_call'] % id_dict

            body_code = self._templates['rate_psp']['body'][operation] % {
                'float_prec': Global.config['precision'],
                'idx_type': idx_type,
                'size_type': size_type,
                'id_proj': proj.id,
                'conn_args': conn_header,
                'target_arg': "sum_"+proj.target,
                'add_args': add_args_header,
                'psp': psp  % ids,
                'thread_init': self._templates['rate_psp']['thread_init'][Global.config['precision']][operation],
                'post_index': ids['post_index']
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
                'float_prec': Global.config['precision'],
                'idx_type': idx_type
            }
        else:
            # Should be equal to ProjectionGenerator._configure_template_ids()
            idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)
            id_dict = {
                'id_proj': proj.id,
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'idx_type': idx_type,
                'size_type': size_type
            }

            #
            # ELLPACK - partition
            conn_header = ELL_CUDA.conn_templates['conn_header'] % id_dict
            conn_call = ELL_CUDA.conn_templates['conn_call'] % id_dict
            ell_ids = {
                'idx_type': idx_type,
                'local_index': "[j*post_size+i]",
                'semiglobal_index': '[i]',
                'global_index': '[0]',
                'pre_index': '[rank_pre[j*post_size+i]]',
                'post_index': '[rank_post[i]]',
                'pre_prefix': 'pre_',
                'post_prefix': 'post_'
            }
            body_code = ELL_CUDA.conn_templates['rate_psp']['body'][operation] % {
                'idx_type': idx_type,
                'float_prec': Global.config['precision'],
                'id_proj': proj.id,
                'conn_args': conn_header,
                'target_arg': "sum_"+proj.target,
                'add_args': add_args_header,
                'psp': psp  % ell_ids,
                'thread_init': ELLR_CUDA.conn_templates['rate_psp']['thread_init'][Global.config['precision']][operation],
                'post_index': ell_ids['post_index']
            }
            header_code = ELL_CUDA.conn_templates['rate_psp']['header'] % {
                'idx_type': idx_type,
                'float_prec': Global.config['precision'],
                'id': proj.id,
                'conn_args': conn_header,
                'target_arg': "sum_"+proj.target,
                'add_args': add_args_header
            }

            #
            # Coordinate - partition
            conn_header = COO_CUDA.conn_templates['conn_header'] % id_dict
            conn_call = COO_CUDA.conn_templates['conn_call'] % id_dict
            coo_ids = {
                'local_index': "[j]",
                'semiglobal_index': '[i]',
                'global_index': '[0]',
                'pre_index': '[column_indices[j]]',
                'post_index': '[row_indices[j]]',
                'pre_prefix': 'pre_',
                'post_prefix': 'post_',
            }
            body_code += COO_CUDA.conn_templates['rate_psp']['body'][operation] % {
                'float_prec': Global.config['precision'],
                'idx_type': idx_type,
                'size_type': size_type,
                'id_proj': proj.id,
                'conn_args': conn_header,
                'target_arg': "sum_"+proj.target,
                'add_args': add_args_header,
                'psp': psp  % coo_ids,
                'thread_init': COO_CUDA.conn_templates['rate_psp']['thread_init'][Global.config['precision']][operation],
                'post_index': coo_ids['post_index']
            }
            header_code += COO_CUDA.conn_templates['rate_psp']['header'] % {
                'float_prec': Global.config['precision'],
                'id': proj.id,
                'conn_args': conn_header,
                'target_arg': "sum_"+proj.target,
                'add_args': add_args_header
            }

            # update dependencies
            add_args_call_coo = add_args_call
            add_args_call_ell = add_args_call
            for dep in proj.synapse_type.description['psp']['dependencies']:
                add_args_call_coo = add_args_call_coo.replace("gpu_"+dep+",", "gpu_"+dep+"->coo,")
                add_args_call_ell = add_args_call_ell.replace("gpu_"+dep+",", "gpu_"+dep+"->ell,")

            call_code = HYB_CUDA.conn_templates['rate_psp']['call'] % {
                'id_proj': proj.id,
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'conn_args': conn_call,
                'target': proj.target,
                'target_arg': ", pop%(id_post)s.gpu__sum_%(target)s" % {'id_post': proj.post.id, 'target': proj.target},
                'add_args_coo': add_args_call_coo,
                'add_args_ell': add_args_call_ell,
                'float_prec': Global.config['precision']
            }

        # Take delays into account if any
        if proj.max_delay > 1:
            # Delayed variables
            if isinstance(proj.pre, PopulationView):
                delayed_variables = proj.pre.population.delayed_variables
            else:
                delayed_variables = proj.pre.delayed_variables

            id_pre = str(proj.pre.id)
            for var in sorted(list(set(delayed_variables))):
                call_code = call_code.replace("pop"+id_pre+".gpu_"+var, "pop"+id_pre+".gpu_delayed_"+var+"[proj"+str(proj.id)+".delay-1]")

        # Profiling
        if self._prof_gen:
            call_code = self._prof_gen.annotate_computesum_rate(proj, call_code)

        return header_code, body_code, call_code

    def _computesum_spiking(self, proj):
        """
        Generate code for the spike propagation. As ANNarchy supports a set of
        different data structures, this method split in up into several sub
        functions.

        In contrast to _computesum_rate() the spike propagation kernel need
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

        idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)
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
            'post_index': '[post_rank]',

            # CPP types
            'idx_type': idx_type,
            'size_type': size_type
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
                    'cpp': var['cpp'],
                    'bounds': get_bounds(var),
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
                    'exact': var['cpp'].replace('(t)', '(t-1)')
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
"""

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

            attr_type, attr_dict = self._get_attr_and_type(proj, dep)
            attr_ids = {
                'id_proj': proj.id,
                'name': attr_dict['name'],
                'type': attr_dict['ctype']
            }

            if attr_type == 'par' and attr_dict['locality'] == "global":
                kernel_args += ", const %(type)s %(name)s" % attr_ids
                kernel_args_call += ", proj%(id_proj)s.%(name)s" % attr_ids

                # replace any occurences of this parameter
                if event_driven_code.strip() != '':
                    event_driven_code = event_driven_code.replace(attr_dict['name']+'%(global_index)s', attr_dict['name'])
                if pre_spike_code.strip() != '':
                    pre_spike_code = pre_spike_code.replace(attr_dict['name']+'%(global_index)s', attr_dict['name'])
            else:
                kernel_args += ", %(type)s* %(name)s" % attr_ids
                kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % attr_ids

        #
        # Finally, fill the templates,
        # we start with the event-driven component.
        #
        if len(pre_spike_code) == 0 and len(psp_code) == 0:
            header = ""
            body = ""
            call = ""
        else:
            # select the correct template
            template = self._templates['spike_transmission']['event_driven'][proj._storage_order]

            # Connectivity description, we need to read-out the view
            # which represents the pre-synaptic entries which means
            # columns in post-to-pre and rows for pre-to-post orientation.
            if proj._storage_order == "post_to_pre":
                conn_header = "%(size_type)s* col_ptr, %(idx_type)s* row_idx, %(idx_type)s* inv_idx, %(float_prec)s *w" % ids
                conn_call = "proj%(id_proj)s.gpu_col_ptr, proj%(id_proj)s.gpu_row_idx, proj%(id_proj)s.gpu_inv_idx, proj%(id_proj)s.gpu_w" % ids
            else:
                conn_header = "%(size_type)s* row_ptr, %(idx_type)s* col_idx, %(float_prec)s *w" % ids
                conn_call = "proj%(id_proj)s._gpu_row_ptr, proj%(id_proj)s._gpu_col_idx, proj%(id_proj)s.gpu_w" % ids

            # Population sizes
            pre_size = proj.pre.size if isinstance(proj.pre, Population) else proj.pre.population.size
            post_size = proj.post.size if isinstance(proj.post, Population) else proj.post.population.size

            # PSP targets
            targets_call = ""
            targets_header = ""
            target_list = proj.target if isinstance(proj.target, list) else [proj.target]
            for target in target_list:
                targets_call += ", pop%(id_post)s.gpu_g_"+target
                targets_header += (", %(float_prec)s* g_"+target) % {'float_prec': Global.config['precision']}

            # finalize call, body and header
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
                'event_driven': tabify(event_driven_code % ids, 2),
                'psp': tabify(psp_code, 3),
                'pre_event': tabify(pre_spike_code % ids, 3),
                'pre_size': pre_size,
                'post_size': post_size,
            }
            header = template['header'] % {
                'id': proj.id,
                'float_prec': Global.config['precision'],
                'conn_header': conn_header + targets_header,
                'kernel_args': kernel_args
            }

        # If the synaptic transmission is not event-based,
        # we need to add a rate-coded-like kernel.
        if 'psp' in  proj.synapse_type.description.keys():
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

            call += template['call'] % {
                'id_proj': proj.id,
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'target_arg': ', pop%(id_post)s.gpu_g_%(target)s' % {'id_post': proj.post.id, 'target': proj.target},
                'target': proj.target,
                'kernel_args': kernel_args_call,
                'float_prec': Global.config['precision']
            }
            body += template['body'] % {
                'id_proj': proj.id,
                'target_arg': proj.target,
                'kernel_args':  kernel_args,
                'psp': psp_code,
                'pre_code': tabify(pre_spike_code % ids, 3),
                'float_prec': Global.config['precision']
            }
            header += template['header'] % {
                'id': proj.id,
                'kernel_args': kernel_args,
                'target_arg': 'g_'+proj.target,
                'float_prec': Global.config['precision']
            }

        return header, body, call


    def _declaration_accessors(self, proj, single_matrix):
        """
        Extend basic declaration statements by CUDA streams.
        """
        declaration, accessor = ProjectionGenerator._declaration_accessors(self, proj, single_matrix)

        declaration['cuda_stream'] = BaseTemplates.cuda_stream
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
            # The variable dep is part of pre-/post population
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

            # The variable dep is part of the projection
            else:
                attr_type, attr_dict = ProjectionGenerator._get_attr_and_type(proj, dep)

                if attr_type == "par":
                    ids = {
                        'id_proj': proj.id,
                        'type': attr_dict['ctype'],
                        'name': attr_dict['name']
                    }

                    if dep in proj.synapse_type.description['global']:
                        kernel_args += ", const %(type)s %(name)s" % ids
                        kernel_args_call += ", proj%(id_proj)s.%(name)s" % ids
                    else:
                        kernel_args += ", %(type)s* %(name)s" % ids
                        kernel_args_call += ", proj%(id_proj)s.gpu_%(name)s" % ids


                elif attr_type == "var":
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
                else:
                    raise ValueError("attr_type for variable " + dep +" is invalid")

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

    def _replace_random(self, loc_eqs, loc_idx, glob_eqs, random_distributions):
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
            print(dist)
            print(loc_eqs)
            if dist['dist'] == "Uniform":
                dist_ids = {
                    'postfix': prec_extension,
                    'rd': dist['name'],
                    'min': dist['args'].split(',')[0],
                    'max': dist['args'].split(',')[1],
                    'local_index': loc_idx
                }

                if dist["locality"] == "local":
                    term = """( curand_uniform%(postfix)s( &state_%(rd)s%(local_index)s ) * (%(max)s - %(min)s) + %(min)s )""" % dist_ids
                    loc_pre += "%(prec)s %(name)s = %(term)s;" % {'prec': Global.config['precision'], 'name': dist['name'], 'term': term}

                    # suppress local index
                    loc_eqs = loc_eqs.replace(dist['name']+loc_idx, dist['name'])
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

        if proj._storage_format == "csr":
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
                    event_driven_code += var['cpp'] + '\n'

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
            post_code += post_eq['cpp'] + '\n'
            post_code += get_bounds(post_eq) + '\n'

            # add dependencies, only right side!
            for deps in post_eq['dependencies']:
                post_deps.append(deps)
            # left side of equations is not part of dependencies
            post_deps.append(post_eq['name'])

        # Create add_args for event-driven eqs and post_event
        kernel_deps = list(set(post_deps+event_deps)) # variables can occur in several eqs
        for dep in kernel_deps:
            if dep == "w":
                continue

            attr_type, attr_dict = self._get_attr_and_type(proj, dep)
            attr_ids = {
                'id': proj.id, 'type': attr_dict['ctype'], 'name': attr_dict['name']
            }
            if attr_type == 'par' and attr_dict['locality'] == "global":
                add_args_header += ', const %(type)s %(name)s' % attr_ids
                add_args_call += ', proj%(id)s.%(name)s' % attr_ids

                if post_code.strip != '':
                    post_code = post_code.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])
                if event_driven_code.strip() != '':
                    event_driven_code = event_driven_code.replace(attr_dict['name']+"%(global_index)s", attr_dict['name'])
            else:
                add_args_header += ', %(type)s* %(name)s' % attr_ids
                add_args_call += ', proj%(id)s.gpu_%(name)s' % attr_ids

        if proj._storage_format == "csr":
            idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)
            conn_ids = {'idx_type': idx_type, 'size_type': size_type}

            if proj._storage_order == "post_to_pre":
                conn_header = "%(size_type)s* row_ptr, %(idx_type)s* col_idx, " % conn_ids
                conn_call = ", proj%(id_proj)s.gpu_row_ptr, proj%(id_proj)s.gpu_pre_rank"
            else:
                conn_header = "%(size_type)s* col_ptr, %(idx_type)s* row_idx, " % conn_ids
                conn_call = ", proj%(id_proj)s.gpu_col_ptr, proj%(id_proj)s.gpu_row_idx"

            templates = self._templates['post_event'][proj._storage_order]

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
            'event_driven': tabify(event_driven_code % ids, 2),
            'post_code': tabify(post_code % ids, 2),
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

        proc_attr = []
        for attr in proj.synapse_type.description['parameters']+proj.synapse_type.description['variables']:
            # avoid doublons
            if attr['name'] in proc_attr:
                continue

            attr_type = "parameter" if attr in proj.synapse_type.description['parameters'] else "variable"
            locality = attr['locality']
            if attr_type == "parameter" and locality == "global":
                continue

            #
            # Host -> Device
            #
            host_device_transfer += self._templates['host_to_device'][locality] % {
                'id': proj.id,
                'name': attr['name'],
                'type': attr['ctype']
            }

            #
            # Device -> Host
            #
            device_host_transfer += self._templates['device_to_host'][locality] % {
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

        # Add pre_rank/post_rank identifier if needed
        rk_assign = ""
        if locality == "semiglobal":
            rk_assign += "%(idx_type)s rk_post = rank_post%(semiglobal_index)s;\n"

        elif locality=="local":
            # rk_pre/rk_pre depend on the matrix format
            if proj._storage_format in ["csr"] :
                rk_assign += "%(idx_type)s rk_pre = rank_pre%(local_index)s;\n"
            elif proj._storage_format in ["dense"]:
                pass
            else:
                rk_assign += "%(idx_type)s rk_post = rank_post%(semiglobal_index)s;\n"
                rk_assign += "%(idx_type)s rk_pre = rank_pre%(local_index)s;\n"

        # finalize rank assignment code
        rk_assign = tabify(rk_assign, 2 if proj._storage_format == "csr" else 3)

        # Gather pre-loop declaration (dt/tau for ODEs)
        pre_loop = ""
        for var in proj.synapse_type.description['variables']:
            if var['locality'] == locality:
                if 'pre_loop' in var.keys() and len(var['pre_loop']) > 0:
                    pre_loop += Global.config['precision'] + ' ' + var['pre_loop']['name'] + ' = ' + var['pre_loop']['value'] + ';\n'
            else:
                continue

        # Global parameters have no index
        for syn_dep in syn_deps:
            attr_type, attr_dict = self._get_attr_and_type(proj, syn_dep)

            if attr_type == "par" and attr_dict['locality'] == "global" :
                equations = equations.replace(attr_dict["name"]+"%(global_index)s", attr_dict["name"])

                if pre_loop.strip() != '':
                    pre_loop = pre_loop.replace(attr_dict["name"]+"%(global_index)s", attr_dict["name"])

        # Finalize equations, add pre-loop and/or rank assignment
        equations = (rk_assign + equations) % ids
        if pre_loop.strip() != '':
            pre_loop = """\n// Updating the step sizes\n""" + pre_loop % ids

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
        pad = 2 if proj._storage_format == "csr" else 3
        local_eq = generate_equation_code(proj.id, proj.synapse_type.description, 'local', 'proj', padding=pad, wrap_w="plasticity")

        # Something to do?
        if global_eq.strip() == '' and semiglobal_eq.strip() == '' and local_eq.strip() == '':
            return "", "", ""

        # Modify the default dictionary for specific formats
        ids = deepcopy(self._template_ids)
        if proj._storage_format == "ell":
            ids['pre_index'] = '[rk_pre]'
            ids['post_index'] = '[rk_post]'
        elif proj._storage_format == "csr":
            ids['pre_index'] = '[rk_pre]'
            ids['post_index'] = '[rk_post]'

        # CPP type for indices
        idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)

        # some commonly needed ids
        ids.update({
            'id_proj': proj.id,
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'float_prec': Global.config['precision'],
            'idx_type': idx_type,
            'size_type': size_type
        })

        # generate the code
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

        # replace the random distributions
        local_eq, global_eq = self._replace_random(local_eq, ids['local_index'], global_eq, proj.synapse_type.description['random_distributions'])

        #
        # replace local function calls
        if len(proj.synapse_type.description['functions']) > 0:
            global_eq, semiglobal_eq, local_eq = self._replace_local_funcs(proj, global_eq, semiglobal_eq, local_eq)

        # connectivity
        conn_header = self._templates['conn_header'] % ids
        conn_call = self._templates['conn_call'] % ids

        # we seperated the execution of global/semiglobal/local into three kernels
        # as the threads would have two different loads.
        if global_eq.strip() != '':
            body_dict = {
                'kernel_args': kernel_args_global,
                'global_eqs': global_eq,
                'pre_loop':  global_pre_code,
            }
            body_dict.update(ids)
            body += self._templates['synapse_update']['global']['body'] % body_dict

            header_dict = {
                'kernel_args': kernel_args_global,
            }
            header_dict.update(ids)
            header += self._templates['synapse_update']['global']['header'] % header_dict

            call_dict = {
                'target': proj.target[0] if isinstance(proj.target, list) else proj.target,
                'kernel_args_call': kernel_args_call_global,
            }
            call_dict.update(ids)
            global_call = self._templates['synapse_update']['global']['call'] % call_dict

        if semiglobal_eq.strip() != '':
            body_dict = {
                'kernel_args': kernel_args_semiglobal,
                'semiglobal_eqs': semiglobal_eq,
                'pre_loop': semiglobal_pre_code,
            }
            body_dict.update(ids)
            body += self._templates['synapse_update']['semiglobal']['body'] % body_dict

            header_dict = {
                'kernel_args': kernel_args_semiglobal,
            }
            header_dict.update(ids)
            header += self._templates['synapse_update']['semiglobal']['header'] % header_dict

            call_dict = {
                'target': proj.target[0] if isinstance(proj.target, list) else proj.target,
                'kernel_args_call': kernel_args_call_semiglobal,
            }
            call_dict.update(ids)
            semiglobal_call = self._templates['synapse_update']['semiglobal']['call'] % call_dict

        if local_eq.strip() != '':
            body_dict = {
                'conn_args': conn_header,
                'kernel_args': kernel_args_local,
                'local_eqs': local_eq,
                'pre_loop': tabify(local_pre_code,1)
            }
            body_dict.update(ids)
            body += self._templates['synapse_update']['local']['body'] % body_dict

            header_dict = {
                'conn_args': conn_header,
                'kernel_args': kernel_args_local
            }
            header_dict.update(ids)
            header += self._templates['synapse_update']['local']['header'] % header_dict

            call_dict = {
                'target': proj.target[0] if isinstance(proj.target, list) else proj.target,
                'conn_args_call': conn_call,
                'kernel_args_call': kernel_args_call_local
            }
            call_dict.update(ids)
            local_call = self._templates['synapse_update']['local']['call'] % call_dict

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
