"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import ANNarchy

# ANNarchy objects
from ANNarchy.core import Global
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.models.Synapses import DefaultRateCodedSynapse
from ANNarchy.intern.ConfigManagement import get_global_config
from ANNarchy.intern import Messages

# Code templates
from ANNarchy.generator.Projection.ProjectionGenerator import ProjectionGenerator, get_bounds
from ANNarchy.generator.Projection.SingleThread import *

# Useful functions
from ANNarchy.generator.Utils import generate_equation_code, tabify, remove_trailing_spaces, check_avx_instructions, determine_idx_type_for_projection

import re
from copy import deepcopy

class SingleThreadGenerator(ProjectionGenerator):
    """
    The class is responsible to generate the C++ header definition for a Population
    object. The code is intendend to run on a single CPU core.
    """
    def __init__(self, profile_generator, net_id):
        # The super here calls all the base classes, so first
        # ProjectionGenerator and afterwards OpenMPConnectivity
        # TODO: this is python 2 syntax
        super(SingleThreadGenerator, self).__init__(profile_generator, net_id)

    def header_struct(self, proj, annarchy_dir):
        """
        Generate the projection header for a given projection. The resulting
        code will be stored in a file called proj<unique_id>.hpp in the
        directory indicated by annarchy_dir.

        This function is called from the CodeGenerator if get_global_config('num_threads')
        was set to 1.

        Returns:

        * proj_desc: a dictionary with all call statements for the required
                     operations (i. e. compute_psp, update_synapse, etc.)
        """
        # Initial state
        self._templates = deepcopy(BaseTemplates.single_thread_templates)
        self._template_ids = {}

        # Select the C++ connectivity template
        sparse_matrix_include, sparse_matrix_format, sparse_matrix_args, single_matrix = self._select_sparse_matrix_format(proj)

        # Update template fill elements
        self._configure_template_ids(proj)

        # Generate declarations and accessors for the variables
        decl, accessor = self._declaration_accessors(proj, single_matrix)

        # Initiliaze the projection
        init_weights, init_delays, init_parameters_variables = self._init_parameters_variables(proj, single_matrix)

        # Synaptic plasticity
        update_prefix, update_variables = self._update_synapse(proj)

        # Update the random distributions
        init_rng = self._init_random_distributions(proj)
        update_rng = self._update_random_distributions(proj)

        post_event = self._post_event(proj)

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

        # Detect delays to generate the code
        has_delay = proj.max_delay > 1
        if has_delay:
            update_max_delay, reset_ring_buffer = self._update_max_delay(proj)
        else:
            update_max_delay = ""
            reset_ring_buffer = ""

        # Hybrid format requires ell_size argument
        add_args = ""
        if proj._storage_format == "hyb":
            # TODO: readout connectivity file/user argument?
            if hasattr(proj, "_ell_size"):
                add_args = ", " + str(proj._ell_size)
            else:
                add_args = ", std::numeric_limits<unsigned int>::max()"

        # Generate connectivity call
        # This can be either construct from LIL (the matrix was build up in Python)
        # or a C++ side implemented pattern.
        if 'declare_connectivity_matrix' not in proj._specific_template.keys():
            connector_call = self._connectivity_init(proj, sparse_matrix_format, sparse_matrix_args) % {
                'sparse_format': sparse_matrix_format,
                'init_weights': init_weights,
                'init_delays': init_delays,
                'rng_idx': "[0]",
                'add_args': add_args,
                'num_threads': "",
                'float_prec': get_global_config('precision'),
                'idx_type': determine_idx_type_for_projection(proj)[0]
            }
            declare_connectivity_matrix = ""
            access_connectivity_matrix = ""
        else:
            # The user is responsible to define the connectivity related variables
            sparse_matrix_format = "SpecificConnectivity"
            sparse_matrix_args = ""
            sparse_matrix_include = "#include \"Specific.hpp\"\n"
            connector_call = ""
            declare_connectivity_matrix = proj._specific_template['declare_connectivity_matrix']
            access_connectivity_matrix = proj._specific_template['access_connectivity_matrix']

        # Local functions
        decl['parameters_variables'] += self._local_functions(proj)

        # Memory management
        size_in_bytes = self._size_in_bytes(proj)
        clear_container = self._clear_container(proj)

        # Structural plasiticity
        creating = self.creating(proj)
        pruning = self.pruning(proj)

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

        # Gather all code segments produced so far
        final_code_dict = {
            # version tag
            'annarchy_version': ANNarchy.__release__,
            # fill code templates
            'id_pre': proj.pre.id,
            'id_post': proj.post.id,
            'id_proj': proj.id,
            'name_pre': proj.pre.name,
            'name_post': proj.post.name,
            'target': proj.target,
            'sparse_matrix_include': sparse_matrix_include,
            'sparse_format': sparse_matrix_format,
            'sparse_format_args': sparse_matrix_args,
            'include_additional': include_additional,
            'include_profile': include_profile,
            'struct_additional': struct_additional,
            'declare_connectivity_matrix': declare_connectivity_matrix,
            'access_connectivity_matrix': access_connectivity_matrix,
            'declare_delays': decl['declare_delay'] if has_delay else "",
            'declare_event_driven': decl['event_driven'] if has_event_driven else "",
            'declare_rng': decl['rng'],
            'declare_parameters_variables': decl['parameters_variables'],
            'declare_additional': decl['additional'],
            'declare_profile': declare_profile,
            'connector_call': connector_call,
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
            'update_max_delay': update_max_delay,
            'reset_ring_buffer': reset_ring_buffer,
            'post_event': post_event,
            'access_parameters_variables': accessor,
            'access_additional': access_additional,
            'size_in_bytes': size_in_bytes,
            'clear_container': clear_container,
            'float_prec': get_global_config('precision'),
            'creating': creating,
            'pruning': pruning
        }

        # Generate the final code
        final_code = self._templates['projection_header'] % final_code_dict

        # remove right-trailing white spaces
        final_code = remove_trailing_spaces(final_code)

        # Store file (default: $(cwd)/annarchy/)
        with open(annarchy_dir+'/generate/net'+str(self._net_id)+'/proj'+str(proj.id)+'.hpp', 'w') as ofile:
            ofile.write(final_code)

        # Dictionary for inclusions in ANNarchy.cpp
        proj_desc = {
            'include': """#include "proj%(id)s.hpp"\n""" % {'id': proj.id},
            'extern': """extern ProjStruct%(id)s proj%(id)s;\n"""% {'id': proj.id},
            'instance': """ProjStruct%(id)s proj%(id)s;\n"""% {'id': proj.id},
            'init': """    proj%(id)s.init_projection();\n""" % {'id' : proj.id}
        }

        # Required call statements for the main loop (singleStep() in ANNarchy.cpp)
        proj_desc['compute_psp'] = """\tproj%(id)s.compute_psp();\n""" % {'id' : proj.id}
        proj_desc['update'] = "" if update_variables == "" else """\tproj%(id)s.update_synapse();\n""" % {'id': proj.id}
        proj_desc['rng_update'] = "" if update_rng == "" else """\tproj%(id)s.update_rng();\n""" % {'id': proj.id}
        proj_desc['post_event'] = "" if post_event == "" else """\tproj%(id)s.post_event();\n""" % {'id': proj.id}

        return proj_desc

    def _configure_template_ids(self, proj):
        """
        Assign the correct template code dictionary (self._templates) based on projection storage format and storage order.
        Also sets the basic template ids (self._template_ids) which are indices and index data field names.

        **Note:**

        In the ANNarchy 4.7.0 release only the *compressed sparse row* (CSR) format allows the 'pre_to_post' ordering.
        """
        # Sanity check
        if proj._storage_order not in ["post_to_pre", "pre_to_post"]:
            raise ValueError("storage_order argument must be either 'post_to_pre' or 'pre_to_post'")

        # Index data types depend on the matrix dimension
        # HD (1st Dec. 2021):   until now, the data type optimization is disabled for
        #                       spiking models. I want to adjust new codes already as I
        #                       hope to update the spike code generation soon ...
        idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)

        # Some common ids
        self._template_ids.update({
            'id_proj' : proj.id,
            'target': proj.target,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,
            'idx_type': idx_type,
            'size_type': size_type,
            'float_prec': get_global_config('precision'),
            'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
            'post_prefix': 'pop'+ str(proj.post.id) + '.',
        })

        # The variable fields and indices depends on matrix format
        # as well as the matrix orientation.
        if proj._storage_format == "lil":
            if proj._storage_order == "post_to_pre":
                self._templates.update(LIL_SingleThread.conn_templates)
                self._template_ids.update(LIL_SingleThread.conn_ids)
            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "coo":
            if proj._storage_order == "post_to_pre":
                self._templates.update(COO_SingleThread.conn_templates)
                self._template_ids.update(COO_SingleThread.conn_ids)
            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "dia":
            if proj._storage_order == "post_to_pre":
                self._templates.update(DIA_SingleThread.conn_templates)
                self._template_ids.update(DIA_SingleThread.conn_ids)
            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "bsr":
            if proj._storage_order == "post_to_pre":
                self._templates.update(BSR_SingleThread.conn_templates)
                self._template_ids.update(BSR_SingleThread.conn_ids)
            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "csr":
            if proj._storage_order == "post_to_pre":
                self._templates.update(CSR_SingleThread.conn_templates)
                self._template_ids.update(CSR_SingleThread.conn_ids)
            else:
                self._templates.update(CSR_T_SingleThread.conn_templates)
                self._template_ids.update(CSR_T_SingleThread.conn_ids)

        elif proj._storage_format == "ellr":
            if proj._storage_order == "post_to_pre":
                self._templates.update(ELLR_SingleThread.conn_templates)
                self._template_ids.update(ELLR_SingleThread.conn_ids)
            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "sell":
            if proj._storage_order == "post_to_pre":
                self._templates.update(SELL_SingleThread.conn_templates)
                self._template_ids.update(SELL_SingleThread.conn_ids)
            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "ell":
            if proj._storage_order == "post_to_pre":
                self._templates.update(ELL_SingleThread.conn_templates)
                self._template_ids.update(ELL_SingleThread.conn_ids)

            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "hyb":
            if proj._storage_order == "post_to_pre":
                self._templates.update(HYB_SingleThread.conn_templates)
                # Implementation note:
                #   In contrast to most of the other formats, we can not define the
                #   indices by one set as they are different for coo/ell part
            else:
                raise Global.InvalidConfiguration("    "+proj.name+": storage_format = " + proj._storage_format + " and storage_order = " + proj._storage_order )

        elif proj._storage_format == "dense":
            if proj._storage_order == "post_to_pre":
                if proj._has_pop_view:
                    self._templates.update(Dense_PV_SingleThread.conn_templates)
                    self._template_ids.update(Dense_PV_SingleThread.conn_ids)
                else:
                    self._templates.update(Dense_SingleThread.conn_templates)
                    self._template_ids.update(Dense_SingleThread.conn_ids)
            else:
                self._templates.update(Dense_T_SingleThread.conn_templates)
                self._template_ids.update(Dense_T_SingleThread.conn_ids)

        else:
            raise Messages.CodeGeneratorException("    "+proj.name+": no template ids available to generate single-thread code and storage_format="+proj._storage_format)

    def creating(self, proj):
        """
        Synapse creation based on neural state variables.

        TODO: documentation
        """
        if 'creating' not in proj.synapse_type.description.keys():
            return ""

        if proj._storage_format != "lil":
            raise NotImplementedError("Structural plasticity is only available for LIL structures.")

        creating_structure = proj.synapse_type.description['creating']

        # Random stuff
        proba = ""
        proba_init = ""
        if 'proba' in creating_structure['bounds'].keys():
            val = creating_structure['bounds']['proba']
            proba += '&&(unif(rng[0])<' + val + ')'
            proba_init += "std::uniform_real_distribution<double> unif(0.0, 1.0);"
        if  creating_structure['rd']:
            proba_init += "\n        " +  creating_structure['rd']['template'] + ' rd(' + creating_structure['rd']['args'] + ');'

        # delays
        delay = ""
        if 'd' in creating_structure['bounds'].keys():
            d = int(creating_structure['bounds']['delay']/get_global_config('dt'))
            if proj.max_delay > 1 and proj.uniform_delay == -1:
                if d > proj.max_delay:
                    Messages._error('creating: you can not add a delay higher than the maximum of existing delays')

                delay = ", " + str(d)
            else:
                if d != proj.uniform_delay:
                    Messages._error('creating: you can not add a delay different from the others if they were constant.')

        creation_ids = deepcopy(self._template_ids)
        creating_condition = creating_structure['cpp'] % creation_ids

        creation_ids.update({
            'eq': creating_structure['eq'],
            'condition': creating_condition,
            'weights': 0.0 if not 'w' in creating_structure['bounds'].keys() else creating_structure['bounds']['w'],
            'proba' : proba, 'proba_init': proba_init,
            'delay': delay
        })
        creating = self._templates['structural_plasticity']['create'] % creation_ids

        return creating

    def pruning(self, proj):
        """
        Synapse removal based on neural state variables.

        TODO: documentation
        """
        if 'pruning' not in proj.synapse_type.description.keys():
            return ""

        if proj._storage_format != "lil":
            raise NotImplementedError("Structural plasticity is only available for LIL structures.")

        pruning_structure = proj.synapse_type.description['pruning']

        proba = ""
        proba_init = ""
        if 'proba' in pruning_structure['bounds'].keys():
            val = pruning_structure['bounds']['proba']
            proba = '&&(unif(rng[0])<' + val + ')'
            proba_init = "std::uniform_real_distribution<double> unif(0.0, 1.0);"
        if pruning_structure['rd']:
            proba_init += "\n        " +  pruning_structure['rd']['template'] + ' rd(' + pruning_structure['rd']['args'] + ');'

        pruning_ids = deepcopy(self._template_ids)
        pruning_condition = pruning_structure['cpp'] % pruning_ids

        pruning_ids.update({
            'eq': pruning_structure['eq'],
            'condition': pruning_condition,
            'proba' : proba,
            'proba_init': proba_init
        })
        pruning = self._templates['structural_plasticity']['prune'] % pruning_ids

        return pruning

    def _computesum_rate(self, proj):
        """
        Create the c++ code for post-synaptic potential computation.
        """
        # Specific projection (e. g. convolution, pooling or user-defined stuff)
        if 'psp_code' in proj._specific_template.keys():
            psp_code = proj._specific_template['psp_code']
            if 'psp_prefix' in proj._specific_template.keys():
                psp_prefix = proj._specific_template['psp_prefix']
            else:
                psp_prefix = ""

            if self._prof_gen:
                psp_code = self._prof_gen.annotate_computesum_rate(proj, psp_code)

            return psp_prefix, psp_code

        # Dictionary of keywords to transform the parsed equations
        ids = deepcopy(self._template_ids)

        # For a default continous transmission we can use a hand-written
        # AVX implementation or unrolled versions of the BSR
        if isinstance(proj.synapse_type, DefaultRateCodedSynapse) or \
                      proj.synapse_type.description['psp']['eq']=="w*pre.r":

            simd_type = None

            # check if SIMD operations are available. As higher order methods
            # always contain the lower, we need to test in order SSE4, AVX, AVX512
            if check_avx_instructions("sse4_1"):
                simd_type = "sse"

            if check_avx_instructions("avx"):
                simd_type = "avx"

            if check_avx_instructions("avx512f"):
                simd_type = "avx512"

            # Does our current system support SIMD and does the selected format offer an implementation?
            if simd_type is not None and "vectorized_default_psp" in self._templates.keys():
                try:
                    # The default weighted sum can be re-formulated for single weights
                    if proj._has_single_weight():
                        template = self._templates["vectorized_default_psp"][simd_type]["single_w"]
                    else:
                        template = self._templates["vectorized_default_psp"][simd_type]["multi_w"]

                    # the access to pre-synaptic firing depends on the delay
                    if proj.max_delay <= 1:
                        # no synaptic delay
                        ids.update({
                            'get_r': ids['pre_prefix']+"r.data()",
                        })

                        psp_code = template["sum"][get_global_config('precision')] % ids

                        if self._prof_gen:
                            psp_code = self._prof_gen.annotate_computesum_rate(proj, psp_code)

                        return "", psp_code

                    elif proj.uniform_delay != -1 and proj.max_delay > 1:
                        # Uniform delay
                        ids.update({
                            'get_r': ids['pre_prefix']+"_delayed_r[delay-1].data()",
                        })

                        psp_code = template["sum"][get_global_config('precision')] % ids

                        if self._prof_gen:
                            psp_code = self._prof_gen.annotate_computesum_rate(proj, psp_code)

                        return "", psp_code

                    else:
                        # HD (3rd June 2021): for non-uniform delays I doubt that it's worth the effort, so we proceed with
                        #                     general code generation
                        pass

                except KeyError:
                    # No fitting code found, so we fall back to normal code generation
                    # TODO: add internal error log, which key was missing?
                    Messages._debug("No SIMD implementation found, fallback to non-SIMD code")
                    template = ""

            else:
                # Other optimizations like loop unroll etc?
                if proj._storage_format == "bsr":
                    try:
                        # not yet implemented
                        if proj._has_single_weight():
                            raise KeyError

                        from ANNarchy.generator.Projection.ProjectionGenerator import determine_bsr_blocksize
                        if hasattr(proj, "_bsr_size"):
                            blockDim = proj._bsr_size
                        else:
                            blockDim = determine_bsr_blocksize(proj.pre.population.size if isinstance(proj.pre, PopulationView) else proj.pre.size, proj.post.population.size if isinstance(proj.post, PopulationView) else proj.post.size)

                        # Loop unroll depends on the block-size
                        unrolled_template = self._templates["unrolled_default_psp"][blockDim]

                        # Check if we implemented a SIMD version
                        if simd_type in unrolled_template.keys():
                            template = unrolled_template[simd_type]['multi_w']['sum'][get_global_config('precision')]
                        else:
                            template = unrolled_template['none']['multi_w']["sum"]

                        # the access to pre-synaptic firing depends on the delay
                        if proj.max_delay <= 1:
                            # no synaptic delay
                            ids.update({
                                'get_r': ids['pre_prefix']+"r.data()",
                            })

                            psp_code = template % ids

                            if self._prof_gen:
                                psp_code = self._prof_gen.annotate_computesum_rate(proj, psp_code)

                            return "", psp_code
                        else:
                            # HD (23th Nov 2021): the unrolled code templates for specific kernels is a highly
                            #                     experimental feature. I will implement the delay case if we
                            #                     are sure that we really use this.
                            raise KeyError

                    except KeyError:
                        # No fitting code found, so we fall back to normal code generation
                        # TODO: add internal error log, which key was missing?
                        Messages._debug("No SIMD implementation found, fallback to non-SIMD code")
                        template = ""

        # Default variables needed in psp_code
        psp_prefix = tabify("%(float_prec)s sum;" % {'float_prec': get_global_config('precision')}, 2)

        # Choose the corresponding summation template
        try:
            template = self._templates['rate_coded_sum']
        except KeyError:
           raise Messages.CodeGeneratorException("    SingleThreadGenerator: no template for this configuration available")

        # The psp uses in almost all cases one time the pre-synaptic index,
        # therefore I want to spare the usage of the explicit rk_pre variable.
        if proj._storage_format == "lil":
            ids['pre_index'] = "[pre_rank[i][j]]"
            ids['post_index'] = "[post_rank[i]]"
        elif proj._storage_format == "csr":
            ids['pre_index'] = "[col_idx[j]]"

        # Dependencies
        dependencies = list(set(proj.synapse_type.description['dependencies']['pre']))

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

        # Allow the use of global variables in psp (issue60)
        for var in dependencies:
            if var in proj.pre.neuron_type.description['global']:
                psp = psp.replace("%(pre_prefix)s"+var+"%(pre_index)s", "%(pre_prefix)s"+var+"%(global_index)s")

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
                        Messages._print(proj.synapse_type.description['psp']['eq'])
                        Messages._error('The psp accesses a global variable with a non-uniform delay!')


            else: # Uniform delays
                for var in delayed_variables:
                    if var in proj.pre.neuron_type.description['local']:
                        psp = psp.replace(
                            '%(pre_prefix)s'+var+'%(pre_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_u)s%(pre_index)s'
                        )
                    else:
                        psp = psp.replace(
                            '%(pre_prefix)s'+var+'%(global_index)s',
                            '%(pre_prefix)s_delayed_'+var+'%(delay_u)s'
                        )

        # If there is a uniform delay, the performance can be improved by
        # pre_loading the delayed variable in advance.
        pre_copy = ""
        if proj.max_delay > 1 and proj.uniform_delay != -1:
            for var in dependencies:
                if var in proj.pre.neuron_type.description['local']:
                    pre_copy += "std::vector<%(float_prec)s> _pre_" + var + " = %(pre_prefix)s_delayed_" + var + "%(delay_u)s;"
                    psp = psp.replace(
                        '%(pre_prefix)s_delayed_'+var+'%(delay_u)s%(pre_index)s',
                        '_pre_'+var+'%(pre_index)s'
                    )

        # Special case for diagonal format
        if proj._storage_format == "dia":
            ids.update({'omp_simd': '' if get_global_config('disable_SIMD_SpMV') else '#pragma omp simd'})

        # The hybrid format needs to be handled seperately
        # as its composed of two parts
        sum_code = ""
        if proj._storage_format != "hyb":
            # Finalize the psp with the correct ids
            psp = psp % ids

            # e. g. pre-load of pre-synaptic variables
            pre_copy = pre_copy % ids

            # add non-default template ids
            ids.update({
                'pre_copy': pre_copy,
                'psp': psp.replace(';', '')
            })

            # Generate the code depending on the operation
            sum_code = template[proj.synapse_type.operation] %  ids

        else:
            ids.update({'delay_u' : '[delay-1]'})

            # take the same indices as used normally
            # (lookup: self._configure_template_ids())
            coo_ids = deepcopy(self._template_ids)
            coo_ids.update({
                'local_index': '->coo[j]',
                'pre_index': '[*col_it]',
                'post_index': '[*row_it]'
            })
            coo_psp = psp % coo_ids

            ell_ids = deepcopy(self._template_ids)
            ell_ids.update({
                'local_index': '->ell[j]',
                'semiglobal_index': '[i]',
                'global_index': '',
                'post_index': '[rk_post]',
                'pre_index': '[rk_pre]'
            })
            ell_psp = psp % ell_ids

            ids.update({
                'pre_copy': pre_copy % ids,
                'coo_psp': coo_psp.replace(';', ''),
                'ell_psp': ell_psp.replace(';', ''),
                'id_pre': proj.pre.id,
                'id_post': proj.post.id,
                'target': proj.target,
                'ell_post_index': ell_ids['post_index'],
                'coo_post_index': coo_ids['post_index'],
            })

            sum_code = template[proj.synapse_type.operation] % ids

        # Finish the code
        final_code = """
        if (_transmission && %(post_prefix)s_active) {
%(code)s
        } // active
        """ % {'post_prefix': ids['post_prefix'], 'code': tabify(sum_code, 3)}

        if self._prof_gen:
            final_code = self._prof_gen.annotate_computesum_rate(proj, final_code)

        return psp_prefix, final_code

    def _process_g_target_code(self, proj, eq, ids):
        """
        TODO: docs!
        """
        # PSP form
        g_target = eq['cpp'].split('=')[1]
        # Operation (g_target is replaced by sum in 'cpp')
        operation = re.search(r'sum (.*?)=', eq['cpp']).group(1).strip() + "="
        # Check targets
        if isinstance(proj.target, str):
            targets = [proj.target]
        else:
            targets = proj.target

        g_target_code = ""
        for target in targets:
            # Special case where w is a single value
            if proj._has_single_weight():
                g_target = re.sub(
                    r'([^\w]+)w%\(local_index\)s',
                    r'\1w',
                    g_target
                )

            # update post-synaptic potential code
            target_dict = {
                'post_prefix': ids['post_prefix'],
                'target': target,
                'g_target': g_target % ids,
                'eq': eq['eq'],
                'post_index': ids['post_index'],
                'operation': operation
            }
            g_target_code += """
    %(post_prefix)sg_%(target)s%(post_index)s %(operation)s %(g_target)s
"""% target_dict

            # Determine bounds
            for key, val in eq['bounds'].items():
                if not key in ['min', 'max']:
                    continue
                try:
                    value = str(float(val))
                except: # TODO: more complex operations
                    value = val % ids

                g_target_code += """
    if (%(post_prefix)sg_%(target)s%(post_index)s %(op)s %(val)s)
        %(post_prefix)sg_%(target)s%(post_index)s = %(val)s;
""" % {'post_prefix': ids['post_prefix'], 'target': target, 'post_index': ids['post_index'], 'op': "<" if key == 'min' else '>', 'val': value}

        return g_target_code

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
            psp_prefix = proj._specific_template['psp_prefix']
            psp_code = proj._specific_template['psp_code']

            if len(psp_code) > 0 and self._prof_gen:
                psp_code = self._prof_gen.annotate_computesum_spiking(proj, psp_code)

            return psp_prefix, psp_code

        psp_prefix = """int nb_post; double sum;"""

        # Basic tags, dependent on storage format are assuming a feedforward
        # transmission.
        ids = deepcopy(self._template_ids)

        # The spike transmission is triggered from pre-synaptic side
        # and the indices need to be changed.
        if proj._storage_format == "lil":
            ids.update({
                'pre_index': '[rk_j]',
                'post_index': '[post_rank[i]]',
            })

        elif proj._storage_format == "csr":
            if proj._storage_order == "post_to_pre":
                ids.update({
                    'local_index': "[_inv_idx[syn]]",
                    'semiglobal_index': '[_row_idx[syn]]',
                    'pre_index': '[_pre]',
                    'post_index': '[_row_idx[syn]]',
                })
            else:
                ids.update({
                    'local_index': "[syn]",
                    'semiglobal_index': '[col_idx_[syn]]',
                    'pre_index': '[_pre]',
                    'post_index': '[col_idx_[syn]]',
                })

        elif proj._storage_format in ["bsr","dense"]:
            # nothing to do here, as the indices can simply switched
            pass

        else:
            # HD (19th May 2022):
            # many formats will need to adapt here. By raising this
            # error, I remember us that we need to check this carefully
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
        g_target_code = ""

        # Analyse all elements of pre_spike
        for eq in proj.synapse_type.description['pre_spike']:
            # g_target is treated differently
            # Must be at the end of the equations
            if eq['name'] == 'g_target':
                g_target_code += self._process_g_target_code(proj, eq, ids)

            else:
                # process equations in pre_spike which
                # are not 'g_target'

                condition = ""
                # Check conditions to update the variable
                if eq['name'] == 'w': # Surround it by the learning flag
                    condition = "_plasticity" # Plasticity can be disabled

                if 'unless_post' in eq['flags']: # Flags avoids pre-spike evaluation when post fires at the same time
                    simultaneous = "%(pre_prefix)slast_spike[%(pre_index)s] != %(post_prefix)slast_spike[%(post_index)s]" % ids
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

        # Special case where w is a single value
        if proj._has_single_weight():
            g_target_code = re.sub(
                r'([^\w]+)w%\(local_index\)s',
                r'\1w',
                g_target_code
            )

        # finalize g_target_code
        g_target_code = g_target_code % ids

        # Generate the default post-conductance increase
        # if no g_target statement is present and no continuous transmission (e.g. gap-junction)
        if not continous_transmission and len(g_target_code) == 0:
            # default g_target += w
            default_code = """
            // Increase the post-synaptic conductance g_target += w
            %(post_prefix)sg_%(target)s%(post_index)s += w%(local_index)s;
"""

            # Special case where w is a single value
            if proj._has_single_weight():
                default_code = re.sub(
                    r'([^\w]+)w%\(local_index\)s',
                    r'\1w',
                    default_code
                )

            # Check targets
            if isinstance(proj.target, str):
                targets = [proj.target]
            else:
                targets = proj.target

            # Iterate over all targets
            for target in targets:
                g_target_code += default_code % {
                    'post_prefix': ids['post_prefix'],
                    'post_index': ids['post_index'],
                    'local_index': ids['local_index'],
                    'target': target
                }

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

        # Choose correct template based on connectivity
        # and take delays into account if any
        pre_array = ""
        if proj.max_delay > 1:
            if proj.uniform_delay == -1: # Non-uniform delays
                Messages._warning('Variable delays for spiking networks is experimental and slow...')
                template = self._templates['spiking_sum_variable_delay']
            else: # Uniform delays
                pre_array = "%(pre_prefix)s_delayed_spike[delay-1]" % ids
                template = self._templates['spiking_sum_fixed_delay']
        else:
            pre_array = "%(pre_prefix)sspiked" % ids
            template = self._templates['spiking_sum_fixed_delay']

        if template == None:
            Messages._error("Code generation error: no template available")

        complete_code = ""

        # Axonal spike events
        spiked_array_fusion_code = ""
        if proj.synapse_type.pre_axon_spike:
            if proj.synapse_type.description['raw_pre_spike'] == proj.synapse_type.description['raw_axon_spike']:
                # default and axonal spike share the same mechanism,
                # therefore we can join the two spike event list
                spiked_array_fusion_code = """
        std::vector<int> tmp_spiked = %(pre_array)s;
        if (_axon_transmission) {
            tmp_spiked.insert( tmp_spiked.end(), %(pre_prefix)saxonal.begin(), %(pre_prefix)saxonal.end() );
        }
    """ % {'pre_prefix': ids['pre_prefix'], 'pre_array': pre_array}

                pre_array = "tmp_spiked"

            else:
                # axon_spike uses a different equation
                # HD (20. Oct. 2023): together with Oliver Maith we decided that the equations are limited
                #                     to g_target += ...
                ids.update({
                    'pre_array': "pop%(id_pre)s.axonal" % {'id_pre': proj.pre.id},
                    'pre_event': "",
                    'g_target': self._process_g_target_code(proj, proj.synapse_type.description['pre_axon_spike'][0], ids),
                    'target': proj.target, # for omp reduce
                    'event_driven': "",
                    'spiked_array_fusion': ""
                })
                complete_code += template % ids
                complete_code = complete_code.replace("(_transmission", "(_axon_transmission")  # TODO: quite hacky ...

                # for the default code generation path
                pre_array = "pop%(id_pre)s.spiked" % {'id_pre': proj.pre.id}

        # Generate the whole code block
        if g_target_code != "" or pre_code != "":
            ids.update({
                'pre_array': pre_array,
                'pre_event': pre_code,
                'g_target': g_target_code,
                'target': proj.target, # for omp reduce
                'event_driven': event_driven_code,
                'spiked_array_fusion': spiked_array_fusion_code
            })
            complete_code += template % ids

        # Add tabs
        complete_code = tabify(complete_code, 2)

        ####################################################
        # Not even-driven summation of psp: like rate-coded
        ####################################################
        if 'psp' in  proj.synapse_type.description.keys(): # not event-based
            # Compute it as if it were rate-coded
            _, psp_code = self._computesum_rate(proj)
            psp_prefix += " int rk_post;"
            psp_prefix = tabify(psp_prefix, 2)

            # Change _sum_target into g_target (TODO: handling of PopulationViews???)
            psp_code = psp_code.replace(
                '%(post_prefix)s_sum_%(target)s' % ids,
                '%(post_prefix)sg_%(target)s' % ids
            )

            # Add it to the main code
            complete_code += """
        // PSP-based summation"""
            complete_code += psp_code

            # HD (8th Oct. 2024): it's a bit hacky ... rate-coded and spiking models use
            #                     different identifiers for indexing the data structures
            if proj._storage_format == "dense":
                complete_code = complete_code.replace("rk_post", "i")

        # Annotate code
        if self._prof_gen:
            complete_code = self._prof_gen.annotate_computesum_spiking(proj, complete_code)

        return psp_prefix, complete_code

    def _header_structural_plasticity(self, proj):
        """
        Generate extension code for C header_struct: variable declaration, add and remove synapses.

        Templates:

            structural_plasticity: 'header_struct' field contains all relevant code templates

        """
        header_tpl = self._templates['structural_plasticity']['header_struct']

        code = ""
        # Pruning defined in the synapse
        if 'pruning' in proj.synapse_type.description.keys():
            code += header_tpl['pruning']

        # Creating defined in the synapse
        if 'creating' in proj.synapse_type.description.keys():
            code += header_tpl['creating']

        # Retrieve the names of extra attributes
        extra_args = ""
        add_var_code = ""
        add_var_remove = ""
        for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
            if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse_type.description['local']:

                if not isinstance(proj.init[var['name']], (int, float, bool)):
                    init = var['init']
                else:
                    init = proj.init[var['name']]
                extra_args += ', ' + var['ctype'] + ' _' +  var['name'] +'='+str(init)
                add_var_code += ' '*8 + var['name'] + '[post].insert('+var['name']+'[post].begin() + idx, _' + var['name'] + ');\n'
                add_var_remove += ' '*8 + var['name'] + '[post].erase(' + var['name'] + '[post].begin() + idx);\n'

        # Delays
        delay_code = ""
        delay_remove= ""
        if proj.max_delay > 1 and proj.uniform_delay == -1:
            delay_code = ' '*8 + "delay[post].insert(delay[post].begin() + idx, _delay);"
            delay_remove = ' '*8 + "delay[post].erase(delay[post].begin() + idx);"

        # Spiking networks must update the inv_pre_rank array
        spiking_addcode = "" if proj.synapse_type.type == 'rate' else header_tpl['spiking_addcode']
        spiking_removecode = "" if proj.synapse_type.type == 'rate' else header_tpl['spiking_removecode']

        # Random distributions
        rd_addcode = ""
        rd_removecode = ""
        for rd in proj.synapse_type.description['random_distributions']:
            rd_addcode += """
        %(name)s[post].insert(%(name)s[post].begin() + idx, 0.0);
""" % {'name': rd['name']}

            rd_removecode += """
        %(name)s[post].erase(%(name)s[post].begin() + idx);
""" % {'name': rd['name']}

        # For spiking models, we need to rebuild the backward view, if synapses are removed/added
        inverse_connectivity_code = '' if proj.synapse_type.type == 'rate' else header_tpl['spiking_rebuild_backwardview']

        # Generate the code
        code += header_tpl['header'] % {
            'extra_args': extra_args,
            'delay_code': delay_code, 'delay_remove': delay_remove,
            'add_code': add_var_code, 'add_remove': add_var_remove,
            'spike_add': spiking_addcode, 'spike_remove': spiking_removecode,
            'rd_add': rd_addcode, 'rd_remove': rd_removecode,
            'inverse_connectivity_rebuild': inverse_connectivity_code
        }

        return code

    def _init_random_distributions(self, proj):
        # Is it a specific population?
        if 'init_rng' in proj._specific_template.keys():
            return proj._specific_template['init_rng']

        code = ""
        for rd in proj.synapse_type.description['random_distributions']:
            ids = {
                'id': proj.id,
                'float_prec': get_global_config('precision'),
                'global_index': ''
            }
            rd_init = rd['definition'] % ids
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
        """
        Generates the code for the post-synaptic updates of event-driven learning rules.
        """
        if proj.synapse_type.type == "rate":
            return ""

        if proj.synapse_type.description['post_spike'] == []:
            return ""

        # Get basic template ids
        ids = deepcopy(self._template_ids)

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

        # Generate the code block
        try:
            ids.update({
                'post_event': post_code,
                'event_driven': event_driven_code,
            })
            code = self._templates['post_event'] % ids
        except KeyError:
            # Template does not exist
            raise KeyError("No template for spiking neurons post event (format = " + proj._storage_format + " and order = " + proj._storage_order+ ")")

        # Annotate code
        if self._prof_gen:
            code = self._prof_gen.annotate_post_event(proj, code)

        return tabify(code, 2)

    def _update_random_distributions(self, proj):
        """
        Step-wise update of random distributed variables which may appear as local (per synapse),
        semiglobal (per dendrite) or global (one value per projection).
        """
        # Is it a specific projection?
        if 'update_rng' in proj._specific_template.keys():
            return proj._specific_template['update_rng']

        if len(proj.synapse_type.description['random_distributions']) == 0:
            return ""

        global_code=""
        semiglobal_code=""
        local_code=""

        for rd in proj.synapse_type.description['random_distributions']:
            if rd['name'] in proj.synapse_type.description["local"]:
                local_code += self._templates["rng_update"][rd["locality"]] % {'rd_name': rd['name']}

        return tabify(self._templates["rng_update"]["template"] % {
            'global_rng': global_code,
            'semiglobal_rng': semiglobal_code,
            'local_rng': local_code
        }, 2)

    def _update_synapse(self, proj):
        """
        Generates the code for the continuous update of synaptic variables of the given projection *proj*.
        """
        prefix = """
        %(idx_type)s rk_post, rk_pre;
        %(float_prec)s _dt = dt * _update_period;""" % {'idx_type': determine_idx_type_for_projection(proj)[0], 'float_prec': get_global_config('precision')}

        # Global variables
        global_eq = generate_equation_code(proj.synapse_type.description, 'global', 'proj', padding=2, wrap_w="_plasticity")

        # Semiglobal variables
        semiglobal_eq = generate_equation_code(proj.synapse_type.description, 'semiglobal', 'proj', padding=2, wrap_w="_plasticity")

        # Local variables
        loc_eq_pad = 3 if not proj._storage_format=="dense" else 4
        local_eq = generate_equation_code(proj.synapse_type.description, 'local', 'proj', padding=loc_eq_pad, wrap_w="_plasticity")

        # Skip generation if there are no equations
        if local_eq.strip() == '' and semiglobal_eq.strip() == '' and global_eq.strip() == '':
            return "", ""

        # Gather pre-loop declaration (dt/tau for ODEs)
        pre_code = ""
        for var in proj.synapse_type.description['variables']:
            if 'pre_loop' in var.keys() and len(var['pre_loop']) > 0:
                pre_code += var['ctype'] + ' ' + var['pre_loop']['name'] + ' = ' + var['pre_loop']['value'] + ';\n'

        if len(pre_code) > 0:
            pre_code = """
    // Updating the step sizes
""" + tabify(pre_code, 1)
            global_eq = pre_code + global_eq

        # adjust dt dependent on the _update_period, this covers only
        # the switch statements
        global_eq = re.sub(
            r'([^\w]+)dt([^\w]+)',
            r'\1_dt\2',
            global_eq
        )
        semiglobal_eq = re.sub(
            r'([^\w]+)dt([^\w]+)',
            r'\1_dt\2',
            semiglobal_eq
        )
        local_eq = re.sub(
            r'([^\w]+)dt([^\w]+)',
            r'\1_dt\2',
            local_eq
        )

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
        try:
            template = self._templates['update_variables']

        except KeyError:
            # either no template code at all, or no 'update_variables' field.
            Messages._error("No synaptic plasticity template found for format = " + proj._storage_format, " and order = " + proj._storage_order)

        template_ids = deepcopy(self._template_ids) # will be extended at the end of this function
        template_ids.update({
            'global': global_eq % self._template_ids,
            'semiglobal': semiglobal_eq % self._template_ids,
            'local': local_eq % self._template_ids,
        })

        # Fill the code template
        if local_eq.strip() != "": # local synapses are updated
            code = template['local'] % template_ids
        else: # Only global variables
            code = template['global'] % template_ids

        if self._prof_gen:
            code = self._prof_gen.annotate_update_synapse(proj, code)

        # Return the code block
        return prefix, tabify(code, 2)

    def _update_max_delay(self, proj):
        "When the maximum delay of a non-uniform spiking projection changes, the ring buffer for delyed spikes must be updated."

        if proj.synapse_type.type == 'rate':
            return "", ""

        if proj.uniform_delay >= 0:
            return "", ""

        update_delay_code = """
        // No need to do anything if the new max delay is smaller than the old one
        if(d <= max_delay)
            return;

        // Update delays
        int prev_max = max_delay;
        max_delay = d;
        int add_steps = d - prev_max;

        // std::cout << "Delayed arrays was " << _delayed_spikes.size() << std::endl;

        // Insert as many empty vectors as need at the current pointer position
        _delayed_spikes.insert(_delayed_spikes.begin() + idx_delay, add_steps, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >() ));

        // The delay index has to be updated
        idx_delay = (idx_delay + add_steps) % max_delay;

        // std::cout << "Delayed arrays is now " << _delayed_spikes.size() << std::endl;
        // std::cout << "Idx " << idx_delay << std::endl;
        // for(int i = 0; i< max_delay; i++)
        //     std::cout << _delayed_spikes[i][0].size() << std::endl;
"""

        reset_ring_buffer_code = self._templates['delay']['nonuniform_spiking']['reset'] % self._template_ids

        return update_delay_code, reset_ring_buffer_code
