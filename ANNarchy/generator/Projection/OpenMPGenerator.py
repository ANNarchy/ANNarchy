"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import ANNarchy

# ANNarchy objects
from ANNarchy.core import Global
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.models.Synapses import DefaultRateCodedSynapse
from ANNarchy.intern.ConfigManagement import get_global_config, _check_precision
from ANNarchy.intern import Messages

# Code templates
from ANNarchy.generator.Projection.ProjectionGenerator import ProjectionGenerator, get_bounds
from ANNarchy.generator.Projection.OpenMP import *

# Useful functions
from ANNarchy.generator.Utils import generate_equation_code, tabify, remove_trailing_spaces, check_avx_instructions, determine_idx_type_for_projection

import re
from copy import deepcopy

class OpenMPGenerator(ProjectionGenerator):
    """
    Generate the header for a Population object to run either on single core
    or multi-cores with OpenMP.
    """
    def __init__(self, profile_generator, net_id):
        # The super here calls all the base classes, so first
        # ProjectionGenerator and afterwards OpenMPConnectivity
        # TODO: this is python 2 syntax
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
        # Initial state
        self._templates = deepcopy(BaseTemplates.openmp_templates)
        self._template_ids = {}

        # Select the C++ connectivity template
        sparse_matrix_include, sparse_matrix_format, sparse_matrix_args, single_matrix = self._select_sparse_matrix_format(proj)

        # Update template fill elements
        self._configure_template_ids(proj, single_matrix)

        # Generate declarations and accessors for the variables
        decl, accessor = self._declaration_accessors(proj, single_matrix)

        # Initiliaze the projection
        init_weights, init_delays, init_parameters_variables = self._init_parameters_variables(proj, single_matrix)

        # Synaptic plasticity
        update_prefix, update_variables = self._update_synapse(proj, single_matrix)

        # Update the random distributions
        init_rng = self._init_random_distributions(proj)
        update_rng = self._update_random_distributions(proj)

        post_event_prefix, post_event = self._post_event(proj, single_matrix)

        # Compute sum is the trickiest part
        if proj.synapse_type.type == 'rate':
            psp_prefix, psp_code = self._computesum_rate(proj, single_matrix)
        else:
            psp_prefix, psp_code = self._computesum_spiking(proj, single_matrix)

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

        # Some Connectivity implementations requires the number of threads in constructor
        if get_global_config('num_threads') > 1:
            if proj._storage_format == "lil":
                if single_matrix or proj._no_split_matrix:
                    num_threads_acc = ""
                else:
                    num_threads_acc = ", global_num_threads"
            elif proj._storage_format == "csr":
                if proj._no_split_matrix or single_matrix:
                    num_threads_acc = ""
                else:
                    num_threads_acc = ", global_num_threads"
            else:
                num_threads_acc = ""
        else:
            num_threads_acc = ""

        # Connectivity template
        if 'declare_connectivity_matrix' not in proj._specific_template.keys():
            connector_call = self._connectivity_init(proj, sparse_matrix_format, sparse_matrix_args) % {
                'sparse_format': sparse_matrix_format,
                'init_weights': init_weights,
                'init_delays': init_delays,
                'rng_idx': "[0]" if single_matrix else "",
                'add_args': "",
                'num_threads': num_threads_acc,
                'float_prec': get_global_config('precision'),
                'idx_type': self._template_ids['idx_type']
            }
            declare_connectivity_matrix = ""
            access_connectivity_matrix = ""
        else:
            sparse_matrix_format = "SpecificConnectivity"
            sparse_matrix_args = ""
            sparse_matrix_include = "#include \"Specific.hpp\"\n"
            connector_call = ""
            declare_connectivity_matrix = proj._specific_template['declare_connectivity_matrix']
            access_connectivity_matrix = proj._specific_template['access_connectivity_matrix']

        # local functions
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

        # Sparse matrix format specific
        # HD (20th May 2022): this is probably not the best way to do it ...
        declare_additional=decl['additional']
        init_additional = ""
        if proj._storage_format == "dense" and proj._storage_order == "pre_to_post":
            declare_additional += """\t// dense matrix - static schedule
    std::vector<int> post_slices_;
        """
            init_additional += """\t// static distribution across threads
        int chunk_size = static_cast<int>(ceil(static_cast<double>(this->num_rows_) / static_cast<double>(global_num_threads)));
        post_slices_ = std::vector<int>(1, 0);
        for (int t = 1; t <= global_num_threads; t++)
            post_slices_.push_back(std::min<int>(t*chunk_size, this->num_rows_));
"""

        # Additional info (overwritten)
        include_additional = ""
        struct_additional = ""
        access_additional = ""
        if 'include_additional' in proj._specific_template.keys():
            include_additional = proj._specific_template['include_additional']
        if 'struct_additional' in proj._specific_template.keys():
            struct_additional = proj._specific_template['struct_additional']
        if 'init_additional' in proj._specific_template.keys():
            init_additional = proj._specific_template['init_additional']
        if 'access_additional' in proj._specific_template.keys():
            access_additional = proj._specific_template['access_additional']

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
            'include_additional': include_additional,
            'include_profile': include_profile,
            'struct_additional': struct_additional,
            'declare_connectivity_matrix': declare_connectivity_matrix,
            'access_connectivity_matrix': access_connectivity_matrix,
            'declare_delays': decl['declare_delay'] if has_delay else "",
            'declare_event_driven': decl['event_driven'] if has_event_driven else "",
            'declare_rng': decl['rng'],
            'declare_parameters_variables': decl['parameters_variables'],
            'declare_additional': declare_additional,
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
            'post_event_prefix': post_event_prefix,
            'post_event': post_event,
            'access_parameters_variables': accessor,
            'access_additional': access_additional,
            'size_in_bytes': size_in_bytes,
            'clear_container': clear_container,
            'sparse_format': sparse_matrix_format,
            'sparse_format_args': sparse_matrix_args,
            'float_prec': get_global_config('precision'),
            'creating': creating,
            'pruning': pruning
        }

        final_code = self._templates['projection_header'] % final_code_dict

        # remove right-trailing white spaces
        final_code = remove_trailing_spaces(final_code)

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

        proj_desc['compute_psp'] = """\tproj%(id)s.compute_psp(tid, nt);\n""" % {'id' : proj.id}
        proj_desc['update'] = "" if update_variables == "" else """\tproj%(id)s.update_synapse(tid, nt);\n""" % {'id': proj.id}
        proj_desc['rng_update'] = "" if update_rng == "" else """\tproj%(id)s.update_rng();\n""" % {'id': proj.id}
        proj_desc['post_event'] = "" if post_event == "" else """\tproj%(id)s.post_event(tid, nt);\n""" % {'id': proj.id}

        return proj_desc

    def _configure_template_ids(self, proj, single_matrix):
        """
        Assign the correct template dictionary based on projection
        storage format. Also sets the basic template ids, e. g. indices
        """
        # Index data types depend on the matrix dimension
        # HD (1st Dec. 2021):   until now, the data type optimization is disabled for
        #                       spiking models. I want to adjust new codes already as I
        #                       hope to update the spike code generation soon ...
        idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)

        self._template_ids.update({
            'id_proj' : proj.id,
            'target': proj.target,
            'id_post': proj.post.id,
            'id_pre': proj.pre.id,
            'float_prec': get_global_config('precision'),
            'pre_prefix': 'pop'+ str(proj.pre.id) + '.',
            'post_prefix': 'pop'+ str(proj.post.id) + '.',
            'idx_type': idx_type,
            'size_type': size_type
        })

        if proj._storage_format == "lil":
            if proj.synapse_type.type == "rate":
                # Rate-coded models LIL
                if single_matrix:
                    self._templates.update(LIL_OpenMP.conn_templates)
                    self._template_ids.update(LIL_OpenMP.conn_ids)
                else:
                    self._templates.update(LIL_Sliced_OpenMP.conn_templates)
                    self._template_ids.update(LIL_Sliced_OpenMP.conn_ids)

            else:
                # Spiking models LIL
                if proj._storage_order == "pre_to_post":
                    raise NotImplementedError

                if single_matrix:
                    self._templates.update(LIL_OpenMP.conn_templates)
                    self._template_ids.update(LIL_OpenMP.conn_ids)
                else:
                    self._templates.update(LIL_Sliced_OpenMP.conn_templates)
                    self._template_ids.update(LIL_Sliced_OpenMP.conn_ids)

        elif proj._storage_format == "bsr":
            if proj.synapse_type.type == "rate":
                if single_matrix:
                    self._templates.update(BSR_OpenMP.conn_templates)
                    self._template_ids.update(BSR_OpenMP.conn_ids)
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

        elif proj._storage_format == "csr":
            if proj.synapse_type.type == "rate":
                self._templates.update(CSR_OpenMP.conn_templates)
                self._template_ids.update(CSR_OpenMP.conn_ids)

            else:
                # Spiking models (CSRC)
                if proj._storage_order == "post_to_pre":
                    if single_matrix:
                        self._templates.update(CSR_OpenMP.conn_templates)
                        self._template_ids.update(CSR_OpenMP.conn_ids)
                    else:
                        self._templates.update(CSR_Sliced_OpenMP.conn_templates)
                        self._template_ids.update(CSR_Sliced_OpenMP.conn_ids)

                else:
                    if single_matrix:
                        self._templates.update(CSR_T_OpenMP.conn_templates)
                        self._template_ids.update(CSR_T_OpenMP.conn_ids)
                    else:
                        self._templates.update(CSR_T_Sliced_OpenMP.conn_templates)
                        self._template_ids.update(CSR_T_Sliced_OpenMP.conn_ids)

        elif proj._storage_format == "coo":
            if proj.synapse_type.type == "rate":
                # Rate-coded models coordinate format
                if single_matrix:
                    self._templates.update(COO_OpenMP.conn_templates)
                    self._template_ids.update(COO_OpenMP.conn_ids)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        elif proj._storage_format == "dia":
            if proj.synapse_type.type == "rate":
                # Rate-coded models coordinate format
                if single_matrix:
                    self._templates.update(DIA_OpenMP.conn_templates)
                    self._template_ids.update(DIA_OpenMP.conn_ids)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        elif proj._storage_format == "ellr":
            if proj.synapse_type.type == "rate":
                # Rate-coded models ELLPACK-R format
                if single_matrix:
                    self._templates.update(ELLR_OpenMP.conn_templates)
                    self._template_ids.update(ELLR_OpenMP.conn_ids)
                else:
                    raise NotImplementedError

            else:
                raise Global.InvalidConfiguration("    "+proj.name+": ELLPACK-R format is not available for spiking models.")

        elif proj._storage_format == "sell":
            if proj.synapse_type.type == "rate":
                # Rate-coded models sliced ELLPACK-R format
                if single_matrix:
                    self._templates.update(SELL_OpenMP.conn_templates)
                    self._template_ids.update(SELL_OpenMP.conn_ids)
                else:
                    raise NotImplementedError

            else:
                raise Global.InvalidConfiguration("    "+proj.name+": sliced ELLPACK format is not available for spiking models.")

        elif proj._storage_format == "ell":
            if proj.synapse_type.type == "rate":
                # Rate-coded models ELLPACK format
                if single_matrix:
                    self._templates.update(ELL_OpenMP.conn_templates)
                    self._template_ids.update(ELL_OpenMP.conn_ids)
                else:
                    raise NotImplementedError

            else:
                raise Global.InvalidConfiguration("    "+proj.name+": ELLPACK format is not available for spiking models.")

        elif proj._storage_format == "dense":
            if proj._storage_order == "post_to_pre":
                self._templates.update(Dense_OpenMP.conn_templates)
                self._template_ids.update(Dense_OpenMP.conn_ids)
            else:
                self._templates.update(Dense_T_OpenMP.conn_templates)
                self._template_ids.update(Dense_T_OpenMP.conn_ids)

        else:
            raise NotImplementedError

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
        creation_ids.update({
            'pre_index': '[rk_pre]',
            'post_index': '[rk_post]'
        })
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

    def _computesum_rate(self, proj, single_matrix):
        """
        Create the c++ code for post-synaptic potential computation. 

        Parameters:

        * proj:             Instance of ANNarchy.core.Projection
        * single_matrix:    Is the basic connectivity a complete matrix or sliced
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

        # Use specialized code templates?
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

            # Does our current system support AVX?
            if simd_type is not None:

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
                            'get_r': ids['pre_prefix']+"r.data()"
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
                        unrolled_template = template = self._templates["unrolled_default_psp"][blockDim]

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
                            pass

                    except KeyError:
                        # No fitting code found, so we fall back to normal code generation
                        # TODO: add internal error log, which key was missing?
                        Messages._debug("No SIMD implementation found, fallback to non-SIMD code")
                        template = ""

        # Choose the relevant summation template
        try:
            template = self._templates['rate_coded_sum']
        except:
            Messages._error("OpenMPGenerator: no template for storage_format = "+proj._storage_format+" configuration available")

        # Dictionary of keywords to transform the parsed equations
        ids = self._template_ids

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

        # OpenMP run modifiers
        schedule = "" if not 'psp_schedule' in proj._omp_config.keys() else proj._omp_config['psp_schedule']

        # Some formats have additional thread-local parameters
        if proj._storage_format=='lil':
            add_clause = 'private(sum)'
        elif proj._storage_format in ['ell', 'ellr']:
            add_clause = 'private(sum) firstprivate(maxnzr_)'
        else:
            add_clause = ''

        # In case of a uniform delay we preload the variable and then provide as
        # thread local argument to the worker threads
        pre_copy = ""
        first_privates = ""
        if proj.max_delay > 1 and proj.uniform_delay != -1:

            for var in dependencies:
                if var in proj.pre.neuron_type.description['local']:
                    pre_copy += "std::vector<%(float_prec)s> _pre_" + var + " = %(pre_prefix)s_delayed_" + var + "%(delay_u)s;"
                    psp = psp.replace(
                        '%(pre_prefix)s_delayed_'+var+'%(delay_u)s%(pre_index)s',
                        '_pre_'+var+'%(pre_index)s'
                    )
                    first_privates += '_pre_%(var)s, ' % {'var': var}

        # Special case for diagonal format
        if proj._storage_format == "dia":
            ids.update({'omp_simd': '#pragma omp for' if get_global_config('disable_SIMD_SpMV') else '#pragma omp for simd'})

        # Finalize the psp with the correct ids
        psp = psp % ids
        pre_copy = pre_copy % ids

        ids.update({
            'pre_copy': pre_copy,
            'omp_code': '#pragma omp for',
            'omp_clause': add_clause,
            'omp_schedule': schedule,
            'psp': psp.replace(';', ''),
            'simd_len': str(4) if _check_precision('double') else str(8),
        })
        # Generate the code depending on the operation
        sum_code = template[proj.synapse_type.operation] % ids

        # Default variables needed in psp_code
        if proj._storage_format == "lil":
            psp_prefix = """
        %(idx_type)s nb_post, nb_pre;
        %(float_prec)s sum;""" % ids
        elif proj._storage_format == "ellr":
            psp_prefix = """
        %(idx_type)s rk_post, rk_pre;
        %(float_prec)s sum;""" % ids
        else:
            psp_prefix = ""

        # Finish the code
        final_code = """
        if (_transmission && %(post_prefix)s_active){
%(code)s
        } // active
        """ % {
            'post_prefix': ids['post_prefix'],
            'code': tabify(sum_code, 3),
        }

        if self._prof_gen:
            final_code = self._prof_gen.annotate_computesum_rate(proj, final_code)

        return psp_prefix, final_code

    def _computesum_spiking(self, proj, single_matrix):
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

        # If the connectivity is stored as post_to_pre, we need to use the
        # inversed matrix view to acces the psp/weight vectors. As we
        # parallelize over pre-neurons, there is a chance of concurrent accesses
        # towards psp.
        #
        # Early implementations used atomics to protect, a clear performance
        # limiter. The user ilyasm proposed a solution using shared arrays and
        # a following reduction. Here we initialize the thread local array.
        if proj._storage_order == "post_to_pre":
            psp_prefix = ""
            if not proj.disable_omp: # TODO: are there other conditions?
                psp_prefix += """
#ifdef _OPENMP"""
                targets = [proj.target] if type(proj.target) == str else proj.target
                for target in targets:
                    psp_prefix += """
        std::vector< double > pop%(id)s_%(target)s_thr(pop%(id)s.get_size()*global_num_threads, 0.0);""" % { 'id': proj.post.id, 'target': target }
                psp_prefix += """
#endif
"""

            psp_prefix += """
        int nb_post;
        double sum;"""
        else:
            psp_prefix = """
        int nb_post;
        double sum;"""

        # Basic tags, dependent on storage format are assuming a feedforward
        # transmission.
        ids = deepcopy(self._template_ids)

        # The psp kernel sometimes use diverging indices
        # HD (10th April 2022): maybe I should remove this in future (TODO)
        if proj._storage_format == "lil":
            if get_global_config('num_threads') == 1 or single_matrix:
                ids.update({
                    'pre_index': '[rk_j]',
                })
            else:
                ids.update({
                    'pre_index': '[rk_j]',
                })

        elif proj._storage_format == "csr":
            if proj._storage_order == "post_to_pre":
                if single_matrix:
                    ids.update({
                        'local_index': "[_inv_idx[syn]]",
                        'semiglobal_index': '[_row_idx[syn]]',
                        'global_index': '',
                        'pre_index': '[_pre]',
                        'post_index': '[_row_idx[syn]]',
                    })
                else:
                    ids.update({
                        'local_index': "[tid][_inv_idx[syn]]",
                        'semiglobal_index': '[_row_idx[syn]]',
                        'global_index': '',
                        'pre_index': '[_pre]',
                        'post_index': '[_row_idx[syn]]',
                    })

            else:
                if single_matrix:
                    ids.update({
                        'local_index': "[syn]",
                        'semiglobal_index': '[col_idx_[syn]]',
                        'global_index': '',
                        'pre_index': '[_pre]',
                        'post_index': '[col_idx_[syn]]',
                    })

                else:
                    ids.update({
                        'local_index': "[tid][syn]",
                        'semiglobal_index': '[col_idx_[syn]]',
                        'global_index': '',
                        'pre_index': '[_pre]',
                        'post_index': '[col_idx_[syn]]',
                    })

        elif proj._storage_format == "dense":
            if single_matrix:
                # Nothing to do
                pass
            else:
                raise NotImplementedError

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

                    target_dict = {
                        'post_prefix': ids['post_prefix'],
                        'target': target,
                        'g_target': g_target % ids,
                        'eq': eq['eq'],
                        'post_index': ids['post_index'],
                        'operation': operation
                    }

                    # access to post variable migth require atomic
                    # operation ( added later if needed )
                    if proj.max_delay > 1 and proj.uniform_delay == -1: # TODO: openMP is switched off for non uniform delays
                        g_target_code += """
            %(post_prefix)sg_%(target)s%(post_index)s %(operation)s %(g_target)s
"""% target_dict
                    elif proj.disable_omp or get_global_config('num_threads') == 1:
                        g_target_code += """
            %(post_prefix)sg_%(target)s%(post_index)s %(operation)s %(g_target)s
"""% target_dict
                    else:
                        raise NotImplementedError

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

            else:
                # process equations in pre_spike which
                # are not 'g_target'

                condition = ""
                # Check conditions to update the variable
                if eq['name'] == 'w': # Surround it by the learning flag
                    condition = "_plasticity" # Plasticity can be disabled

                if 'unless_post' in eq['flags']: # Flags avoids pre-spike evaluation when post fires at the same time
                    simultaneous = "%(pre_prefix)slast_spike[pre_rank[i][j]] != %(post_prefix)slast_spike[post_rank[i]]" % {'id_post': proj.post.id, 'id_pre': proj.pre.id}
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
            %(post_prefix)sg_%(target)s%(post_index)s += w%(local_index)s;
"""

        # Special case where w is a single value
        if proj._has_single_weight():
            g_target_code = re.sub(
                r'([^\w]+)w%\(local_index\)s',
                r'\1w',
                g_target_code
            )

        # finalize g_target_code
        g_target_code = g_target_code % ids

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
                template = self._templates['spiking_sum_fixed_delay'][proj._parallel_pattern]
                pre_array = "%(pre_prefix)s_delayed_spike[delay-1]" % ids
        else:
            pre_array = "%(pre_prefix)sspiked" % ids
            template = self._templates['spiking_sum_fixed_delay'][proj._parallel_pattern]

        # sanity check
        if template == None:
            raise Messages.CodeGeneratorException("\tproj{}: no template available (Configuration: format={}, order={}, single_matrix={}, pattern={})".format(proj.id, proj._storage_format, proj._storage_order, single_matrix, proj._parallel_pattern))

        # Axonal spike events
        spiked_array_fusion_code = ""
        if proj.synapse_type.pre_axon_spike:
            spiked_array_fusion_code = """
    std::vector<int> tmp_spiked = %(pre_array)s;
    if (_axon_transmission) {
        tmp_spiked.insert( tmp_spiked.end(), %(pre_prefix)saxonal.begin(), %(pre_prefix)saxonal.end() );
    }
""" % {'pre_prefix': ids['pre_prefix'], 'pre_array': pre_array}

            pre_array = "tmp_spiked"

        # Generate the whole code block
        code = ""
        if g_target_code != "" or pre_code != "":
            ids.update({
                'pre_array': pre_array,
                'pre_event': pre_code,
                'g_target': g_target_code,
                'event_driven': event_driven_code,
                'spiked_array_fusion': spiked_array_fusion_code
            })
            code = template % ids

        # Add tabs
        code = tabify(code, 2)

        ####################################################
        # Not even-driven summation of psp: like rate-coded
        ####################################################
        if 'psp' in  proj.synapse_type.description.keys(): # not event-based
            # Compute it as if it were rate-coded
            _, psp_code = self._computesum_rate(proj, single_matrix)
            
            psp_prefix = tabify("%(float_prec)s sum; int nb_pre, nb_post;" % {'float_prec': get_global_config('precision')}, 2)

            # Change _sum_target into g_target (TODO: handling of PopulationViews???)
            psp_code = psp_code.replace(
                '%(post_prefix)s_sum_%(target)s' % ids,
                '%(post_prefix)sg_%(target)s' % ids
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
            rd_init = rd['definition'] % self._template_ids
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

    def _post_event(self, proj, single_matrix):
        """
        Generates the code for the post-synaptic updates of event-driven learning rules.
        """
        if proj.synapse_type.type == "rate":
            return "", ""

        if proj.synapse_type.description['post_spike'] == []:
            return "", ""

        # Get basic template ids and update if necessary.
        ids = deepcopy(self._template_ids)
        if proj._storage_format == "lil":
            ids.update({
                'post_index': '[rk_post]',
            })

        elif proj._storage_format == "csr":
            if proj._storage_order == "post_to_pre":
                if single_matrix:
                    ids.update({
                        'semiglobal_index': '[*it]',
                        'pre_index': '[_col_idx[j]]',
                        'post_index': '[rk_post]',
                    })
                else:
                    ids.update({
                        'local_index': "[tid][j]",
                        'semiglobal_index': '[tid][*it]',
                        'pre_index': '[_col_idx[j]]',
                        'post_index': '[rk_post]',
                    })

            else:
                if single_matrix:
                    ids.update({
                        'local_index': "[inv_idx_[j]]",
                        'semiglobal_index': '[*it]',
                        'global_index': '',
                        'pre_index': '[row_idx_[j]]',
                        'post_index': '[rk_post]',
                    })
                else:
                    ids.update({
                        'local_index': "[tid][inv_idx_[j]]",
                        'semiglobal_index': "[tid][*it]",
                        'global_index': "",
                        'pre_index': "[row_idx_[j]]",
                        'post_index': "[rk_post]",
                    })

        elif proj._storage_format == "dense":
            if proj._storage_order == "post_to_pre":
                if single_matrix:
                    pass    # currently working on
                else:
                    raise NotImplementedError
            else:
                if single_matrix:
                    pass    # currently working on
                else:
                    raise NotImplementedError

        else:
            # Sanity: this should not reached
            raise NotImplementedError

        # Definition of format-specific local variables.
        if proj._storage_format == "lil":
            post_event_prefix = ""
        elif proj._storage_format == "csr":
            post_event_prefix = """
        int rk_post;
        std::vector<int>::iterator it;
        """
        elif proj._storage_format == "dense":
            post_event_prefix = """
        int rk_post;
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

        # Generate the code block
        try:
            ids.update({
                'post_event': post_code,
                'event_driven': event_driven_code
            })
            code = self._templates['post_event'] % ids

        except KeyError:
            # Template does not exist
            Messages._error("No template for spiking neurons post event (format = {}, order = {}, and single_matrix = {})".format(proj._storage_format, proj._storage_order, single_matrix))

        # Annotate code
        if self._prof_gen:
            code = self._prof_gen.annotate_post_event(proj, code)

        return post_event_prefix, tabify(code, 2)

    def _update_random_distributions(self, proj):
        """
        Step-wise update of random distributed variables which may appear as local (per synapse),
        semiglobal (per dendrite) or global (one value per projection).

        TODO: implement parallel RNG (currently one thread is responsible for draw the random numbers)
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
            'local_rng': local_code,
            'idx_type': self._template_ids['idx_type']
        }, 2)


    def _update_synapse(self, proj, single_matrix):
        """Updates the continuously changed variables of the projection."""

        # Global variables
        global_eq = generate_equation_code(proj.synapse_type.description, 'global', 'proj', padding=2, wrap_w="_plasticity")

        # Code layout
        off = 1 if not single_matrix else 0 # fix tabs for sliced matrix
        off = 1 if proj._storage_format=="dense" else 0 # fix tabs for dense matrix

        # Semiglobal variables
        semiglobal_eq = generate_equation_code(proj.synapse_type.description, 'semiglobal', 'proj', padding=2+off, wrap_w="_plasticity")

        # Local variables
        local_eq = generate_equation_code(proj.synapse_type.description, 'local', 'proj', padding=3+off, wrap_w="_plasticity")

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
            Messages._error("No synaptic plasticity template found for format = " + proj._storage_format, ", order = " + proj._storage_order, " and single_matrix =", single_matrix)

        ids = deepcopy(self._template_ids)

        # Fill the code template
        if local_eq.strip() != "": # local synapses are updated
            ids.update({
                'global': global_eq % self._template_ids,
                'semiglobal': semiglobal_eq % self._template_ids,
                'local': local_eq % self._template_ids,
            })
            code = template['local'] % ids
        else: # Only global variables
            ids.update({
                'global': global_eq % self._template_ids,
                'semiglobal': semiglobal_eq % self._template_ids,
            })
            code = template['global'] % ids

        psp_prefix = """
        %(idx_type)s rk_post, rk_pre;
        %(float_prec)s _dt = dt * _update_period;""" % ids

        if self._prof_gen:
            code = self._prof_gen.annotate_update_synapse(proj, code)

        # Return the code block
        return psp_prefix, tabify(code, 2)

    def _update_max_delay(self, proj):
        "When the maximum delay of a non-uniform spiking projection changes, the ring buffer for delyed spikes must be updated."

        if proj.synapse_type.type == 'rate':
            return "", ""

        if proj.uniform_delay >= 0:
            return "", ""

        if proj._no_split_matrix:
            update_delay_code = """
        // No need to do anything if the new max delay is smaller than the old one
        if(d <= max_delay)
            return;

        // Update delays
        int prev_max = max_delay;
        max_delay = d;
        int add_steps = d - prev_max;

    #ifdef _DEBUG
        std::cout << "Delayed arrays was " << _delayed_spikes.size() << std::endl;
    #endif

        // Insert as many empty vectors as need at the current pointer position
        _delayed_spikes.insert(_delayed_spikes.begin() + idx_delay, add_steps, std::vector< std::vector< int > >(post_rank.size(), std::vector< int >() ));

        // The delay index has to be updated
        idx_delay = (idx_delay + add_steps) % max_delay;
"""
        else:
            update_delay_code = """
        // No need to do anything if the new max delay is smaller than the old one
        if(d <= max_delay)
            return;

        // Update delays
        int prev_max = max_delay;
        max_delay = d;
        int add_steps = d - prev_max;

    #ifdef _DEBUG
        std::cout << "Delayed arrays was " << std::endl;
        for (int tid = 0; tid < global_num_threads; tid++)
            std::cout << _delayed_spikes.size() << " for thread " << tid << std::endl;
    #endif

        // Insert as many empty vectors as need at the current pointer position
        for (int tid = 0; tid < global_num_threads; tid++) {
            _delayed_spikes[tid].insert(_delayed_spikes[tid].begin() + idx_delay, add_steps, std::vector< std::vector< int > >(sub_matrices_[tid]->post_rank.size(), std::vector< int >() ));
        }

        // The delay index has to be updated
        idx_delay = (idx_delay + add_steps) % max_delay;

    #ifdef _DEBUG
        std::cout << "Delayed arrays is now " << _delayed_spikes.size() << std::endl;
        std::cout << "Idx " << idx_delay << std::endl;
        for(int i = 0; i< max_delay; i++)
            std::cout << _delayed_spikes[i][0].size() << std::endl;
    #endif
"""

        reset_ring_buffer_code = self._templates['delay']['nonuniform_spiking']['reset'] % self._template_ids

        return update_delay_code, reset_ring_buffer_code
