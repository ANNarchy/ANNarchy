#===============================================================================
#
#     ProjectionGenerator.py
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
from ANNarchy.core import Global
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.core import Random as ANNRandom
from ANNarchy.extensions.convolution import Transpose

# Useful functions
from ANNarchy.generator.Utils import tabify, determine_idx_type_for_projection, cpp_connector_available

class ProjectionGenerator(object):
    """
    Abstract definition of a ProjectionGenerator.
    """
    def __init__(self, profile_generator, net_id):
        """
        Initialization of the class object and store some ids.
        """
        super(ProjectionGenerator, self).__init__()

        self._prof_gen = profile_generator
        self._net_id = net_id

        self._templates = {}
        self._template_ids = {}
        self._connectivity_class = None

    def header_struct(self, proj, annarchy_dir):
        """
        Generate and store the projection code in a single header file. The
        name is defined as
        proj%(id)s.hpp.

        Parameters:

            proj: Projection object
            annarchy_dir: working directory

        Returns:

            (str, str): include directive, pointer definition

        Templates:

            header_struct: basic template

        """
        raise NotImplementedError

    def creating(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def pruning(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _computesum_rate(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _computesum_spiking(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _configure_template_ids(self, proj):
        """
        This function should be called before any other method of the
        Connectivity class is called, as the templates are adapted.
        """
        raise NotImplementedError

    def _select_sparse_matrix_format(self, proj):
        """
        The sparse matrix format determines the fundamental structure for
        connectivity representation. It depends on the model type as well
        as hardware paradigm.

        Returns (str1, str2, bool):

            * str1:     sparse matrix format declaration
            * str2:     sparse matrix format arguments if needed (e. g. sizes)
            * bool:     if the matrix is a complete (True) or sliced matrix (False)
        """
        if Global.config["structural_plasticity"] and proj._storage_format != "lil":
            raise Global.InvalidConfiguration("Structural plasticity is only allowed for LIL format.")

        # get preferred index type
        idx_type, _, size_type, _ = determine_idx_type_for_projection(proj)

        # Check for the provided format + paradigm combination if a suitable implementation is available.
        if proj.synapse_type.type == "rate":
            # Sanity check
            if proj._storage_order == "pre_to_post":
                Global.CodeGeneratorException("    The storage_order 'pre_to_post' is invalid for rate-coded synapses (Projection: "+proj.name+")")

            if proj._storage_format == "lil":
                if Global._check_paradigm("openmp"):
                    if Global.config['num_threads'] == 1:
                        sparse_matrix_format = "LILMatrix<"+idx_type+", "+size_type+">"
                        single_matrix = True
                    else:
                        if proj._no_split_matrix:
                            sparse_matrix_format = "LILMatrix<"+idx_type+", "+size_type+">"
                            single_matrix = True
                        else:
                            sparse_matrix_format = "ParallelLIL< LILMatrix<"+idx_type+">, "+idx_type+">"
                            single_matrix = False
                else:
                    Global.CodeGeneratorException("    No implementation assigned for rate-coded synapses using LIL and paradigm="+str(Global.config['paradigm'])+" (Projection: "+proj.name+")")

            elif proj._storage_format == "coo":
                if Global._check_paradigm("openmp"):
                    sparse_matrix_format = "COOMatrix<"+idx_type+", "+size_type+">"
                    single_matrix = True

                elif Global._check_paradigm("cuda"):
                    sparse_matrix_format = "COOMatrixCUDA<"+idx_type+", "+size_type+">"
                    single_matrix = True

                else:
                    Global.CodeGeneratorException("    No implementation assigned for rate-coded synapses using COO and paradigm="+str(Global.config['paradigm'])+" (Projection: "+proj.name+")")

            elif proj._storage_format == "csr":
                if Global._check_paradigm("openmp"):
                    sparse_matrix_format = "CSRMatrix<"+idx_type+", "+size_type+">"
                    single_matrix = True

                elif Global._check_paradigm("cuda"):
                    sparse_matrix_format = "CSRMatrixCUDA<"+idx_type+", "+size_type+">"
                    single_matrix = True
                
                else:
                    Global.CodeGeneratorException("    No implementation assigned for rate-coded synapses using CSR and paradigm="+str(Global.config['paradigm'])+" (Projection: "+proj.name+")")

            elif proj._storage_format == "ellr":
                if Global._check_paradigm("openmp"):
                    sparse_matrix_format = "ELLRMatrix<"+idx_type+", "+size_type+">"
                    single_matrix = True

                elif Global._check_paradigm("cuda"):
                    sparse_matrix_format = "ELLRMatrixCUDA<"+idx_type+">"
                    single_matrix = True

                else:
                    Global.CodeGeneratorException("    No implementation assigned for rate-coded synapses using ELLPACK-R and paradigm="+str(Global.config['paradigm'])+" (Projection: "+proj.name+")")

            elif proj._storage_format == "ell":
                if Global._check_paradigm("openmp"):
                    sparse_matrix_format = "ELLMatrix<"+idx_type+", "+size_type+">"
                    single_matrix = True

                else:
                    Global.CodeGeneratorException("    No implementation assigned for rate-coded synapses using ELLPACK and paradigm="+str(Global.config['paradigm'])+" (Projection: "+proj.name+")")

            elif proj._storage_format == "hyb":
                if Global._check_paradigm("openmp"):
                    sparse_matrix_format = "HYBMatrix<"+idx_type+", "+size_type+", true>"
                    single_matrix = True

                elif Global._check_paradigm("cuda"):
                    sparse_matrix_format = "HYBMatrixCUDA<"+idx_type+", "+size_type+">"
                    single_matrix = True

                else:
                    Global.CodeGeneratorException("    No implementation assigned for rate-coded synapses using Hybrid (COO+ELL) and paradigm="+str(Global.config['paradigm'])+" (Projection: "+proj.name+")")

            else:
                Global.CodeGeneratorException("    No implementation assigned for rate-coded synapses using '"+proj._storage_format+"' storage format (Projection: "+proj.name+")")

        elif proj.synapse_type.type == "spike":
            # Check for the provided format + paradigm
            # combination if it's availability

            if proj._storage_format == "lil":
                if proj._storage_order == "pre_to_post":
                    Global.CodeGeneratorException("    The storage_order 'pre_to_post' is invalid for LIL representations (Projection: "+proj.name+")")

                if Global._check_paradigm("openmp"):
                    if Global.config['num_threads'] == 1 or proj._no_split_matrix:
                        sparse_matrix_format = "LILInvMatrix<"+idx_type+", "+size_type+">"
                        single_matrix = True
                    else:
                        sparse_matrix_format = "PartitionedMatrix<LILInvMatrix<"+idx_type+", "+size_type+">, "+idx_type+", "+size_type+">"
                        single_matrix = False

                else:
                    Global.CodeGeneratorException("    No implementation assigned for spiking synapses using LIL and paradigm="+str(Global.config['paradigm'])+ " (Projection: "+proj.name+")")

            elif proj._storage_format == "csr":
                if proj._storage_order == "post_to_pre":
                    if Global._check_paradigm("openmp"):
                        sparse_matrix_format = "CSRCMatrix<"+idx_type+", "+size_type+">"
                        single_matrix = True

                    elif Global._check_paradigm("cuda"):
                        sparse_matrix_format = "CSRCMatrixCUDA<"+idx_type+", "+size_type+">"
                        single_matrix = True

                    else:
                        raise NotImplementedError

                else:
                    if Global._check_paradigm("openmp"):
                        if Global.config['num_threads'] == 1 or proj._no_split_matrix:
                            sparse_matrix_format = "CSRCMatrixT<"+idx_type+", "+size_type+">"
                            single_matrix = True
                        else:
                            sparse_matrix_format = "PartitionedMatrix<CSRCMatrixT<"+idx_type+", "+size_type+">, "+idx_type+", "+size_type+">"
                            single_matrix = False

                    else:
                        raise NotImplementedError


            else:
                Global.CodeGeneratorException("    No implementation assigned for spiking synapses using '"+proj._storage_format+"' storage format (Projection: "+proj.name+")")

        else:
            Global.CodeGeneratorException("    Invalid synapse type " + proj.synapse_type.type)

        # HD (6th Oct 2020)
        # Currently I unified this by flipping the dimensions in CSRCMatrixT in the C++ code
        sparse_matrix_args = " %(post_size)s, %(pre_size)s" % {
            'pre_size': proj.pre.population.size if isinstance(proj.pre, PopulationView) else proj.pre.size,
            'post_size': proj.post.population.size if isinstance(proj.post, PopulationView) else proj.post.size
        }

        if Global.config['verbose']:
            print("Selected", sparse_matrix_format, "(", sparse_matrix_args, ")", "for projection ", proj.name, "and single_matrix =", single_matrix )

        return sparse_matrix_format, sparse_matrix_args, single_matrix

    def _connectivity_init(self, proj, sparse_matrix_format, sparse_matrix_args):
        """
        Each of the pre-defined pattern requires probably different initialization values.
        If no C++ implementation for a pattern is available a default construction from
        LIL is set.
        """
        #
        # Define the correct projection init code. Not all patterns have specialized
        # implementations.
        if proj.connector_name == "Random" and cpp_connector_available("Random", proj._storage_format, proj._storage_order):
            connector_call = """
    void fixed_probability_pattern(std::vector<%(idx_type)s> post_ranks, std::vector<%(idx_type)s> pre_ranks, %(float_prec)s p, %(float_prec)s w_dist_arg1, %(float_prec)s w_dist_arg2, %(float_prec)s d_dist_arg1, %(float_prec)s d_dist_arg2, bool allow_self_connections) {
        static_cast<%(sparse_format)s*>(this)->fixed_probability_pattern(post_ranks, pre_ranks, p, allow_self_connections, rng%(rng_idx)s%(num_threads)s);

%(init_weights)s
%(init_delays)s
    }
"""
        elif proj.connector_name == "Random Convergent" and cpp_connector_available("Random Convergent", proj._storage_format, proj._storage_order):
            connector_call = """
    void fixed_number_pre_pattern(std::vector<%(idx_type)s> post_ranks, std::vector<%(idx_type)s> pre_ranks, unsigned int nnz_per_row, %(float_prec)s w_dist_arg1, %(float_prec)s w_dist_arg2, %(float_prec)s d_dist_arg1, %(float_prec)s d_dist_arg2) {
        static_cast<%(sparse_format)s*>(this)->fixed_number_pre_pattern(post_ranks, pre_ranks, nnz_per_row, rng%(rng_idx)s%(num_threads)s);

%(init_weights)s
%(init_delays)s
    }
"""
        else:
            connector_call = """
    void init_from_lil( std::vector<%(idx_type)s> &row_indices,
                        std::vector< std::vector<%(idx_type)s> > &column_indices,
                        std::vector< std::vector<%(float_prec)s> > &values,
                        std::vector< std::vector<int> > &delays) {
        static_cast<%(sparse_format)s*>(this)->init_matrix_from_lil(row_indices, column_indices%(add_args)s%(num_threads)s);

%(init_weights)s
%(init_delays)s

        init_attributes();
    #ifdef _DEBUG_CONN
        static_cast<%(sparse_format)s*>(this)->print_data_representation();
    #endif
    }
"""

        return connector_call

    def _declaration_accessors(self, proj, single_matrix):
        """
        Generate declaration and accessor code for variables/parameters of the projection.

        Returns:
            (dict, str): first return value contain declaration code and last one the accessor code.

            The declaration dictionary has the following fields:
                delay, event_driven, rng, parameters_variables, additional, cuda_stream
        """
        # create the code for non-specific projections
        declare_event_driven = ""
        declare_rng = ""
        declare_additional = ""

        # Delays
        if proj.max_delay > 1:
            if proj.uniform_delay > 1 :
                key_delay = "uniform"
            else:
                if Global._check_paradigm("cuda"):
                    Global.CodeGeneratorException("Non-uniform delays on rate-coded or spiking synapses are not available for CUDA devices.")

                if proj.synapse_type.type == "rate":
                    key_delay = "nonuniform_rate_coded"
                else:
                    key_delay = "nonuniform_spiking"

            declare_delay = self._templates['delay'][key_delay]['declare']
            init_delay = self._templates['delay'][key_delay]['init']
        else:
            declare_delay = ""
            init_delay = ""

        # Code for declarations and accessors
        declare_parameters_variables, accessor = self._generate_default_get_set(proj, single_matrix)

        # If no psp is defined, it's event-driven
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break
        if has_event_driven:
            declare_event_driven = self._templates['event_driven']['declare']

        # Arrays for the random numbers
        if len(proj.synapse_type.description['random_distributions']) > 0:
            declare_rng += """
    // Random numbers
"""
            for rd in proj.synapse_type.description['random_distributions']:
                declare_rng += self._templates['rng'][rd['locality']]['decl'] % {
                    'rd_name' : rd['name'],
                    'type': rd['ctype'],
                    'float_prec': Global.config['precision'],
                    'template': rd['template'] % {'float_prec':Global.config['precision']}
                }

        # Structural plasticity
        if Global.config['structural_plasticity']:
            declare_parameters_variables += self._header_structural_plasticity(proj)

        # Specific projections can overwrite
        if 'declare_parameters_variables' in proj._specific_template.keys():
            declare_parameters_variables = proj._specific_template['declare_parameters_variables']
        if 'access_parameters_variables' in proj._specific_template.keys():
            accessor = proj._specific_template['access_parameters_variables']
        if 'declare_rng' in proj._specific_template.keys():
            declare_rng = proj._specific_template['declare_rng']
        if 'declare_event_driven' in proj._specific_template.keys():
            declare_event_driven = proj._specific_template['declare_event_driven']
        if 'declare_additional' in proj._specific_template.keys():
            declare_additional = proj._specific_template['declare_additional']

        # Finalize the declarations
        declaration = {
            'declare_delay': declare_delay,
            'init_delay': init_delay,            
            'event_driven': declare_event_driven,
            'rng': declare_rng,
            'parameters_variables': declare_parameters_variables,
            'additional': declare_additional
        }

        return declaration, accessor

    def _generate_default_get_set(self, proj, single_matrix):
        """
        Instead of generating a code block with get/set for each variable we generate a common
        function which receives the name of the variable.
        """
        local_accessor_template = """
    std::vector<std::vector<%(ctype)s>> get_local_attribute_all_%(ctype_name)s(std::string name) {
%(local_get1)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_local_attribute_all_%(ctype_name)s: " << name << " not found" << std::endl;
        return std::vector<std::vector<%(ctype)s>>();
    }

    std::vector<%(ctype)s> get_local_attribute_row_%(ctype_name)s(std::string name, int rk_post) {
%(local_get2)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_local_attribute_row_%(ctype_name)s: " << name << " not found" << std::endl;
        return std::vector<%(ctype)s>();
    }

    %(ctype)s get_local_attribute_%(ctype_name)s(std::string name, int rk_post, int rk_pre) {
%(local_get3)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all_%(ctype_name)s(std::string name, std::vector<std::vector<%(ctype)s>> value) {
%(local_set1)s
    }

    void set_local_attribute_row_%(ctype_name)s(std::string name, int rk_post, std::vector<%(ctype)s> value) {
%(local_set2)s
    }

    void set_local_attribute_%(ctype_name)s(std::string name, int rk_post, int rk_pre, %(ctype)s value) {
%(local_set3)s
    }
"""

        semiglobal_accessor_template = """
    std::vector<%(ctype)s> get_semiglobal_attribute_all_%(ctype_name)s(std::string name) {
%(semiglobal_get1)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_semiglobal_attribute_all_%(ctype_name)s: " << name << " not found" << std::endl;
        return std::vector<%(ctype)s>();
    }

    %(ctype)s get_semiglobal_attribute_%(ctype_name)s(std::string name, int rk_post) {
%(semiglobal_get2)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_semiglobal_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_semiglobal_attribute_all_%(ctype_name)s(std::string name, std::vector<%(ctype)s> value) {
%(semiglobal_set1)s
    }

    void set_semiglobal_attribute_%(ctype_name)s(std::string name, int rk_post, %(ctype)s value) {
%(semiglobal_set2)s
    }
"""
        global_accessor_template = """
    %(ctype)s get_global_attribute_%(ctype_name)s(std::string name) {
%(global_get)s

        // should not happen
        std::cerr << "ProjStruct%(id_proj)s::get_global_attribute_%(ctype_name)s: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute_%(ctype_name)s(std::string name, %(ctype)s value) {
%(global_set)s
    }
"""

        declare_parameters_variables = ""

        # The transpose projection contains no own synaptic parameters
        if isinstance(proj, Transpose):
            return "", "" 

        # choose templates dependend on the paradigm
        decl_template = self._templates['attribute_decl']

        attributes = []
        code_ids_per_type = {}

        # Sort the parameters/variables per type
        for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue

            # add an empty list for this type if needed
            if var['ctype'] not in code_ids_per_type.keys():
                code_ids_per_type[var['ctype']] = []

            # important properties for code generation
            locality = var['locality']
            attr_type = 'parameter' if var in proj.synapse_type.description['parameters'] else 'variable'

            # Special case for single weights
            if var['name'] == "w" and proj._has_single_weight():
                locality = 'global'

            # For GPUs we need to tell the host that this variable need to be updated
            if Global._check_paradigm("cuda"):
                if locality == "global" and attr_type=="parameter":
                    write_dirty_flag = ""
                    read_dirty_flag = ""
                else:
                    write_dirty_flag = "%(name)s_host_to_device = true;" % {'name': var['name']}
                    read_dirty_flag = "if ( %(name)s_device_to_host < t ) device_to_host();" % {'name': var['name']}
            else:
                write_dirty_flag = ""
                read_dirty_flag = ""

            code_ids_per_type[var['ctype']].append({
                'type' : var['ctype'],
                'name': var['name'],
                'locality': locality,
                'attr_type': attr_type,
                'read_dirty_flag': read_dirty_flag,
                'write_dirty_flag': write_dirty_flag
            })

            attributes.append(var['name'])

        # Final code, can contain of multiple sets of accessor functions
        final_code = ""
        for ctype in code_ids_per_type.keys():
            # Attribute accessors/declarators
            local_attribute_get1 = ""
            local_attribute_get2 = ""
            local_attribute_get3 = ""
            local_attribute_set1 = ""
            local_attribute_set2 = ""
            local_attribute_set3 = ""
            semiglobal_attribute_get1 = ""
            semiglobal_attribute_get2 = ""
            semiglobal_attribute_set1 = ""
            semiglobal_attribute_set2 = ""
            global_attribute_get = ""
            global_attribute_set = ""

            for ids in code_ids_per_type[ctype]:
                # Locality of a variable detemines the correct template
                # In case of CUDA also the attribute type is important
                locality = ids['locality']
                attr_type = ids['attr_type']

                #
                # Local variables can be vec[vec[d]], vec[d] or d
                if locality == "local":
                    local_attribute_get1 += """
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_matrix_variable_all<%(type)s>(%(name)s);
        }
""" % ids
                    local_attribute_set1 += """
        if ( name.compare("%(name)s") == 0 ) {
            update_matrix_variable_all<%(type)s>(%(name)s, value);
            %(write_dirty_flag)s
            return;
        }
""" % ids
                    local_attribute_get2 += """
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_matrix_variable_row<%(type)s>(%(name)s, rk_post);
        }
""" % ids
                    local_attribute_set2 += """
        if ( name.compare("%(name)s") == 0 ) {
            update_matrix_variable_row<%(type)s>(%(name)s, rk_post, value);
            %(write_dirty_flag)s
            return;
        }
""" % ids
                    local_attribute_get3 += """
        if ( name.compare("%(name)s") == 0 ) {
            %(read_dirty_flag)s
            return get_matrix_variable<%(type)s>(%(name)s, rk_post, rk_pre);
        }
""" % ids
                    local_attribute_set3 += """
        if ( name.compare("%(name)s") == 0 ) {
            update_matrix_variable<%(type)s>(%(name)s, rk_post, rk_pre, value);
            %(write_dirty_flag)s
            return;
        }
""" % ids

                #
                # Semiglobal variables can be vec[d] or d
                elif locality == "semiglobal":
                    semiglobal_attribute_get1 += """
        if ( name.compare("%(name)s") == 0 ) {
            return get_vector_variable_all<%(type)s>(%(name)s);
        }
""" % ids
                    semiglobal_attribute_get2 += """
        if ( name.compare("%(name)s") == 0 ) {
            return get_vector_variable<%(type)s>(%(name)s, rk_post);
        }
""" % ids
                    semiglobal_attribute_set1 += """
        if ( name.compare("%(name)s") == 0 ) {
            update_vector_variable_all<%(type)s>(%(name)s, value);
            %(write_dirty_flag)s
            return;
        }
""" % ids
                    semiglobal_attribute_set2 += """
        if ( name.compare("%(name)s") == 0 ) {
            update_vector_variable<%(type)s>(%(name)s, rk_post, value);
            %(write_dirty_flag)s
            return;
        }
""" % ids

                #
                # Global variables are only d
                else:
                    global_attribute_get += """
        if ( name.compare("%(name)s") == 0 ) {
            return %(name)s;
        }
""" % ids
                    global_attribute_set += """
        if ( name.compare("%(name)s") == 0 ) {
            %(name)s = value;
            %(write_dirty_flag)s
            return;
        }
""" % ids

                if Global._check_paradigm("cuda") and locality=="global":
                    declare_parameters_variables += decl_template[locality][attr_type] % ids
                else:
                    declare_parameters_variables += decl_template[locality] % ids
                attributes.append(var['name'])

            # build up the final codes
            if local_attribute_get1 != "":
                final_code += local_accessor_template % {
                    'local_get1' : local_attribute_get1,
                    'local_get2' : local_attribute_get2,
                    'local_get3' : local_attribute_get3,
                    'local_set1' : local_attribute_set1,
                    'local_set2' : local_attribute_set2,
                    'local_set3' : local_attribute_set3,
                    'id_proj': proj.id,
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }

            if semiglobal_attribute_get1 != "":
                final_code += semiglobal_accessor_template % {
                    'semiglobal_get1' : semiglobal_attribute_get1,
                    'semiglobal_get2' : semiglobal_attribute_get2,
                    'semiglobal_set1' : semiglobal_attribute_set1,
                    'semiglobal_set2' : semiglobal_attribute_set2,
                    'id_proj': proj.id,
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }
            
            if global_attribute_get != "":
                final_code += global_accessor_template % {
                    'global_get' : global_attribute_get,
                    'global_set' : global_attribute_set,
                    'id_proj': proj.id,
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }

        return declare_parameters_variables, final_code

    @staticmethod
    def _get_attr_and_type(proj, name):
        """
        Small helper function, used for instance in self.update_spike_neuron().

        For a given variable name, the data container is searched and checked,
        whether it is a local or global variable, a random variable or a
        variable related to global operations.

        **Hint**:

        Returns (None, None) by default, if none of this cases is true, indicating
        an error in code generation procedure.
        """
        desc = proj.synapse_type.description
        for attr in desc['parameters']:
            if attr['name'] == name:
                return 'par', attr

        for attr in desc['variables']:
            if attr['name'] == name:
                return 'var', attr

        for attr in desc['random_distributions']:
            if attr['name'] == name:
                return 'rand', attr

        return None, None

    def _header_structural_plasticity(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _init_parameters_variables(self, proj, single_spmv_matrix):
        """
        Generate initialization code for variables / parameters of the
        projection *proj*.

        Returns 3 values:

            ret1 (str): weight initialization
            ret2 (str): delay initialization
            ret3 (str): other initializations (e. g. event-driven)
        """
        # Is it a specific projection?
        if 'init_parameters_variables' in proj._specific_template.keys():
            return proj._specific_template['init_parameters_variables']

        # Learning by default
        code = ""
        weight_code = ""

        # choose initialization templates based on chosen paradigm
        attr_init_tpl = self._templates['attribute_cpp_init']

        attributes = []

        # Initialize parameters
        for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue

            # Important to select which template
            locality = var['locality']
            attr_type = 'parameter' if var in proj.synapse_type.description['parameters'] else 'variable'

            # The synaptic weight
            if var['name'] == 'w':
                if var['locality'] == "global" or proj._has_single_weight():
                    if cpp_connector_available(proj.connector_name, proj._storage_format, proj._storage_order):
                        weight_code = tabify("w = w_dist_arg1;", 2)
                    else:
                        weight_code = tabify("w = values[0][0];", 2)
                    
                elif var['locality'] == "local":
                    if cpp_connector_available(proj.connector_name, proj._storage_format, proj._storage_order):   # Init weights in CPP
                        if proj.connector_weight_dist == None:
                            init_code = self._templates['attribute_cpp_init']['local'] % {
                                'init': 'w_dist_arg1',
                                'type': var['ctype'],
                                'attr_type': 'parameter' if var in proj.synapse_type.description['parameters'] else 'variable',
                                'name': var['name']
                            }

                        elif isinstance(proj.connector_weight_dist, ANNRandom.Uniform):
                            if single_spmv_matrix:
                                init_code = "w = init_matrix_variable_uniform<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng[0]);"
                            else:
                                init_code = "w = init_matrix_variable_uniform<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng);"

                        elif isinstance(proj.connector_weight_dist, ANNRandom.Normal):
                            if single_spmv_matrix:
                                init_code = "w = init_matrix_variable_normal<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng[0]);"
                            else:
                                init_code = "w = init_matrix_variable_normal<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng);"

                        elif isinstance(proj.connector_weight_dist, ANNRandom.LogNormal):
                            if proj.connector_weight_dist.min==None and proj.connector_weight_dist.max==None:
                                if single_spmv_matrix:
                                    init_code = "w = init_matrix_variable_log_normal<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng[0]);"
                                else:
                                    init_code = "w = init_matrix_variable_log_normal<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng);"
                            else:
                                min_code = "std::numeric_limits<%(float_prec)s>::min()" if proj.connector_weight_dist.min==None else str(proj.connector_weight_dist.min)
                                max_code = "std::numeric_limits<%(float_prec)s>::max()" if proj.connector_weight_dist.max==None else str(proj.connector_weight_dist.max)
                                if single_spmv_matrix:
                                    init_code = "w = init_matrix_variable_log_normal_clip<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng[0], "+min_code+", "+max_code+");"
                                else:
                                    init_code = "w = init_matrix_variable_log_normal_clip<%(float_prec)s>(w_dist_arg1, w_dist_arg2, rng, "+min_code+", "+max_code+");"

                        else:
                            raise NotImplementedError( str(type(proj.connector_weight_dist)) + " is not available for CPP-side connection patterns.")

                        if Global._check_paradigm("cuda"):
                            init_code += "\ngpu_w = init_matrix_variable_gpu<%(float_prec)s>(w);"

                        weight_code = tabify(init_code % {'float_prec': Global.config['precision']}, 2)

                    # Init_from_lil
                    else:
                        init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
                        weight_code = attr_init_tpl[locality] % {
                            'id': proj.id,
                            'id_post': proj.post.id,
                            'name': var['name'],
                            'type': var['ctype'],
                            'init': init,
                            'attr_type': attr_type,
                            'float_prec': Global.config['precision']
                        }
                        weight_code += tabify("update_matrix_variable_all<%(float_prec)s>(w, values);" % {'float_prec': Global.config['precision']}, 2)
                        if Global._check_paradigm("cuda"):
                            weight_code += tabify("\nw_host_to_device = true;", 2)

                else:
                    raise NotImplementedError

            # All other variables
            else:
                init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
                var_ids = {
                    'id': proj.id,
                    'id_post': proj.post.id,
                    'name': var['name'],
                    'type': var['ctype'],
                    'init': init,
                    'attr_type': attr_type,
                    'float_prec': Global.config['precision']
                }
                if Global._check_paradigm("cuda") and locality == "global":
                    code += attr_init_tpl[locality][attr_type] % var_ids
                else:
                    code += attr_init_tpl[locality] % var_ids

            attributes.append(var['name'])

        # Initialize delays differs for construction from LIL or CPP inited patterns
        if proj.max_delay > 1:
            # uniform delay
            if proj.connector_delay_dist == None:
                if cpp_connector_available(proj.connector_name, proj._storage_format, proj._storage_order):
                    delay_code = tabify("delay = d_dist_arg1;", 2)
                else:
                    delay_code = self._templates['delay']['uniform']['init']

            # non-uniform delay
            elif isinstance(proj.connector_delay_dist, ANNRandom.RandomDistribution):
                if cpp_connector_available(proj.connector_name, proj._storage_format, proj._storage_order):
                    rng_init = "rng[0]" if single_spmv_matrix else "rng"
                    delay_code = tabify("""
delay = init_matrix_variable_discrete_uniform<int>(d_dist_arg1, d_dist_arg2, %(rng_init)s);
max_delay = -1;""" % {'id_pre': proj.pre.id, 'rng_init': rng_init}, 2)

                else:
                    id_pre = proj.pre.id if not isinstance(proj.pre, PopulationView) else proj.pre.population.id
                    if proj.synapse_type.type == "rate":
                        delay_code = self._templates['delay']['nonuniform_rate_coded']['init'] % {'id_pre': id_pre}
                    else:
                        delay_code = self._templates['delay']['nonuniform_spiking']['init'] % {'id_pre': id_pre}
            else:
                raise NotImplementedError( str(type(proj.connector_weight_dist)) + " is not available.")
        else:
            delay_code = ""

        # If no psp is defined, it's event-driven
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break
        if has_event_driven:
            code += self._templates['event_driven']['cpp_init']

        # Pruning
        if Global.config['structural_plasticity']:
            if 'pruning' in proj.synapse_type.description.keys():
                code += """
        // Pruning
        _pruning = false;
        _pruning_period = 1;
        _pruning_offset = 0;
"""
            if 'creating' in proj.synapse_type.description.keys():
                code += """
        // Creating
        _creating = false;
        _creating_period = 1;
        _creating_offset = 0;
"""

        return weight_code, delay_code, code

    def _init_random_distributions(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _local_functions(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _post_event(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _update_random_distributions(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _update_synapse(self, proj):
        "Implemented by child class"
        raise NotImplementedError

    def _determine_size_in_bytes(self, proj):
        """
        Generate code template to determine size in bytes for the C++ object *proj*. Please note, that this contain only
        default elementes (parameters, variables). User defined elements, parallelization support data structures or similar
        are not considered.

        Consequently implementing generators should extent the resulting code template. This is done by the 'size_in_bytes'
        field in the _specific_template dictionary.
        """
        if 'size_in_bytes' in proj._specific_template.keys():
            return proj._specific_template['size_in_bytes']

        from ANNarchy.generator.Utils import tabify
        code = ""

        # Connectivity
        sparse_matrix_format, _, single_matrix = self._select_sparse_matrix_format(proj)
        code += """
        // connectivity
        size_in_bytes += static_cast<%(spm)s*>(this)->size_in_bytes();
""" % {'spm': sparse_matrix_format}

        # Other variables
        for attr in proj.synapse_type.description['variables']+proj.synapse_type.description['parameters']:
            ids = {
                'ctype': attr['ctype'],
                'name': attr['name'],
                'attr_type': "parameter" if attr in proj.synapse_type.description['parameters'] else "variable"
            }

            locality = attr['locality']
            if attr['name'] == "w" and proj._has_single_weight():
                locality = "global"

            code += self._templates['attribute_cpp_size'][locality] % ids

        return code

    def _clear_container(self, proj):
        """
        Generate code template to destroy allocated container of the C++ object *proj*.

        User defined elements, parallelization support data structures or similar are not considered.
        Consequently implementing generators should extent the resulting code template.
        """
        spm_format, _, _ = self._select_sparse_matrix_format(proj)

        # SpecificProjection should define this field
        if 'clear' in proj._specific_template.keys():
            return proj._specific_template["clear"]

        # Connectivity
        if 'declare_connectivity_matrix' not in proj._specific_template.keys():
            code = """
        // Connectivity
        static_cast<%(spm)s*>(this)->clear();
"""  % {'spm': spm_format}
        else:
            code = ""

        # Variables
        for attr in proj.synapse_type.description['variables'] + proj.synapse_type.description['parameters']:
            ids = {'ctype': attr['ctype'], 'name': attr['name']}

            if attr['name'] == "w" and proj._has_single_weight():
                code += self._templates['attribute_cpp_delete']['global'] % ids    
            else:
                code += self._templates['attribute_cpp_delete'][attr['locality']] % ids

        return code


######################################
### Code generation
######################################
def get_bounds(param):
    """
    Analyses the bounds of a variables used in pre- and post-spike
    statements in a synapse description and returns a code template.
    """
    code = ""
    for bound, val in param['bounds'].items():
        if bound == "init":
            continue

        # Min-Max bounds
        code += """if(%(var)s%(index)s %(operator)s %(val)s)
    %(var)s%(index)s = %(val)s;
""" % {
        'index': "%(local_index)s",
        'var' : param['name'],
        'val' : val,
        'operator': '<' if bound == 'min' else '>'
      }
    return code