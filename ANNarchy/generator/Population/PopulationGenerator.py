"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.generator.Utils import tabify

class PopulationGenerator(object):
    """
    Base class for population generators in ANNarchy. Inherited by

    * OpenMPGenerator: single-thread and multi-core implementation
    * CUDAGenerator: gpu implementation
    """
    def __init__(self, profile_generator, net_id):
        """
        Initialize PopulationGenerator.
        """
        self._prof_gen = profile_generator
        self._net_id = net_id

    def header_struct(self, pop, annarchy_dir):
        """
        Generate the c-style struct definition for a population object.

        Parameters:

            pop: Population object
            annarchy_dir: working directory

        Returns:

            dictionary: include directive, pointer definition, call statements
                        and other informations needed by CodeGenerator to
                        create the ANNarchy.cpp/.cu code.

                        Please note, the returned dictionary may vary dependent
                        on population type and used parallelization paradigm.

        Exceptions:

            Will cause a NotImplementedError as it should implemented by the
            inheriting generator classes.
        """
        raise NotImplementedError

    def reset_computesum(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _delay_code(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _generate_decl_and_acc(self, pop):
        """
        Data exchange between Python and ANNarchyCore library is done by specific 
        get-/set-methods. This function creates for all variables and parameters
        the corresponding methods.
        """
        # Parameters, Variables
        declaration, accessors, already_processed = self._generate_default_get_set(pop)

        # The conductance/current variables for spiking neurons are stored in
        # pop.neuron_type.description['variables'] but only if they are used.
        if pop.neuron_type.type == 'spike':
            try:
                all_targets = set(pop.neuron_type.description['targets'] + pop.targets)
            except TypeError:
                # The projection has multiple targets
                all_targets = set(pop.neuron_type.description['targets'] + pop.targets[0])

            for target in sorted(list(all_targets)):
                attr_name = 'g_'+target
                if attr_name not in already_processed:
                    # we assume here, that targets are local variables
                    id_dict = {
                        'type' : get_global_config('precision'),
                        'name': attr_name,
                        'attr_type': 'variable'
                    }
                    declaration += self._templates['attr_decl']['local'] % id_dict
                    already_processed.append(attr_name)

        # Global operations
        if len(pop.global_operations) != 0:
            declaration += """
    // Global operations
"""
            for op in pop.global_operations:
                op_dict = {
                    'type': get_global_config('precision'),
                    'op': op['function'],
                    'var': op['variable']
                }
                
                if _check_paradigm("openmp"):
                    declaration += """    %(type)s _%(op)s_%(var)s;
""" % op_dict
                elif _check_paradigm("cuda"):
                    declaration += """
    %(type)s _%(op)s_%(var)s;
    %(type)s* _gpu_%(op)s_%(var)s;
""" % op_dict
                else:
                    raise NotImplementedError

        # Arrays for the random numbers
        declaration += """
    // Random numbers
"""
        for rd in pop.neuron_type.description['random_distributions']:
            declaration += self._templates['rng'][rd['locality']]['decl'] % {
                'rd_name' : rd['name'],
                'type': rd['ctype'],
                'template': rd['template'] % {'float_prec':get_global_config('precision')}
            }

        return declaration, accessors

    @staticmethod
    def _get_attr(pop, name):
        """
        Small helper function, used for instance in self.update_spike_neuron()
        """
        for attr in pop.neuron_type.description['variables'] + pop.neuron_type.description['parameters']:
            if attr['name'] == name:
                return attr

        return None

    @staticmethod
    def _get_attr_and_type(pop, name):
        """
        Small helper function, used for instance in self.update_spike_neuron().

        For a given variable name, the data container is searched and checked,
        whether it is a local or global variable, a random variable or a
        variable related to global operations.
        """
        for attr in pop.neuron_type.description['parameters']:
            if attr['name'] == name:
                return 'par', attr

        for attr in pop.neuron_type.description['variables']:
            if attr['name'] == name:
                return 'var', attr

        for attr in pop.neuron_type.description['random_distributions']:
            if attr['name'] == name:
                return 'rand', attr

        # the given name wasn't either an attribute nor a random distribution,
        # lets test if it was a psp
        for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
            if name == "sum("+target+")":
                return 'psp', { 'ctype': get_global_config('precision'), 'name': '_sum_'+target }

        return None, None

    def _init_fr(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _init_globalops(self, pop):
        """
        Generate the C++ codes for the initialization of global operations
        within the Population::init_population() method.
        """
        if len(pop.global_operations) == 0:
            return ""

        code = "\n// Initialize global operations\n"
        for op in pop.global_operations:
            ids = {
                'op': op['function'],
                'var': op['variable'],
                'type': get_global_config('precision')
            }

            if _check_paradigm("openmp"):
                code += """_%(op)s_%(var)s = 0.0;
""" % ids

            elif _check_paradigm("cuda"):
                code += """_%(op)s_%(var)s = 0.0;
cudaMalloc((void**)&_gpu_%(op)s_%(var)s, sizeof(%(type)s));
""" % ids

            else:
                raise NotImplementedError

        return tabify(code, 2)

    def _init_population(self, pop):
        """
        Generate the codes for the C++ function Population::init_population() method.
        """
        code = ""
        attr_tpl = self._templates['attribute_cpp_init']
        already_processed = []

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            # Avoid doublons
            if var['name'] in already_processed:
                continue

            if _check_paradigm("cuda") and var['locality'] == "global":
                code += attr_tpl[var['locality']]['parameter'] % {'name': var['name']}
            else:
                init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
                var_ids = {'id': pop.id, 'name': var['name'], 'type': var['ctype'],
                        'init': init, 'attr_type': 'parameter'}
                code += attr_tpl[var['locality']] % var_ids

            already_processed.append(var['name'])

        # Variables
        for var in pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in already_processed:
                continue

            init = 'false' if var['ctype'] == 'bool' else ('0' if var['ctype'] == 'int' else '0.0')
            var_ids = {'id': pop.id, 'name': var['name'], 'type': var['ctype'],
                       'init': init, 'attr_type': 'variable'}

            if _check_paradigm("cuda") and var['locality'] == "global":
                code += attr_tpl[var['locality']]['variable'] % var_ids
            else:
                code += attr_tpl[var['locality']] % var_ids

            already_processed.append(var['name'])

        # Random numbers 
        if _check_paradigm("openmp"):
            if len(pop.neuron_type.description['random_distributions']) > 0:
                rng_code = "\n// Random numbers\n"
                for rd in pop.neuron_type.description['random_distributions']:
                    rng_ids = {
                        'id': pop.id,
                        'rd_name': rd['name'],
                        'type': rd['ctype'],
                    }
                    rng_code += self._templates['rng'][rd['locality']]['init'] % rng_ids
                
                code += tabify(rng_code,2)

        else:
            if len(pop.neuron_type.description['random_distributions']) > 0:
                code += """
                // Random numbers"""
                for dist in pop.neuron_type.description['random_distributions']:
                    rng_ids = {
                        'id': pop.id,
                        'rd_name': dist['name'],
                    }
                    code += self._templates['rng'][dist['locality']]['init'] % rng_ids

        # Global operations
        code += self._init_globalops(pop)

        # rate-coded targets
        if pop.neuron_type.type == 'rate':
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                ids = {
                    'id': pop.id,
                    'name': "_sum_"+target,
                    'attr_type': 'psp',
                    'type': get_global_config('precision'),
                    'init': 0.0
                }
                code += attr_tpl['local'] % ids

        # or unused synaptic spiking targets
        else:
            try:
                all_targets = set(pop.neuron_type.description['targets'] + pop.targets)
            except TypeError:
                # The projection has multiple targets
                all_targets = set(pop.neuron_type.description['targets'] + pop.targets[0])

            for target in sorted(list(all_targets)):
                attr_name = 'g_'+target
                if attr_name not in already_processed:
                    id_dict = {
                        'type' : get_global_config('precision'),
                        'name': attr_name,
                        'attr_type': 'variable',
                        'init': 0.0
                    }
                    code += self._templates['attribute_cpp_init']['local'] % id_dict
                    already_processed.append(attr_name)
                    
        return code

    def _local_functions(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _stop_condition(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_fr(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_globalops(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_random_distributions(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_rate_neuron(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _update_spiking_neuron(self, pop):
        "Implemented by child class"
        raise NotImplementedError

    def _clear_container(self, pop):
        """
        Generate code template to destroy allocated container of the C++ object *pop*. Should be implemented by child
        class.

        User defined elements, parallelization support data structures or similar are not considered. Consequently
        implementing generators should extent the resulting code template.
        """
        from ANNarchy.generator.Utils import tabify
        code = """
    #ifdef _DEBUG
        std::cout << "PopStruct%(id)s::clear()" << std::endl;
    #endif
""" % {'id': pop.id}

       # Variables
        code += "// Parameters\n"
        for attr in pop.neuron_type.description['parameters']:
            # HD: we need to clear only local parameters, the others are no vectors
            if attr['locality'] == "local":
                # HD: clear alone does not deallocate, it only resets size.
                #     So we need to call shrink_to_fit afterwards.
                ids = {'ctype': attr['ctype'], 'name': attr['name']}
                code += "%(name)s.clear();\n" % ids
                code += "%(name)s.shrink_to_fit();\n" % ids
        code+="\n"

        # Variables
        code += "// Variables\n"
        for attr in pop.neuron_type.description['variables']:
            # HD: we need to clear only local variables, the others are no vectors
            if attr['locality'] == "local":
                # HD: clear alone does not deallocate, it only resets size.
                #     So we need to call shrink_to_fit afterwards.
                ids = {'ctype': attr['ctype'], 'name': attr['name']}
                code += "%(name)s.clear();\n" % ids
                code += "%(name)s.shrink_to_fit();\n" % ids

        # Spike-specific code
        if pop.neuron_type.description['type'] == 'spike':
            code += self._templates['spike_specific']['spike']['clear']

            # Mean - FR
            code += """
// Mean Firing Rate
for (auto it = _spike_history.begin(); it != _spike_history.end(); it++) {
    while(!it->empty())
        it->pop();
}
_spike_history.clear();
_spike_history.shrink_to_fit();
"""

        # Random variables
        code += "\n// RNGs\n"
        for dist in pop.neuron_type.description['random_distributions']:
            rng_ids = {
                'id': pop.id,
                'rd_name': dist['name'],
            }
            code += self._templates['rng'][dist['locality']]['clear'] % rng_ids

        code = tabify(code, 2)
        return code

    def _size_in_bytes(self, pop):
        """
        Generate code template to determine size in bytes for the C++ object *pop*. Please note, that this contain only
        default elements (parameters, variables). User defined elements, parallelization support data structures or
        similar are not considered.

        Consequently implementing generators should extent the resulting code template. This is done by filling the
        'size_in_bytes' field in the _specific_template.
        """
        if 'size_in_bytes' in pop._specific_template.keys():
            return pop._specific_template['size_in_bytes']

        from ANNarchy.generator.Utils import tabify
        code = ""

        # Parameters
        code += "// Parameters\n"
        for attr in pop.neuron_type.description['parameters']:
            ids = {'ctype': attr['ctype'], 'name': attr['name']}
            if attr['locality'] == "global":
                code += "size_in_bytes += sizeof(%(ctype)s);\t// %(name)s\n" % ids
            else:
                code += "size_in_bytes += sizeof(std::vector<%(ctype)s>) + sizeof(%(ctype)s) * %(name)s.capacity();\t// %(name)s\n" % ids

        # Variables
        code += "// Variables\n"
        for attr in pop.neuron_type.description['variables']:
            ids = {'ctype': attr['ctype'], 'name': attr['name']}
            if attr['locality'] == "global":
                code += "size_in_bytes += sizeof(%(ctype)s);\t// %(name)s\n" % ids
            else:
                code += "size_in_bytes += sizeof(std::vector<%(ctype)s>) + sizeof(%(ctype)s) * %(name)s.capacity();\t// %(name)s\n" % ids

        # Random variables
        code +="// RNGs\n"
        if _check_paradigm("openmp"):
            for dist in pop.neuron_type.description['random_distributions']:
                ids = {
                    'ctype': dist['ctype'],
                    'name': dist['name']
                }
                if dist['locality'] == "local":
                    code += "size_in_bytes += sizeof(std::vector<%(ctype)s>) + sizeof(%(ctype)s) * %(name)s.capacity();\t// %(name)s\n" % ids
                else:
                    code += "size_in_bytes += sizeof(%(ctype)s);\t// %(name)s\n" % ids
        else:
            for dist in pop.neuron_type.description['random_distributions']:
                code += "size_in_bytes += sizeof(curandState*);\t// gpu_%(name)s\n" % {'name': dist['name']}

        code = tabify(code, 2)
        return code

    def _generate_default_get_set(self, pop):
        """
        Generate a get/set template for all attributes in the given population
        """
        declaration = "" # member declarations
        accessors = "" # export member functions
        already_processed = []
        code_ids_per_type = {}

        # Sort the parameters/variables per type
        for var in pop.neuron_type.description['parameters'] + pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in already_processed:
                continue

            # add an empty list for this type if needed
            if var['ctype'] not in code_ids_per_type.keys():
                code_ids_per_type[var['ctype']] = []

            # Important which template to choose
            locality = var['locality']
            attr_type = 'parameter' if var in pop.neuron_type.description['parameters'] else 'variable'

            # For GPUs we need to tell the host that this variable need to be updated
            if _check_paradigm("cuda"):
                if attr_type == "parameter" and locality == "global":
                    read_dirty_flag = ""
                    write_dirty_flag = ""
                else:
                    write_dirty_flag = "%(name)s_host_to_device = true;" % {'name': var['name']}
                    read_dirty_flag = "if ( %(name)s_device_to_host < t ) device_to_host();" % {'name': var['name']}
            else:
                read_dirty_flag = ""
                write_dirty_flag = ""

            # add to the processing list
            code_ids_per_type[var['ctype']].append({
                'type' : var['ctype'],
                'name': var['name'],
                'locality': locality,
                'attr_type': attr_type,
                'write_dirty_flag': write_dirty_flag,
                'read_dirty_flag': read_dirty_flag
            })

            already_processed.append(var['name'])

        # For rate-coded models add _sum_target
        if pop.neuron_type.type == "rate":
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                prec_type = get_global_config('precision')

                # add to the processing list
                code_ids_per_type[prec_type].append({
                    'type' : prec_type,
                    'name': "_sum_"+target,
                    'locality': 'local',
                    'attr_type': 'psp',
                    'write_dirty_flag': "_sum_"+target+"_host_to_device = true;",
                    'read_dirty_flag': "if ( _sum_"+target+"_device_to_host < t ) device_to_host();"
                })

        # For spiking models we add spike vector
        if pop.neuron_type.type == "spike":
            # create new type category if needed
            if "int" not in code_ids_per_type.keys():
                code_ids_per_type["int"] = []

            # add to the processing list
            code_ids_per_type["int"].append({
                'type' : "int",
                'name': "spiked",
                'locality': 'local',
                'attr_type': 'spike',
                'write_dirty_flag': "",
                'read_dirty_flag': ""
            })

        # Final code, can contain of multiple sets of accessor functions
        accessors = ""

        for ctype in code_ids_per_type.keys():
            local_attribute_get1 = ""
            local_attribute_get2 = ""
            local_attribute_set1 = ""
            local_attribute_set2 = ""
            global_attribute_get = ""
            global_attribute_set = ""

            for ids in code_ids_per_type[ctype]:
                locality = ids['locality']

                # Accessor codes
                if locality == "local":
                    local_attribute_get1 += self._templates["attr_acc"]["local_get_all"] % ids
                    local_attribute_get2 += self._templates["attr_acc"]["local_get_single"] % ids

                    local_attribute_set1 += self._templates["attr_acc"]["local_set_all"] % ids
                    local_attribute_set2 += self._templates["attr_acc"]["local_set_single"] % ids

                elif locality == "global":
                    global_attribute_get += self._templates["attr_acc"]["global_get"] % ids
                    global_attribute_set += self._templates["attr_acc"]["global_set"] % ids

                else:
                    raise ValueError("PopulationGenerator: invalild locality type for attribute")

                # Declaration codes
                if ids['name'] == "spiked":
                    declaration += ""   # already declared
                elif _check_paradigm("cuda") and locality == "global":
                    declaration += self._templates['attr_decl'][locality][ids['attr_type']] % ids
                else:
                    declaration += self._templates['attr_decl'][locality] % ids

            # build up the final codes
            if local_attribute_get1 != "":
                accessors += self._templates["accessor_template"]["local"] % {
                    'local_get1' : local_attribute_get1,
                    'local_get2' : local_attribute_get2,
                    'local_set1' : local_attribute_set1,
                    'local_set2' : local_attribute_set2,
                    'id': pop.id,
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }

            if global_attribute_get != "":
                accessors += self._templates["accessor_template"]["global"] % {
                    'global_get' : global_attribute_get,
                    'global_set' : global_attribute_set,
                    'id': pop.id,
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }

        return declaration, accessors, already_processed