"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.core import Global
from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.core.Population import Population
from ANNarchy.core.Projection import Projection

from ANNarchy.extensions.bold import BoldMonitor
from ANNarchy.extensions.convolution import Transpose

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.intern.ConfigManagement import get_global_config, _check_paradigm
from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.intern.Profiler import Profiler

from ANNarchy.generator.Template import PyxTemplate
from ANNarchy.generator.Population import OpenMPTemplates as omp_templates
from ANNarchy.generator.Population import CUDATemplates as cuda_templates
from ANNarchy.generator.Projection.SingleThread import *
from ANNarchy.generator.Projection.OpenMP import *
from ANNarchy.generator.Projection.CUDA import *
from ANNarchy.generator.Utils import tabify, determine_idx_type_for_projection, cpp_connector_available

class PyxGenerator(object):
    """
    Generate the python extension (*.pyx) file comprising of wrapper
    classes for the individual objects. Secondly the definition of accessible
    methods, e. g. simulate(int steps). Generally an extension consists of two
    parts: a struct definition (define accessible parts of the C++ object) and
    the wrapper object.

    In detail, there are extensions available for:

        * parsed populations
        * parsed projections
        * recorder objects

    Implementation Note (HD: 17.06.2015)

        This class could be implemented in general as a set of functions.
        Nevertheless we chose an object-oriented approach, as it is easier to
        use from Generator object. As the submethods has no access to data
        stored in PyxGenerator (there is no real data at all) all private
        functions are marked with @staticmethod to make this clear.

    TODO:

        * handling of specific populations, projections is currently done over
        generator struct in the population/projection object. This should be
        changed as in the C++ code generation.
    """
    def __init__(self, annarchy_dir, populations, projections, net_id):
        """
        Store a list of population und projection objects for later processing.
        """
        self._annarchy_dir = annarchy_dir
        self._populations = populations
        self._projections = projections
        self._net_id = net_id

    def generate(self):
        """
        Perform code generation.
        """
        # Custom user-defined functions (add_function())
        custom_functions_export, functions_wrapper = self._custom_functions()

        # Custom user-defined constants (add_constant())
        custom_constants_export, constants_wrapper = self._custom_constants()

        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""
        for pop in self._populations:
            # Header export
            pop_struct += self._pop_struct(pop)
            # Population instance
            pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {
    'id': pop.id,
}

        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in self._projections:
            # Header export
            proj_struct += self._proj_struct(proj)

            # Projection instance
            proj_ptr += """
    ProjStruct%(id_proj)s proj%(id_proj)s"""% {
    'id_proj': proj.id,
}

        # struct declaration for each monitor
        monitor_struct = "" #self._pyx_struct_monitor()
        for pop in self._populations:
            monitor_struct += self._pop_monitor_struct(pop)
        for proj in self._projections:
            monitor_struct += self._proj_monitor_struct(proj)
        for mon in NetworkManager().get_monitors(net_id=self._net_id):
            if isinstance(mon, BoldMonitor):
                mon_dict = {
                    'pop_id': mon.object.id,
                    'pop_name': mon.object.name,
                    'mon_id': mon.id,
                    'float_prec': get_global_config('precision')
                }
                monitor_struct += mon._specific_template['pyx_struct'] % mon_dict

        # Cython wrappers for the populations
        pop_class = ""
        for pop in self._populations:
            pop_class += self._pop_wrapper(pop)

        # Cython wrappers for the projections
        proj_class = ""
        for proj in self._projections:
            proj_class += self._proj_wrapper(proj)

        # Cython wrappers for the monitors
        monitor_class = ""
        for pop in self._populations:
            monitor_class += self._pop_monitor_wrapper(pop)
        for proj in self._projections:
            monitor_class += self._proj_monitor_wrapper(proj)
        for mon in NetworkManager().get_monitors(net_id=self._net_id):
            if isinstance(mon, BoldMonitor):
                mon_dict = {
                    'pop_id': mon.object.id,
                    'pop_name': mon.object.name,
                    'mon_id': mon.id,
                    'float_prec': get_global_config('precision')
                }
                monitor_class += mon._specific_template['pyx_wrapper'] % mon_dict

        if Profiler().enabled:
            prof_class = PyxTemplate.pyx_profiler_template
        else:
            prof_class = ""

        from .Template.PyxTemplate import pyx_template
        return pyx_template % {
            'custom_functions_export': custom_functions_export,
            'custom_constants_export': custom_constants_export,
            'prof_class': prof_class,
            'pop_struct': pop_struct,
            'pop_ptr': pop_ptr,
            'proj_struct': proj_struct,
            'proj_ptr': proj_ptr,
            'pop_class' : pop_class,
            'proj_class': proj_class,
            'monitor_struct': monitor_struct,
            'monitor_wrapper': monitor_class,
            'functions_wrapper': functions_wrapper,
            'constants_wrapper': constants_wrapper,
            'float_prec': get_global_config('precision'),
            'device_specific_export': PyxTemplate.pyx_device_specific[get_global_config('paradigm')]['export'],
            'device_specific_wrapper': PyxTemplate.pyx_device_specific[get_global_config('paradigm')]['wrapper'],
        }

    @staticmethod
    def _get_proj_template(proj):
        """
        Choose the correct template collection dependent on target platform and storage formate.

        For example the export of delay related variables.
        """
        if get_global_config('paradigm') == "openmp":
            if proj._storage_format == "lil":
                if get_global_config('num_threads') == 1:
                    return LIL_SingleThread.conn_templates
                else:
                    if proj._no_split_matrix:
                        return LIL_OpenMP.conn_templates
                    else:
                        return LIL_Sliced_OpenMP.conn_templates

            elif proj._storage_format == "coo":
                if get_global_config('num_threads') == 1:
                    return COO_SingleThread.conn_templates
                else:
                    return COO_OpenMP.conn_templates

            elif proj._storage_format == "dia":
                if get_global_config('num_threads') == 1:
                    return DIA_SingleThread.conn_templates
                else:
                    return DIA_OpenMP.conn_templates

            elif proj._storage_format == "bsr":
                if get_global_config('num_threads') == 1:
                    return BSR_SingleThread.conn_templates
                else:
                    return BSR_OpenMP.conn_templates

            elif proj._storage_format == "csr":
                if get_global_config('num_threads') == 1:
                    return CSR_SingleThread.conn_templates
                else:
                    return CSR_OpenMP.conn_templates

            elif proj._storage_format == "ellr":
                if get_global_config('num_threads') == 1:
                    return ELLR_SingleThread.conn_templates
                else:
                    return ELLR_OpenMP.conn_templates

            elif proj._storage_format == "sell":
                if get_global_config('num_threads') == 1:
                    return SELL_SingleThread.conn_templates
                else:
                    return SELL_OpenMP.conn_templates

            elif proj._storage_format == "ell":
                if get_global_config('num_threads') == 1:
                    return ELL_SingleThread.conn_templates
                else:
                    return ELL_OpenMP.conn_templates

            elif proj._storage_format == "hyb":
                if get_global_config('num_threads') == 1:
                    return HYB_SingleThread.conn_templates
                else:
                    raise NotImplementedError

            elif proj._storage_format == "dense":
                if get_global_config('num_threads') == 1:
                    return Dense_SingleThread.conn_templates
                else:
                    return Dense_OpenMP.conn_templates

            else:
                raise Global.InvalidConfiguration("    No python extension definition available for format = "+str(proj._storage_format)+" on CPUs")

        elif get_global_config('paradigm') == "cuda":
            if proj._storage_format == "bsr":
                return BSR_CUDA.conn_templates
            elif proj._storage_format == "csr":
                return CSR_CUDA.conn_templates
            elif proj._storage_format == "csr_scalar":
                return CSR_SCALAR_CUDA.conn_templates
            elif proj._storage_format == "csr_vector":
                return CSR_VECTOR_CUDA.conn_templates
            elif proj._storage_format == "coo":
                return COO_CUDA.conn_templates
            elif proj._storage_format == "sell":
                return SELL_CUDA.conn_templates
            elif proj._storage_format == "ellr":
                return ELLR_CUDA.conn_templates
            elif proj._storage_format == "ell":
                return ELL_CUDA.conn_templates
            elif proj._storage_format == "hyb":
                return HYB_CUDA.conn_templates
            elif proj._storage_format == "dense":
                return Dense_CUDA.conn_templates
            else:
                raise Global.InvalidConfiguration("    No python extension definition available for format = "+str(proj._storage_format)+" on GPUs")

        else:
            raise NotImplementedError

#######################################################################
############## Functions #############################################
#######################################################################
    @staticmethod
    def _custom_functions(obj=None):
        """
        Generate the Python extension code (export and the wrapper code) dependent
        on the type provided in *obj*.
        """
        desc_list = []

        # Check if there are functions where code must be generated
        if obj is None:
            if GlobalObjectManager().number_functions() == 0:
                return "", ""

            from ANNarchy.parser.Extraction import extract_functions
            for _, func in GlobalObjectManager().get_functions():
                desc_list.append(extract_functions(func, local_global=True)[0])
            wrapper_prefix = ""
            export_prefix = "func_"

        elif isinstance(obj, Population):
            if (len(obj.neuron_type.description['functions']) == 0):
                return "", ""
            desc_list = obj.neuron_type.description['functions']
            wrapper_prefix = "pop%(id)s." % {'id': obj.id}
            export_prefix = ""

        elif isinstance(obj, Projection):
            if len(obj.synapse_type.description['functions']) == 0:
                return "", ""
            desc_list = obj.synapse_type.description['functions']
            wrapper_prefix = "proj%(id)s." % {'id': obj.id}
            export_prefix = ""

        # Generate the code
        export = ""
        wrapper = ""
        for desc in desc_list:
            # Export
            export += desc['return_type'] + " " + desc['name'] + '('
            for idx, arg in enumerate(desc['arg_types']):
                export += arg
                if idx < len(desc['arg_types']) - 1:
                    export += ', '
            export += ')' + '\n'

            # Wrapper
            arguments=""
            wrapper += "cpdef np.ndarray " + export_prefix + desc['name'] + '('
            if obj is not None:
                wrapper += "self, "
            for idx, arg in enumerate(desc['args']):
                # Function call
                wrapper += arg
                if idx < len(desc['args']) - 1:
                    wrapper += ', '
                # Element access
                arguments += arg + "[i]"
                if idx < len(desc['args']) - 1:
                    arguments += ', '
            wrapper += '):'
            wrapper += """
    return np.array([%(wrapper_prefix)s%(funcname)s(%(args)s) for i in range(len(%(first_arg)s))])
""" % {'wrapper_prefix': wrapper_prefix, 'funcname': desc['name'], 'first_arg' : desc['args'][0], 'args': arguments}

        # Tabs depend on type
        if obj is None:
            export = tabify(export, 1)
        else:
            export = tabify(export, 2)
            wrapper = tabify(wrapper, 1)

        return export, wrapper

#######################################################################
############## Constants  #############################################
#######################################################################
    def _custom_constants(self):
        if GlobalObjectManager().number_constants() == 0:
            return "", ""

        export = ""
        wrapper = ""
        for obj in GlobalObjectManager().get_constants():
            export += """
    void set_%(name)s(%(float_prec)s)""" % {'name': obj.name, 'float_prec': get_global_config('precision')}
            wrapper += """
def _set_%(name)s(%(float_prec)s value):
    set_%(name)s(value)""" % {'name': obj.name, 'float_prec': get_global_config('precision')}

        return export, wrapper


#######################################################################
############## Population #############################################
#######################################################################
    @staticmethod
    def _pop_struct(pop):
        """
        Generate population struct definition, mimics the c++ class.
        """

        # Spiking neurons have additional data
        export_refractory = ""
        if pop.neuron_type.type == 'spike':
            if pop.neuron_type.refractory or pop.refractory:
                if get_global_config('paradigm') == "openmp":
                    export_refractory = omp_templates.spike_specific["refractory"]["pyx_export"]
                elif get_global_config('paradigm') == "cuda":
                    export_refractory = cuda_templates.spike_specific["refractory"]["pyx_export"]
                else:
                    raise NotImplementedError

        export_parameters_variables = ""
        datatypes = PyxGenerator._get_datatypes(pop)
        # Local parameters and variables
        for ctype in datatypes["local"]:
            export_parameters_variables += PyxTemplate.pyx_default_pop_attribute_export["local"] % {
                'ctype': ctype,
                'ctype_name': ctype.replace(" ", "_")
            }

        # Global parameters and variables
        for ctype in datatypes["global"]:
            export_parameters_variables += PyxTemplate.pyx_default_pop_attribute_export["global"] % {
                'ctype': ctype,
                'ctype_name': ctype.replace(" ", "_")
            }

        if 'export_parameters_variables' in pop._specific_template.keys():
            export_parameters_variables = pop._specific_template['export_parameters_variables']

        # Local functions
        export_functions, _ = PyxGenerator._custom_functions(pop)

        # Mean firing rate
        export_mean_fr = ""
        if pop.neuron_type.type == 'spike':
            export_mean_fr = """
        # Compute firing rate
        void compute_firing_rate(%(float_prec)s window)""" %{'float_prec': get_global_config('precision')}

        # Additional exports
        export_additional = ""
        if 'export_additional' in pop._specific_template.keys():
            export_additional = pop._specific_template['export_additional']

        # Finalize the code
        return PyxTemplate.pop_pyx_struct % {
            'id': pop.id, 'name': pop.name,
            'export_refractory': export_refractory,
            'export_parameters_variables': export_parameters_variables,
            'export_functions': export_functions,
            'export_mean_fr': export_mean_fr,
            'export_additional': export_additional,
        }


    @staticmethod
    def _pop_wrapper(pop):
        """
        Generate population wrapper definition.
        """
        wrapper_args = "size, max_delay"
        wrapper_init = """
        pop%(id)s.set_size(size)
        pop%(id)s.set_max_delay(max_delay)""" % {'id': pop.id}
        wrapper_access_parameters_variables = ""
        wrapper_access_refractory = ""
        wrapper_access_additional = ""


        # Spiking neurons have aditional data
        if pop.neuron_type.type == 'spike':
            if pop.neuron_type.refractory or pop.refractory:
                if get_global_config('paradigm') == "openmp":
                    wrapper_access_refractory += omp_templates.spike_specific['refractory']['pyx_wrapper'] % {'id': pop.id}
                elif get_global_config('paradigm') == "cuda":
                    wrapper_access_refractory += cuda_templates.spike_specific['refractory']['pyx_wrapper'] % {'id': pop.id}
                else:
                    raise NotImplementedError

        # Attributes
        wrapper_access_parameters_variables = PyxGenerator._pop_generate_default_wrapper(pop)

        # Local functions
        _, wrapper_access_functions = PyxGenerator._custom_functions(pop)

        # Mean firing rate
        wrapper_access_mean_fr = ""
        if pop.neuron_type.type == 'spike':
            wrapper_access_mean_fr = """
    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop%(id)s.compute_firing_rate(window)"""% {'id': pop.id}

        # Specific populations can overwrite
        if 'wrapper_args' in pop._specific_template.keys():
            wrapper_args = pop._specific_template['wrapper_args']
        if 'wrapper_init' in pop._specific_template.keys():
            wrapper_init = pop._specific_template['wrapper_init']
        if 'wrapper_access_refractory' in pop._specific_template.keys():
            wrapper_access_refractory = pop._specific_template['wrapper_access_refractory']
        if 'wrapper_access_parameters_variables' in pop._specific_template.keys():
            wrapper_access_parameters_variables = pop._specific_template['wrapper_access_parameters_variables']
        if 'wrapper_access_additional' in pop._specific_template.keys():
            wrapper_access_additional = pop._specific_template['wrapper_access_additional']

        # Finalize the code
        return PyxTemplate.pop_pyx_wrapper % {
            'id': pop.id, 'name': pop.name,
            'wrapper_args' : wrapper_args,
            'wrapper_init' : wrapper_init,
            'wrapper_access_parameters_variables' : wrapper_access_parameters_variables,
            'wrapper_access_functions' : wrapper_access_functions,
            'wrapper_access_refractory' : wrapper_access_refractory,
            'wrapper_access_mean_fr' : wrapper_access_mean_fr,
            'wrapper_access_additional' : wrapper_access_additional,
        }

    @staticmethod
    def _pop_generate_default_wrapper(pop):
        """
        Generates the Python wrapper code for registered attributes.
        """
        datatypes = PyxGenerator._get_datatypes(pop)

        get_local_all = ""
        set_local_all = ""
        get_local = ""
        set_local = ""
        get_global = ""
        set_global = ""

        # Local parameters/variables
        for ctype in datatypes["local"]:
            ids = {
                'id': pop.id,
                'ctype': ctype,
                'ctype_name': ctype.replace(" ", "_")
            }

            # Population.get_cython_attribute expect np.array
            get_local_all += """
        if ctype == "%(ctype)s":
            return np.array(pop%(id)s.get_local_attribute_all_%(ctype_name)s(cpp_string))
""" % ids
            get_local += """
        if ctype == "%(ctype)s":
            return pop%(id)s.get_local_attribute_%(ctype_name)s(cpp_string, rk)
""" % ids

            # Setter
            set_local_all += """
        if ctype == "%(ctype)s":
            pop%(id)s.set_local_attribute_all_%(ctype_name)s(cpp_string, value)
""" % ids
            set_local += """
        if ctype == "%(ctype)s":
            pop%(id)s.set_local_attribute_%(ctype_name)s(cpp_string, rk, value)
""" % ids

        # Global parameters/variables
        for ctype in datatypes["global"]:
            ids = {
                'id': pop.id,
                'ctype': ctype,
                'ctype_name': ctype.replace(" ", "_")
            }

            get_global += """
        if ctype == "%(ctype)s":
            return pop%(id)s.get_global_attribute_%(ctype_name)s(cpp_string)
""" % ids

            set_global += """
        if ctype == "%(ctype)s":
            pop%(id)s.set_global_attribute_%(ctype_name)s(cpp_string, value)
""" % ids

        # Finalize code
        wrapper_code = ""

        if get_local_all != "":
            wrapper_code += PyxTemplate.pyx_default_pop_attribute_wrapper["local"] % {
                'get_local_all': get_local_all,
                'set_local_all': set_local_all,
                'get_local': get_local,
                'set_local': set_local
            }

        if get_global != "":
            wrapper_code += PyxTemplate.pyx_default_pop_attribute_wrapper["global"] % {
                'get_global': get_global,
                'set_global': set_global
            }
        
        return wrapper_code

#######################################################################
############## Projection #############################################
#######################################################################
    @staticmethod
    def _proj_struct(proj):
        """
        The python extension wrapper needs a definition of the corresponding
        C object. The pyx_struct contains all methods, which should be accessible
        by the python extension wrapper.

        Templates:

            structural_plasticity: pruning, creating, calling method
            delay, exact_integ: variables accessed by the wrapper

        """
        # Check for exact intgeration
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break

        # basic
        ids = {
            'id': proj.id,
            'float_prec': get_global_config('precision')
        }

        # Check if we need delay code
        has_delay = (proj.max_delay > 1)
        if proj.uniform_delay > 1 :
            key_delay = "uniform"
        else:
            if proj.synapse_type.type == "rate":
                key_delay = "nonuniform_rate_coded"
            else:
                key_delay = "nonuniform_spiking"

        # get the base templates
        template_dict = PyxGenerator._get_proj_template(proj)

        # Delay
        export_delay = ""
        if has_delay:
            if get_global_config('paradigm') == "openmp":
                export_delay = template_dict['delay'][key_delay]['pyx_struct'] % ids
            elif get_global_config('paradigm') == "cuda":
                export_delay = template_dict['delay'][key_delay]['pyx_struct'] % ids
            else:
                raise NotImplementedError

        # Event-driven
        export_event_driven = ""
        if has_event_driven:
            if get_global_config('paradigm') == "openmp":
                export_event_driven = template_dict['event_driven']['pyx_struct']
            elif get_global_config('paradigm') == "cuda":
                export_event_driven = template_dict['event_driven']['pyx_struct']
            else:
                raise NotImplementedError

        # Determine all export methods
        export_parameters_variables = ""

        # The transpose projection contains no own synaptic parameters
        if isinstance(proj, Transpose):
            export_parameters_variables = ""

        else:
            export_parameters_variables = ""
            datatypes = PyxGenerator._get_datatypes(proj)
            # Local parameters and variables
            for ctype in datatypes["local"]:
                export_parameters_variables += PyxTemplate.pyx_proj_attribute_export["local"] % {
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }

            # Semiglobal parameters and variables
            for ctype in datatypes["semiglobal"]:
                export_parameters_variables += PyxTemplate.pyx_proj_attribute_export["semiglobal"] % {
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }

            # Global parameters and variables
            for ctype in datatypes["global"]:
                export_parameters_variables += PyxTemplate.pyx_proj_attribute_export["global"] % {
                    'ctype': ctype,
                    'ctype_name': ctype.replace(" ", "_")
                }

        # Local functions
        export_functions, _ = PyxGenerator._custom_functions(proj)

        # Structural plasticity
        structural_plasticity = ""
        if get_global_config('structural_plasticity'):
            sp_tpl = template_dict['structural_plasticity']['pyx_struct']

            # Pruning in the synapse
            if 'pruning' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['pruning']
            if 'creating' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['creating']

            # Retrieve the names of extra attributes
            extra_args = ""
            for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
                if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse_type.description['local']:
                    extra_args += ', ' + var['ctype'] + ' ' +  var['name']
            # Generate the code
            structural_plasticity += sp_tpl['func'] % {'extra_args': extra_args}
            if proj.synapse_type.type == "spike":
                structural_plasticity += " "*8 + "void check_and_rebuild_inverse_connectivity()\n"

        # Check if either a custom definition or a CPP side init
        # is available otherwise fall back to init from LIL
        if proj.connector_name == "All-to-All" and cpp_connector_available("All-to-All", proj._storage_format, proj._storage_order):
            export_connector = tabify("bool all_to_all_pattern(vector[%(idx_type)s], vector[%(idx_type)s], %(float_prec)s, %(float_prec)s, %(float_prec)s, %(float_prec)s, bool)", 2)
        elif proj.connector_name == "Random" and cpp_connector_available("Random", proj._storage_format, proj._storage_order):
            export_connector = tabify("bool fixed_probability_pattern(vector[%(idx_type)s], vector[%(idx_type)s], %(float_prec)s, %(float_prec)s, %(float_prec)s, %(float_prec)s, %(float_prec)s, bool)", 2)
        elif proj.connector_name == "Random Convergent" and cpp_connector_available("Random Convergent", proj._storage_format, proj._storage_order):
            export_connector = tabify("bool fixed_number_pre_pattern(vector[%(idx_type)s], vector[%(idx_type)s], %(idx_type)s, %(float_prec)s, %(float_prec)s, %(float_prec)s, %(float_prec)s)", 2)
        else:
            export_connector = tabify("bool init_from_lil(vector[%(idx_type)s], vector[vector[%(idx_type)s]], vector[vector[%(float_prec)s]], vector[vector[int]], bool)", 2)

        # Data types, only of interest if "only_int_idx_type" configuration flag is false
        idx_types = determine_idx_type_for_projection(proj)
        idx_type_dict = {
            'float_prec': get_global_config('precision'),
            'idx_type': idx_types[1],
            'size_type': idx_types[3]
        }

        # Default LIL Definition/ Accessors
        # with an additional accessor spike
        default_conn_export = PyxTemplate.pyx_default_conn_export
        if proj.synapse_type.type == "spike":
            default_conn_export += """
        map[%(idx_type)s, %(idx_type)s] nb_efferent_synapses()
"""
        export_connector = export_connector % idx_type_dict
        export_connector_access = default_conn_export % idx_type_dict

        # Specific projections can overwrite
        if 'export_connector_call' in proj._specific_template.keys():
            export_connector = proj._specific_template['export_connector_call']
        if 'export_connectivity' in proj._specific_template.keys():
            export_connector_access = proj._specific_template['export_connectivity']
        if 'export_delay' in proj._specific_template.keys() and has_delay:
            export_delay = proj._specific_template['export_delay']
        if 'export_event_driven' in proj._specific_template.keys() and has_event_driven:
            export_event_driven = proj._specific_template['export_event_driven']
        if 'export_parameters_variables' in proj._specific_template.keys():
            export_parameters_variables = proj._specific_template['export_parameters_variables']

        # CUDA configuration update
        export_cuda_launch_config = ""
        if _check_paradigm("cuda"):
            export_cuda_launch_config = tabify("void update_launch_config(int, int)", 2)

        return PyxTemplate.proj_pyx_struct % {
            'id_proj': proj.id,
            'export_connectivity': export_connector+export_connector_access,
            'export_delay': export_delay,
            'export_event_driven': export_event_driven,
            'export_parameters_variables': export_parameters_variables,
            'export_functions': export_functions,
            'export_structural_plasticity': structural_plasticity,
            'export_additional': proj._specific_template['export_additional'] if 'export_additional' in proj._specific_template.keys() else "",
            'export_cuda_launch_config': export_cuda_launch_config
        }

    @staticmethod
    def _proj_wrapper(proj):
        """
        Generates the python extension wrapper, which allows access from Python
        to the C module. There are three optional parts (structural plasticity,
        non-uniform delays and exact integration of synaptic events) which we
        need to handle seperatly. The rest of the variables/parameters is handled
        by the standard accessors.

        Templates:

            attribute_pyx_wrapper: normal accessors for variables/parameters
            structural_plasticity: pruning, creating, calling method
            delay, exact_integ: __cinit__ code

        """
        # Check for exact intgeration
        has_event_driven = False
        for var in proj.synapse_type.description['variables']:
            if var['method'] == 'event-driven':
                has_event_driven = True
                break

        # basic
        ids = {
            'id_proj': proj.id,
            'float_prec': get_global_config('precision')
        }

        # Check if we need delay code
        has_delay = (proj.max_delay > 1)
        if proj.uniform_delay > 1 :
            key_delay = "uniform"
        else:
            if proj.synapse_type.type == "rate":
                key_delay = "nonuniform_rate_coded"
            else:
                key_delay = "nonuniform_spiking"

        # Import attributes templates
        wrapper_access_parameters_variables = PyxGenerator._proj_generate_default_export(proj)

        # select the base template
        template_dict = PyxGenerator._get_proj_template(proj)

        # Delays
        if not has_delay:
            wrapper_init_delay = ""
            wrapper_access_delay = ""
        else:
            # Initialize the wrapper
            wrapper_init_delay = template_dict['delay'][key_delay]['pyx_wrapper_init'] % ids
            # Access in wrapper
            wrapper_access_delay = template_dict['delay'][key_delay]['pyx_wrapper_accessor'] % ids

        # Local functions
        _, wrapper_access_functions = PyxGenerator._custom_functions(proj)


        # Additional declarations
        additional_declarations = ""

        # Structural plasticity
        structural_plasticity = ""
        if get_global_config('structural_plasticity'):
            if get_global_config('paradigm') == "openmp":
                sp_tpl = template_dict['structural_plasticity']['pyx_wrapper']
            else:
                sp_tpl = {}

            # Pruning in the synapse
            if 'pruning' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['pruning'] % {'id' : proj.id}

            # Creating in the synapse
            if 'creating' in proj.synapse_type.description.keys():
                structural_plasticity += sp_tpl['creating'] % {'id' : proj.id}

            # Retrieve the names of extra attributes
            extra_args = ""
            extra_values = ""
            for var in proj.synapse_type.description['parameters'] + proj.synapse_type.description['variables']:
                if not var['name'] in ['w', 'delay'] and  var['name'] in proj.synapse_type.description['local']:
                    extra_args += ', ' + var['ctype'] + ' ' +  var['name']
                    extra_values += ', ' +  var['name']

            # Generate the code
            structural_plasticity += sp_tpl['func'] % {'id' : proj.id, 'extra_args': extra_args, 'extra_values': extra_values}
            if proj.synapse_type.type == 'spike':
                structural_plasticity += """    def check_and_rebuild_inverse_connectivity(self):
        proj%(id_proj)s.check_and_rebuild_inverse_connectivity()
""" % {'id_proj': proj.id}

        # Check if either a custom definition or a CPP side init
        # is available otherwise fall back to init from LIL
        if proj.connector_name == "All-to-All" and cpp_connector_available("All-to-All", proj._storage_format, proj._storage_order):
            wrapper_connector_call = """
    def all_to_all(self, post_ranks, pre_ranks, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2, allow_self_connections):
        return proj%(id_proj)s.all_to_all_pattern(post_ranks, pre_ranks, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2, allow_self_connections)
""" % {'id_proj': proj.id}
        elif proj.connector_name == "Random" and cpp_connector_available("Random", proj._storage_format, proj._storage_order):
            wrapper_connector_call = """
    def fixed_probability(self, post_ranks, pre_ranks, p, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2, allow_self_connections):
        return proj%(id_proj)s.fixed_probability_pattern(post_ranks, pre_ranks, p, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2, allow_self_connections)
""" % {'id_proj': proj.id}
        elif proj.connector_name == "Random Convergent" and cpp_connector_available("Random Convergent", proj._storage_format, proj._storage_order):
            wrapper_connector_call = """
    def fixed_number_pre(self, post_ranks, pre_ranks, number_synapses_per_row, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2):
        return proj%(id_proj)s.fixed_number_pre_pattern(post_ranks, pre_ranks, number_synapses_per_row, w_dist_arg1, w_dist_arg2, d_dist_arg1, d_dist_arg2)
""" % {'id_proj': proj.id}
        else:
            wrapper_connector_call = """
    def init_from_lil_connectivity(self, synapses):
        " synapses is an instance of LILConnectivity "
        return proj%(id_proj)s.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay, synapses.requires_sorting)

    def init_from_lil(self, post_rank, pre_rank, w, delay, requires_sorting):
        return proj%(id_proj)s.init_from_lil(post_rank, pre_rank, w, delay, requires_sorting)
""" % {'id_proj': proj.id}

        wrapper_args = ""
        wrapper_init = tabify("pass",3)
        wrapper_access_connectivity = PyxTemplate.pyx_default_conn_wrapper 
        if proj.synapse_type.type == "spike":   # additional for spike
            wrapper_access_connectivity += """
    def nb_efferent_synapses(self):
        return proj%(id_proj)s.nb_efferent_synapses()
"""
        wrapper_access_connectivity = wrapper_access_connectivity % {'id_proj': proj.id}

        # Specific projections can overwrite
        if 'wrapper_args' in proj._specific_template.keys():
            wrapper_args = proj._specific_template['wrapper_args']
        if 'wrapper_init_connectivity' in proj._specific_template.keys():
            wrapper_init = proj._specific_template['wrapper_init_connectivity']
        if 'wrapper_access_connectivity' in proj._specific_template.keys():
            wrapper_access_connectivity = proj._specific_template['wrapper_access_connectivity']
        if 'wrapper_connector_call' in proj._specific_template.keys():
            wrapper_connector_call = proj._specific_template['wrapper_connector_call']
        if 'wrapper_init_delay' in proj._specific_template.keys() and has_delay:
            wrapper_init_delay = proj._specific_template['wrapper_init_delay']
        if 'wrapper_access_delay' in proj._specific_template.keys() and has_delay:
            wrapper_access_delay = proj._specific_template['wrapper_access_delay']
        if 'wrapper_access_parameters_variables' in proj._specific_template.keys():
            wrapper_access_parameters_variables = proj._specific_template['wrapper_access_parameters_variables']
        if 'wrapper_access_additional' in proj._specific_template.keys():
            additional_declarations = proj._specific_template['wrapper_access_additional']

        # CUDA configuration update
        wrapper_cuda_launch_config = ""
        if _check_paradigm("cuda"):
            wrapper_cuda_launch_config = """
    def update_launch_config(self, nb_blocks=-1, threads_per_block=32):
        proj%(id_proj)s.update_launch_config(nb_blocks, threads_per_block)
""" % {'id_proj': proj.id}

        return PyxTemplate.proj_pyx_wrapper % {
            'id_proj': proj.id,
            'pre_size': proj.pre.population.size if isinstance(proj.pre, PopulationView) else proj.pre.size,
            'post_size': proj.post.population.size if isinstance(proj.post, PopulationView) else proj.post.size,
            'wrapper_args' : wrapper_args,
            'wrapper_init' : wrapper_init,
            'wrapper_connector_call': wrapper_connector_call,
            'wrapper_init_delay': wrapper_init_delay,
            'wrapper_access_connectivity': wrapper_access_connectivity,
            'wrapper_access_delay': wrapper_access_delay,
            'wrapper_access_parameters_variables': wrapper_access_parameters_variables,
            'wrapper_access_functions': wrapper_access_functions,
            'wrapper_access_structural_plasticity': structural_plasticity,
            'wrapper_access_additional': additional_declarations,
            'wrapper_cuda_launch_config': wrapper_cuda_launch_config
        }

    @staticmethod
    def _proj_generate_default_export(proj):
        """
        To prevent overloaded functions by different types, we need to declare
        all accessors per c++ data type which is used.
        """
        # The transpose projection contains no own synaptic parameters
        if isinstance(proj, Transpose):
            return ""

        get_local_all = ""
        set_local_all = ""
        get_local_row = ""
        set_local_row = ""
        get_local = ""
        set_local = ""
        get_semiglobal_all = ""
        set_semiglobal_all = ""
        get_semiglobal = ""
        set_semiglobal = ""
        set_global = ""
        get_global = ""

        datatypes = PyxGenerator._get_datatypes(proj)
        for ctype in datatypes["local"]:
            ids = {
                'id_proj': proj.id,
                'ctype': ctype,
                'ctype_name': ctype.replace(" ", "_")
            }

            get_local_all += """
        if ctype == "%(ctype)s":
            return proj%(id_proj)s.get_local_attribute_all_%(ctype_name)s(cpp_string)
""" % ids
            set_local_all += """
        if ctype == "%(ctype)s":
            proj%(id_proj)s.set_local_attribute_all_%(ctype_name)s(cpp_string, value)
""" % ids
            get_local_row += """
        if ctype == "%(ctype)s":
            return proj%(id_proj)s.get_local_attribute_row_%(ctype_name)s(cpp_string, rk_post)
""" % ids
            set_local_row += """
        if ctype == "%(ctype)s":
            proj%(id_proj)s.set_local_attribute_row_%(ctype_name)s(cpp_string, rk_post, value)
""" % ids
            get_local += """
        if ctype == "%(ctype)s":
            return proj%(id_proj)s.get_local_attribute_%(ctype_name)s(cpp_string, rk_post, rk_pre)
""" % ids
            set_local += """
        if ctype == "%(ctype)s":
            proj%(id_proj)s.set_local_attribute_%(ctype_name)s(cpp_string, rk_post, rk_pre, value)
""" % ids

        for ctype in datatypes["semiglobal"]:
            ids = {
                'id_proj': proj.id,
                'ctype': ctype,
                'ctype_name': ctype.replace(" ", "_")
            }

            get_semiglobal_all += """
        if ctype == "%(ctype)s":
            return proj%(id_proj)s.get_semiglobal_attribute_all_%(ctype_name)s(cpp_string)
""" % ids
            set_semiglobal_all += """
        if ctype == "%(ctype)s":
            proj%(id_proj)s.set_semiglobal_attribute_all_%(ctype_name)s(cpp_string, value)
""" % ids
            get_semiglobal += """
        if ctype == "%(ctype)s":
            return proj%(id_proj)s.get_semiglobal_attribute_%(ctype_name)s(cpp_string, rk_post)
""" % ids
            set_semiglobal += """
        if ctype == "%(ctype)s":
            proj%(id_proj)s.set_semiglobal_attribute_%(ctype_name)s(cpp_string, rk_post, value)
""" % ids

        for ctype in datatypes["global"]:
            ids = {
                'id_proj': proj.id,
                'ctype': ctype,
                'ctype_name': ctype.replace(" ", "_")
            }

            get_global += """
        if ctype == "%(ctype)s":            
            return proj%(id_proj)s.get_global_attribute_%(ctype_name)s(cpp_string)
""" % ids
            set_global += """
        if ctype == "%(ctype)s":            
            proj%(id_proj)s.set_global_attribute_%(ctype_name)s(cpp_string, value)
""" % ids

        # Finalize code
        wrapper_code = ""

        if get_local_all != "":
            wrapper_code += PyxTemplate.pyx_proj_attribute_wrapper["local"] % {
                'get_local_all': get_local_all,
                'set_local_all': set_local_all,
                'get_local_row': get_local_row,
                'set_local_row': set_local_row,
                'get_local': get_local,
                'set_local': set_local,
                'id_proj': proj.id
            }

        if get_semiglobal_all != "":
            wrapper_code += PyxTemplate.pyx_proj_attribute_wrapper["semiglobal"] % {
                'get_semiglobal_all': get_semiglobal_all,
                'set_semiglobal_all': set_semiglobal_all,
                'get_semiglobal': get_semiglobal,
                'set_semiglobal': set_semiglobal,
                'id_proj': proj.id
            }

        if get_global != "":
            wrapper_code += PyxTemplate.pyx_proj_attribute_wrapper["global"] % {
                'get_global': get_global,
                'set_global': set_global,
                'id_proj': proj.id
            }

        return wrapper_code

#######################################################################
############## Monitors  ##############################################
#######################################################################
    @staticmethod
    def _pop_monitor_struct(pop):
        """
        Generate recorder struct.
        """
        tpl_code = """
    # Population %(id)s (%(name)s) : Monitor
    cdef cppclass PopRecorder%(id)s (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder%(id)s* get_instance(int)
        long int size_in_bytes()
        void clear()
"""
        attributes = []
        for var in pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            if var['name'] in pop.neuron_type.description['local']:
                tpl_code += """
        vector[vector[%(type)s]] %(name)s
        bool record_%(name)s
""" % {'name': var['name'], 'type': var['ctype']}
            elif var['name'] in pop.neuron_type.description['global']:
                tpl_code += """
        vector[%(type)s] %(name)s
        bool record_%(name)s
""" % {'name': var['name'], 'type': var['ctype']}

        if pop.neuron_type.type == 'spike':
            tpl_code += """
        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()
"""
            if pop.neuron_type.axon_spike:
                tpl_code += """
        map[int, vector[long]] axon_spike
        bool record_axon_spike
        void clear_axon_spike()
"""

        # Arrays for the presynaptic sums
        if pop.neuron_type.type == 'rate':
            tpl_code += """
        # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                tpl_code += """
        vector[vector[%(float_prec)s]] _sum_%(target)s
        bool record__sum_%(target)s
""" % {'target': target, 'float_prec': get_global_config('precision')}

        return tpl_code % {'id' : pop.id, 'name': pop.name}

    @staticmethod
    def _pop_monitor_wrapper(pop):
        """
        Generate recorder wrapper.
        """
        tpl_code = """
# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder%(id)s_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder%(id)s.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder%(id)s.get_instance(self.id)).size_in_bytes()

    def clear(self):
        return (PopRecorder%(id)s.get_instance(self.id)).clear()

    property period:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).period_
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).period_ = val

    property period_offset:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).period_offset_
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).period_offset_ = val
"""
        attributes = []
        for var in pop.neuron_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])
            tpl_code += """
    property %(name)s:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).%(name)s
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).%(name)s = val
    property record_%(name)s:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).record_%(name)s
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).record_%(name)s = val
    def clear_%(name)s(self):
        (PopRecorder%(id)s.get_instance(self.id)).%(name)s.clear()
""" % {'id' : pop.id, 'name': var['name']}

        if pop.neuron_type.type == 'spike':
            tpl_code += """
    property spike:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).spike
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).spike = val
    property record_spike:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).record_spike
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).record_spike = val
    def clear_spike(self):
        (PopRecorder%(id)s.get_instance(self.id)).clear_spike()
""" % {'id' : pop.id}

            if pop.neuron_type.axon_spike:
                tpl_code += """
    property axon_spike:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).axon_spike
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).axon_spike = val
    property record_axon_spike:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).record_axon_spike
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).record_axon_spike = val
    def clear_axon_spike(self):
        (PopRecorder%(id)s.get_instance(self.id)).clear_axon_spike()
""" % {'id' : pop.id}

        # Arrays for the presynaptic sums
        if pop.neuron_type.type == 'rate':
            tpl_code += """
    # Targets"""
            for target in sorted(list(set(pop.neuron_type.description['targets'] + pop.targets))):
                tpl_code += """
    property %(name)s:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).%(name)s
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).%(name)s = val
    property record_%(name)s:
        def __get__(self): return (PopRecorder%(id)s.get_instance(self.id)).record_%(name)s
        def __set__(self, val): (PopRecorder%(id)s.get_instance(self.id)).record_%(name)s = val
    def clear_%(name)s(self):
        (PopRecorder%(id)s.get_instance(self.id)).%(name)s.clear()
""" % {'id' : pop.id, 'name': '_sum_'+target}

        return tpl_code % {'id' : pop.id, 'name': pop.name}

    @staticmethod
    def _proj_monitor_struct(proj):
        """
        Generate projection recorder struct
        """

        # Specific template
        if 'monitor_export' in proj._specific_template.keys():
            return proj._specific_template['monitor_export']


        code = """
    # Projection %(id)s : Monitor
    cdef cppclass ProjRecorder%(id)s (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder%(id)s* get_instance(int)
"""

        templates = {
        'local': """
        vector[vector[vector[%(type)s]]] %(name)s
        bool record_%(name)s
""",
        'semiglobal': """
        vector[vector[%(type)s]] %(name)s
        bool record_%(name)s
""",
        'global': """
        vector[%(type)s] %(name)s
        bool record_%(name)s
"""
        }

        attributes = []
        for var in proj.synapse_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            # Get the locality
            locality = var['locality']

            # Special case for single weights
            if var['name'] == "w" and proj._has_single_weight():
                locality = 'global'

            # Use the correct template
            code +=  templates[locality] % {'name': var['name'], 'type': var['ctype']}

        return code % {'id' : proj.id}

    @staticmethod
    def _proj_monitor_wrapper(proj):
        """
        Generate projection recorder struct
        """

        # Specific template
        if 'monitor_wrapper' in proj._specific_template.keys():
            return proj._specific_template['monitor_wrapper']

        code = """
# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder%(id)s_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder%(id)s.create_instance(ranks, period, period_offset, offset)
"""

        attributes = []
        for var in proj.synapse_type.description['variables']:
            # Avoid doublons
            if var['name'] in attributes:
                continue
            attributes.append(var['name'])

            code += """
    property %(name)s:
        def __get__(self): return (ProjRecorder%(id)s.get_instance(self.id)).%(name)s
        def __set__(self, val): (ProjRecorder%(id)s.get_instance(self.id)).%(name)s = val
    property record_%(name)s:
        def __get__(self): return (ProjRecorder%(id)s.get_instance(self.id)).record_%(name)s
        def __set__(self, val): (ProjRecorder%(id)s.get_instance(self.id)).record_%(name)s = val
    def clear_%(name)s(self):
        (ProjRecorder%(id)s.get_instance(self.id)).%(name)s.clear()
""" % {'id' : proj.id, 'name': var['name']}

        return code % {'id' : proj.id}

#######################################################################
############## Helpers   ##############################################
#######################################################################
    @staticmethod
    def _get_datatypes(obj):
        """
        Helper method used in proj_wrapper/proj_export or pop_wrapper/pop_export
        """
        if isinstance(obj, Projection):
            datatypes = {
                'local': [],
                'semiglobal': [],
                'global': []
            }

            for var in obj.synapse_type.description['parameters'] + obj.synapse_type.description['variables']:
                locality = var['locality']
                if var['name'] == "w" and obj._has_single_weight():
                    locality = 'global'

                if var['ctype'] not in datatypes[locality]:
                    datatypes[locality].append(var['ctype'])

        elif isinstance(obj, Population):
            datatypes = {
                'local': [],
                'global': []
            }

            if obj.neuron_type.type == 'spike':
                if 'int' not in datatypes['local']:
                    datatypes['local'].append('int')

            for var in obj.neuron_type.description['parameters'] + obj.neuron_type.description['variables']:
                locality = var['locality']
                if var['ctype'] not in datatypes[locality]:
                    datatypes[locality].append(var['ctype'])

        else:
            ValueError("PyxGenerator._get_datatypes() expects either Population or Projection instance.")

        return datatypes
