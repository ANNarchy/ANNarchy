"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern.NetworkManager import NetworkManager

from ANNarchy.generator.Utils import tabify
from ANNarchy.generator import Profile
from ANNarchy.generator.Template import SingleThreadBaseTemplate
from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_st
from ANNarchy.generator.SimCore.SimCoreGenerator import SimCoreGenerator

class SingleThreadGenerator(SimCoreGenerator):
    """
    The class is responsible to generate the C++ header definition for a Population
    object. The code is intendend to run on a single CPU core.
    """

    def __init__(self, profile_generator, net_id):
        # The super here calls all the base classes, so first
        # ProjectionGenerator and afterwards OpenMPConnectivity
        # TODO: this is python 2 syntax
        super(SingleThreadGenerator, self).__init__(profile_generator, net_id)

        self._net_id = net_id

    #####################################################################
    # header-related functions                                          #
    #####################################################################
    def generate_header(self, source_dest, pop_desc, proj_desc):
        """
        Generate the ANNarchyCore[net_id].hpp code. This header represents the interface to
        the Python extension and therefore includes all network objects.
        """
        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""

        for pop in pop_desc:
            pop_struct += pop["include"]
            pop_ptr += pop["extern"]

        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in proj_desc:
            proj_struct += proj["include"]
            proj_ptr += proj["extern"]

        # Custom functions
        custom_func = self._header_custom_functions()

        # Custom constants
        custom_constant = self._header_custom_constants()

        # data type used for floating values.
        float_type = ConfigManager().get("default_dtype", self._net_id)

        header_code = SingleThreadBaseTemplate.header_template % {
            "float_prec": float_type.cpp_decl_type,
            "py_float_prec": float_type.py_decl_type,
            "pop_struct": pop_struct,
            "proj_struct": proj_struct,
            "pop_ptr": pop_ptr,
            "proj_ptr": proj_ptr,
            "custom_func": custom_func,
            "custom_constant": custom_constant,
            "built_in": SingleThreadBaseTemplate.built_in_functions
            + SingleThreadBaseTemplate.integer_power
            % {"float_prec": ConfigManager().get("precision", self._net_id)},
        }

        # Generate header code for the analysed pops and projs
        if ConfigManager().get("paradigm", self._net_id) == "openmp":
            with open(source_dest + "ANNarchyCore" + str(self._net_id) + ".hpp", "w", encoding="utf-8") as ofile:
                ofile.write(header_code)

    def _header_custom_constants(self):
        """
        Generate code for custom constants
        """
        network = NetworkManager().get_network(self._net_id)
        constants = network.get_constants()

        if len(constants) == 0:
            return ""

        code = ""
        for obj in constants:
            obj_str = {
                "name": obj.name,
                "float_prec": ConfigManager().get("precision", self._net_id),
            }

            code += (
                """
extern %(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value);"""
                % obj_str
            )

        return code

    #####################################################################
    # body-related functions                                            #
    #####################################################################
    def generate_body(self, annarchy_dir, source_dest, prof_gen, pop_gen, pop_desc, proj_desc):
        """
        Generate the codes 'main' library file. The generated code
        will be used in different files, dependent on the chosen
        target platform:

        * openmp: ANNarchyCore[net_id].cpp
        * cuda: ANNarchyHost.cu and ANNarchyDevice.cu
        """
        # struct declaration for each population
        pop_ptr = ""
        for pop in pop_desc:
            pop_ptr += pop["instance"]

        # struct declaration for each projection
        proj_ptr = ""
        for proj in proj_desc:
            proj_ptr += proj["instance"]

        # Code for the global operations
        glop_definition = self._body_def_glops()
        update_globalops = ""
        for pop in pop_desc:
            if "gops_update" in pop.keys():
                update_globalops += pop["gops_update"]

        # Reset presynaptic sums
        reset_sums = self._body_resetcomputesum_pop(pop_gen)

        # Compute presynaptic sums
        compute_sums = ""
        for proj in proj_desc:
            compute_sums += proj["compute_psp"]

        # Update random distributions
        rd_update_code = ""
        for desc in pop_desc + proj_desc:
            if "rng_update" in desc.keys():
                rd_update_code += desc["rng_update"]

        # Equations for the neural variables
        update_neuron = ""
        for pop in pop_desc:
            if "update" in pop.keys():
                update_neuron += pop["update"]

        # Enque delayed outputs
        delay_code = ""
        for pop in pop_desc:
            if "delay_update" in pop.keys():
                delay_code += pop["delay_update"]

        # Equations for the synaptic variables
        update_synapse = ""
        for proj in proj_desc:
            if "update" in proj.keys():
                update_synapse += proj["update"]

        # Equations for the post-events
        post_event = ""
        for proj in proj_desc:
            if "post_event" in proj.keys():
                post_event += proj["post_event"]

        # Structural plasticity
        structural_plasticity, sp_spike_backward_view_update = (
            self._body_structural_plasticity()
        )

        # Early stopping
        run_until = self._body_run_until()

        # Profiling
        if prof_gen:
            prof_dict = prof_gen.generate_body_dict()
            prof_dict["prof_include"] = prof_dict["prof_include"].replace("extern ", "")
        else:
            prof_dict = Profile.ProfileGenerator(
                annarchy_dir, self._net_id
            ).generate_body_dict()

        # custom constants
        custom_constant, _ = self._body_custom_constants()

        # data type used for floating values.
        float_type = ConfigManager().get("default_dtype", self._net_id)

        # code fields for openMP/single thread template
        base_dict = {
            "net_id": self._net_id,
            "float_prec": float_type.cpp_decl_type,
            "py_float_prec": float_type.py_decl_type,
            "pop_ptr": pop_ptr,
            "proj_ptr": proj_ptr,
            "glops_def": glop_definition,
            "initialize": self._body_initialize(prof_gen, pop_desc, proj_desc),
            "run_until": run_until,
            "compute_sums": compute_sums,
            "reset_sums": reset_sums,
            "update_neuron": update_neuron,
            "update_globalops": update_globalops,
            "update_synapse": update_synapse,
            "random_dist_update": rd_update_code,
            "delay_code": delay_code,
            "post_event": post_event,
            "structural_plasticity": structural_plasticity,
            "custom_constant": custom_constant,
            "sp_spike_backward_view_update": sp_spike_backward_view_update,
        }

        # profiling
        base_dict.update(prof_dict)

        # complete code template
        body_code = SingleThreadBaseTemplate.body_template % base_dict

        # ... and write to file
        with open(source_dest + "ANNarchyCore" + str(self._net_id) + ".cpp", "w", encoding='utf-8') as ofile:
            ofile.write(body_code)

    def _body_custom_constants(self):
        """
        Generate code for custom constants dependent on the target paradigm
        set in global settings.

        Returns:

        * decl_code: declarations in header file
        * init_code: initialization code
        """
        network = NetworkManager().get_network(self._net_id)
        constants = network.get_constants()

        if len(constants) == 0:
            return "", ""

        decl_code = ""
        init_code = ""
        for obj in constants:
            obj_str = {
                "name": obj.name,
                "value": obj.value,
                "float_prec": ConfigManager().get("precision", self._net_id),
            }
            decl_code += (
                """
%(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value){%(name)s = value;};"""
                % obj_str
            )
            init_code += (
                """
        %(name)s = %(float_prec)s{0};"""
                % obj_str
            )

        return decl_code, init_code

    def _body_def_glops(self):
        """
        Dependent on the used global operations we add pre-defined templates
        to the ANNarchy body file.

        Return:

            dependent on the used paradigm we return one string (single thread, OpenMP)
            or tuple(string, string) (CUDA).
        """
        ops = []
        for pop in NetworkManager().get_network(self._net_id).get_populations():
            for op in pop.global_operations:
                ops.append(op["function"])

        # no global operations
        if ops == []:
            return ""

        type_def = {"type": ConfigManager().get("precision", self._net_id)}

        code = ""
        for op in sorted(list(set(ops))):
            code += global_operation_templates_st[op] % type_def

        return code

    def _body_initialize(self, prof_gen, pop_desc, proj_desc):
        """
        Define codes for the method initialize(), comprising of population and projection
        initializations, optionally profiling class.
        """
        profiling_init = (
            "" if prof_gen is None else prof_gen.generate_init_network()
        )

        # Initialize populations
        population_init = "    // Initialize populations\n"
        for pop in pop_desc:
            population_init += pop["init"]

        # Initialize projections
        projection_init = "    // Initialize projections\n"
        for proj in proj_desc:
            projection_init += proj["init"]

        # Custom  constants
        _, custom_constant = self._body_custom_constants()

        return SingleThreadBaseTemplate.initialize_template % {
            "cpp_float_prec": ConfigManager().get("default_dtype", self._net_id).cpp_decl_type,
            "prof_init": profiling_init,
            "pop_init": population_init,
            "proj_init": projection_init,
            "custom_constant": custom_constant,
        }

    def _body_run_until(self):
        """
        Generate the code for conditioned stop of simulation
        """
        # Check if it is useful to generate anything at all
        for pop in NetworkManager().get_network(self._net_id).get_populations():
            if pop.stop_condition:
                break
        else:
            # No stop conditions were detected
            return SingleThreadBaseTemplate.run_until_template["default"]

        # a condition has been defined, so we generate corresponding code
        cond_code = ""
        for pop in NetworkManager().get_network(self._net_id).get_populations():
            if pop.stop_condition:
                cond_code += SingleThreadBaseTemplate.run_until_template["single_pop"] % {"id": pop.id}

        return SingleThreadBaseTemplate.run_until_template["body"] % {"run_until": cond_code}

    def _body_structural_plasticity(self):
        """
        Call of pruning or creating methods if necessary.

        Returns two strings:
            * call statements called within singleStep()
            * call statements called at begin of simulation loop
        """
        # Pruning if any
        pruning = ""
        creating = ""
        rebuild_in_cpp = ""
        rebuild_out_cpp = ""

        if ConfigManager().get("structural_plasticity", self._net_id):
            for proj in NetworkManager().get_network(self._net_id).get_projections():
                rebuild_needed = False
                if "pruning" in proj.synapse_type.description.keys():
                    pruning += tabify(f"proj{proj.id}->pruning();\n", 1)
                    rebuild_needed = True
                if "creating" in proj.synapse_type.description.keys():
                    creating += tabify(f"proj{proj.id}->creating();\n", 1)
                    rebuild_needed = True
                # we only check those projections which are possibly modified
                if rebuild_needed and proj.synapse_type.type == "spike":
                    rebuild_in_cpp += tabify(
                        f"proj{proj.id}->check_and_rebuild_inverse_connectivity();\n", 1
                    )
                # we don't know which projection the user modifies, so we need to check all
                if proj.synapse_type.type == "spike":
                    rebuild_out_cpp += tabify(
                        f"proj{proj.id}->check_and_rebuild_inverse_connectivity();\n", 1
                    )

        return creating + pruning + rebuild_in_cpp, rebuild_out_cpp
