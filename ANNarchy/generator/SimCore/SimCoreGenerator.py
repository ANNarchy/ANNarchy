"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.GlobalObjects import GlobalObjectManager

from ANNarchy.intern.NetworkManager import NetworkManager
from ANNarchy.parser.Extraction import extract_functions

class SimCoreGenerator(object):
    """
    Abstract definition of a SimCoreGenerator class.

    Note that, the structuring of the code, i.e., the number of files, or
    the implementation of specific functions, e.g., custom constants,
    can heavily rely on the target paradigm. Therefore, this class defines
    mostly a set of functions which need to be implemented for each desired
    target paradigm.

    The functions *generate_header* and *generate_body* are called from
    the CodeGenerator class.
    """

    def __init__(self, profile_generator, net_id):
        """
        Initialization of the class object and store some ids.
        """
        super(SimCoreGenerator, self).__init__()

        self._prof_gen = profile_generator
        self._net_id = net_id

    #################################################################################
    # Functions called from CodeGenerator                                           #
    #################################################################################
    def generate_header(self, source_dest: str, pop_desc: dict, proj_desc: dict):
        """
        Generate source files, e.g., *.hpp or *.cuh implementing the declaration the defined network.
        For instance being imported by the nanobind-wrapper.

        Args:

            source_dest: location for the generated files.
            pop_desc: a dictionary containing code snippets for initialization of population objects.
            proj_desc: a dictionary containing code snippets for initialization of projection objects.

        Returns:

            None, but the successful call will lead to generated source files stored in *source_dest*.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError

    def generate_body(self, annarchy_dir: str, source_dest: str, prof_gen, pop_gen, pop_desc: dict, proj_desc: dict):
        """
        Generate source files, e.g., *.cpp or *.cu implementing the definition and execution of
        the defined network.

        Args:

            annarchy_dir: location of ANNarchy base directory.
            source_dest: location for the generated files.
            prof_gen: ProfileGenerator instance, if None no profiling code will be added.
            pop_gen: PopulationGenerator instance, can not be None.
            pop_desc: a dictionary containing code snippets for initialization of population objects.
            proj_desc: a dictionary containing code snippets for initialization of projection objects.

        Returns:

            None, but the successful call will lead to generated source files stored in *source_dest*.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError

    #################################################################################
    # Functions called from SimCoreGenerator class (i.e., mostly child classes)     #
    #################################################################################
    def _header_custom_functions(self):
        """
        Generate code for custom functions defined globally and being used within
        neuron or synapse model implementationss. These functions can only rely on
        provided arguments.

        Implementation note:

            This is currently the same for all implementor classes, as we support for
            CUDA custom functions only on the host-side yet.
        """
        if GlobalObjectManager().number_functions() == 0:
            return ""

        code = ""
        for _, func in GlobalObjectManager().get_functions():
            code += (
                extract_functions(
                    description=func, local_global=True, net_id=self._net_id
                )[0]["cpp"]
                + "\n"
            )

        return code

    def _header_custom_constants(self):
        """
        Generate declarations for custom functions defined globally and are usable
        witihn neuron or synapse descriptions. These functions can only rely on
        provided arguments.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError

    def _body_custom_constants(self):
        """
        Generate definitions of custom functions defined globally and are usable
        witihn neuron or synapse descriptions. These functions can only rely on
        provided arguments.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError

    def _body_def_glops(self):
        """
        Implementation of required global operations, such as min, max, or mean.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError

    def _body_initialize(self, prof_gen, pop_desc: dict, proj_desc: dict):
        """
        Define codes for the method initialize(), comprising of population and projection
        initializations, optionally profiling class.

        Args:

            prof_gen: ProfileGenerator instance, if None no profiling code will be added.
            pop_desc: a dictionary containing code snippets for initialization of population objects.
            proj_desc: a dictionary containing code snippets for initialization of projection objects.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError

    def _body_resetcomputesum_pop(self, pop_gen):
        """
        Rate-coded neurons sum up the received inputs in temporary variables.
        They need to be cleared in each time step. As this is platform-specific,
        the PopulationGenerator need to be accessed.

        Args:
            pop_gen: a PopulationGenerator instance.
        """
        code = ""
        for pop in NetworkManager().get_network(self._net_id).get_populations():
            if pop.neuron_type.type == "rate":
                code += pop_gen.reset_computesum(pop)

        return code

    def _body_run_until(self):
        """
        ANNarchy allows to interrupt the simulation based on a condition.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError

    def _body_structural_plasticity(self):
        """
        These code snippets implement the activity-based (or other rules) change
        of the connectivity between populations.

        Implementation note:

            Should be implemented by child-class.
        """
        raise NotImplementedError
