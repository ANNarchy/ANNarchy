"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import argparse
from ANNarchy.intern import Messages
from ANNarchy.intern import ConfigManagement

class CmdLineArgParser(object):
    """
    ANNarchy scripts can be run by several command line arguments. These are
    checked with the ArgumentParser provided by Python.

    Generally, we have a group of flags which should be set before *compile()*
    and some which are important for simulation.
    """
    def __init__(self):
        # Create parser instance
        self.setup_parser()
    
    def setup_parser(self):
        """
        We setup the list of possible command line arguments. This selection is presented if *--help* has been added to the script call.
        """
        # override the error behavior of OptionParser,
        # normally an unknwon arg would raise an exception
        self.parser = argparse.ArgumentParser(description='ANNarchy: Artificial Neural Networks architect.')

        group = self.parser.add_argument_group('General')
        group.add_argument("-c", "--clean", help="Forces recompilation.", action="store_true", default=False, dest="clean")
        group.add_argument("--prec", help="Set the floating precision used.", action="store", type=str, default=None, dest="precision")
        group.add_argument("--report", help="Create a network overview using either .tex or .md.", action="store", type=str, default=None, dest="report")

        group = self.parser.add_argument_group('Debugging')
        group.add_argument("--trace", help="Trace initialization of objects (init), function calls during simulation (simulate), or all (both).", action="store", default=None, dest="trace_calls")
        group.add_argument("-d", "--debug", help="Compilation with debug symbols and additional checks.", action="store_true", default=False, dest="debug")
        group.add_argument("-v", "--verbose", help="Shows all messages.", action="store_true", default=None, dest="verbose")

        group = self.parser.add_argument_group('Performance-related')
        group.add_argument("--auto-tuning", help="Enable automatic sparse matrix format selection.", action="store_true", default=False, dest="auto_tuning")

        group = self.parser.add_argument_group('OpenMP')
        group.add_argument("-j", "--num-threads", help="Number of threads to use.", type=int, action="store", default=None, dest="num_threads")
        group.add_argument("--visible-cores", help="Cores where the threads should be placed.", type=str, action="store", default=None, dest="visible_cores")

        group = self.parser.add_argument_group('CUDA')
        group.add_argument("--gpu", help="Enables CUDA and optionally specifies the GPU id (default: 0).", type=int, action="store", nargs='?', default=-1, const=0, dest="gpu_device")

        group = self.parser.add_argument_group('Internal')
        group.add_argument("--profile", help="Enables profiling.", action="store_true", default=None, dest="profile")
        group.add_argument("--profile-out", help="Target file for profiling data.", action="store", type=str, default=None, dest="profile_out")

    def parse_arguments_for_setup(self):
        """
        We already parse some arguments which should be known before **any** other ANNarchy call has happened.
        """
        options, _ = self.parser.parse_known_args()

        # Enables time measurement within the C++ simulation core
        if options.profile:
            ConfigManagement._update_global_config('profiling', options.profile)

        # if the parameters set on command-line they overwrite Global.config
        if options.num_threads is not None:
            ConfigManagement._update_global_config('num_threads', options.num_threads)

        if options.visible_cores is not None:
            try:
                core_list = [int(x) for x in options.visible_cores.split(",")]
                ConfigManagement._update_global_config('visible_cores', core_list)
            except:
                Messages._error("As argument for 'visible_cores' a comma-seperated list of integers is expected.")

        # Get Performance-related flags
        if options.auto_tuning:
            Messages._info("Automatic format selection is an experimental feature. We greatly appreciate bug reports.")
            ConfigManagement.setup(sparse_matrix_format="auto", sparse_matrix_storage_order="auto")

        # Get CUDA configuration
        if options.gpu_device >= 0:
            ConfigManagement._update_global_config('paradigm', 'cuda')

        # Verbose
        if options.verbose is not None:
            ConfigManagement.setup(verbose = options.verbose)

        # Trace init/simulation steps
        if options.trace_calls is not None:
            if options.trace_calls in ["init", "simulate", "both"]:
                ConfigManagement._update_global_config('trace_calls', options.trace_calls)
            else:
                raise ValueError("--trace accepts only one of the following keys: init, simulate, both")

        # Precision
        if options.precision is not None:
            ConfigManagement._update_global_config('precision', options.precision)

