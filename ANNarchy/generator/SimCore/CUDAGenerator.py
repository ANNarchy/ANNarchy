"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from math import log, ceil

from ANNarchy.intern.GlobalObjects import GlobalObjectManager
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern.NetworkManager import NetworkManager

from ANNarchy.parser.Extraction import extract_functions

from ANNarchy.generator.Utils import tabify
from ANNarchy.generator import Profile
from ANNarchy.generator.Template import CUDABaseTemplate
from ANNarchy.generator.Template.GlobalOperationTemplate import global_operation_templates_cuda
from ANNarchy.generator.SimCore.SimCoreGenerator import SimCoreGenerator

class CUDAGenerator(SimCoreGenerator):
    """
    """

    def __init__(self, cuda_config, profile_generator, net_id):
        super(CUDAGenerator, self).__init__(profile_generator, net_id)

        self._cuda_config = cuda_config
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

        # kernel declaration
        invoke_kernel_def = ""
        for pop in pop_desc:
            invoke_kernel_def += pop["update_header"]

        for proj in proj_desc:
            invoke_kernel_def += proj["psp_kernel_decl"]
            invoke_kernel_def += proj["update_synapse_header"]
            invoke_kernel_def += proj["postevent_header"]

        glob_ops_header, _, _ = self._body_def_glops()
        invoke_kernel_def += glob_ops_header

        device_invoke_header = CUDABaseTemplate.device_invoke_header % {
            "float_prec": ConfigManager().get("precision", self._net_id),
            "invoke_kernel_def": invoke_kernel_def,
        }

        host_header_code = CUDABaseTemplate.header_template % {
            "float_prec": float_type.py_decl_type,
            "pop_struct": pop_struct,
            "proj_struct": proj_struct,
            "pop_ptr": pop_ptr,
            "proj_ptr": proj_ptr,
            "custom_func": custom_func,
            "built_in": CUDABaseTemplate.built_in_functions,
            "custom_constant": custom_constant,
        }

        with open(source_dest + "ANNarchyKernel" + str(self._net_id) + ".cuh", "w", encoding="utf-8") as ofile:
            ofile.write(device_invoke_header)
        with open(source_dest + "ANNarchyCore" + str(self._net_id) + ".hpp", "w", encoding="utf-8") as ofile:
            ofile.write(host_header_code)

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
        update_globalops = ""
        for pop in pop_desc:
            if "gops_update" in pop.keys():
                update_globalops += pop["gops_update"]

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

        # Implementation notice ( HD: 10. June, 2015 )
        #
        # The CUDA linking process is a big problem for object oriented approaches
        # and the seperation of implementation codes into several files. Even in the
        # current SDK 5.0 this problem is not fully solved. Linking is available, but
        # only for small, independent code pieces, by far not sufficient for full
        # object-oriented approaches ...
        #
        # For us, this currently have one consequence: we cannot completely seperate
        # the implementation of objects into several files. To hold a certain equality
        # between the structures of objects, I implemented the following workaround:
        #
        # We create the c-structs holding data fields and accessors as in OpenMP. We also
        # create the kernels, call entity in the corresponding generator objects, and
        # return the codes via the descriptor dictionary.
        #
        # This ensures a consistent interface in the generators and also in the generated
        # codes, but sometimes require additional overhead. Hopefully NVidia will improve
        # their linker in the next releases, so one could remove this overhead.
        psp_call = ""
        for proj in proj_desc:
            psp_call += proj["psp_host_call"]

        # custom constants
        device_custom_constant, _ = self._body_custom_constants()

        # custom functions
        custom_func = ""
        for pop in pop_desc:
            custom_func += pop["custom_func"]
        for proj in proj_desc:
            custom_func += proj["custom_func"]
        for _, func in GlobalObjectManager().get_functions():
            custom_func += (
                extract_functions(
                    description=func, local_global=True, net_id=self._net_id
                )[0]["cpp"].replace("inline", "__device__")
                + "\n"
            )

        # pre-defined/common available kernel
        common_kernel = self._cuda_common_kernel(NetworkManager().get_network(self._net_id).get_projections())

        pop_kernel = ""
        pop_invoke_kernel = ""
        for pop in pop_desc:
            pop_kernel += pop["update_body"]
            pop_invoke_kernel += pop["update_invoke"]

        pop_update_fr = ""
        for pop in pop_desc:
            pop_update_fr += pop["update_FR"]

        psp_device_kernel = ""
        psp_invoke_kernel = ""
        for proj in proj_desc:
            psp_device_kernel += proj["psp_device_kernel"]
            psp_invoke_kernel += proj["psp_invoke_kernel"]

        delay_code = ""
        for pop in pop_desc:
            if "update_delay" in pop.keys():
                delay_code += pop["update_delay"]

        syn_kernel = ""
        syn_invoke_kernel = ""
        for proj in proj_desc:
            syn_kernel += proj["update_synapse_body"]
            syn_invoke_kernel += proj["update_synapse_invoke"]

        syn_call = ""
        for proj in proj_desc:
            syn_call += proj["update_synapse_call"]

        postevent_device_kernel = ""
        postevent_invoke_kernel = ""
        for proj in proj_desc:
            postevent_device_kernel += proj["postevent_body"]
            postevent_invoke_kernel += proj["postevent_invoke"]

        postevent_call = ""
        for proj in proj_desc:
            postevent_call += proj["postevent_call"]

        clear_sums = self._body_resetcomputesum_pop(pop_gen)

        # global operations
        _, glob_ops_invoke, glob_ops_body = self._body_def_glops()

        # determine number of threads per kernel
        threads_per_kernel = self._cuda_kernel_config()

        # concurrent kernel execution
        stream_setup = self._cuda_stream_config()

        # memory transfers
        host_device_transfer, device_host_transfer = "", ""
        for pop in pop_desc:
            host_device_transfer += pop["host_to_device"]
            # DtoH is performed only when an accessor is called.

        for proj in proj_desc:
            host_device_transfer += proj["host_to_device"]
            # DtoH is performed only when an accessor is called.

        # Profiling
        if prof_gen:
            prof_dict = prof_gen.generate_body_dict()
            prof_dict["prof_include"] = prof_dict["prof_include"].replace(
                "extern ", ""
            )
        else:
            prof_dict = Profile.ProfileGenerator(
                annarchy_dir, self._net_id
            ).generate_body_dict()

        device_code = (
            CUDABaseTemplate.device_kernel
            % {  # Target: ANNarchyKernel[net_id].cu
                "net_id": self._net_id,
                "common_kernel": common_kernel,
                "pop_kernel": pop_kernel,
                "pop_invoke_kernel": pop_invoke_kernel,
                "psp_kernel": psp_device_kernel,
                "psp_invoke_kernel": psp_invoke_kernel,
                "syn_kernel": syn_kernel,
                "syn_invoke_kernel": syn_invoke_kernel,
                "glob_ops_kernel": glob_ops_body,
                "glob_ops_invoke_kernel": glob_ops_invoke,
                "postevent_kernel": postevent_device_kernel,
                "postevent_invoke_kernel": postevent_invoke_kernel,
                "custom_func": custom_func,
                "custom_constant": device_custom_constant,
                "built_in": CUDABaseTemplate.built_in_functions
                + CUDABaseTemplate.integer_power
                % {"float_prec": ConfigManager().get("precision", self._net_id)},
                "float_prec": ConfigManager().get("precision", self._net_id),
            }
        )

        base_dict = {
            # network definitions
            "net_id": self._net_id,
            "float_prec": ConfigManager().get("precision", self._net_id),
            "pop_ptr": pop_ptr,
            "proj_ptr": proj_ptr,
            "run_until": run_until,
            "clear_sums": clear_sums,
            "compute_sums": psp_call,
            "update_neuron": update_neuron,
            "update_FR": pop_update_fr,
            "update_globalops": update_globalops,
            "update_synapse": syn_call,
            "post_event": postevent_call,
            "delay_code": delay_code,
            "initialize": self._body_initialize(prof_gen, pop_desc, proj_desc),
            "structural_plasticity": "",
            # cuda host specific
            "stream_setup": stream_setup,
            "host_device_transfer": host_device_transfer,
            "device_host_transfer": device_host_transfer,
            "kernel_config": threads_per_kernel,
            "sp_spike_backward_view_update": "",
        }
        base_dict.update(prof_dict)
        host_code = (
            CUDABaseTemplate.host_body_template % base_dict
        )  # Target: ANNarchyCore[net_id].cpp

        with open(source_dest + "ANNarchyCore" + str(self._net_id) + ".cpp", "w", encoding="utf-8") as ofile:
            ofile.write(host_code)
        with open(source_dest + "ANNarchyKernel" + str(self._net_id) + ".cu", "w", encoding="utf-8") as ofile:
            ofile.write(device_code)

    def _body_custom_constants(self):
        """
        Generate code for custom constants dependent on the target paradigm
        set in global settings.

        Returns (openMP):

        * decl_code: declarations in header file
        * init_code: initialization code

        Returns (CUDA):

        * device_decl_code: declarations in header file (device side)
        * host_init_code: initialization code (host side)

        """
        network = NetworkManager().get_network(self._net_id)
        constants = network.get_constants()

        if len(constants) == 0:
            return "", ""

        host_init_code = ""
        device_decl_code = ""
        for obj in constants:
            obj_str = {
                "name": obj.name,
                "value": obj.value,
                "float_prec": ConfigManager().get("precision", self._net_id),
            }
            device_decl_code += (
                    """__device__ __constant__ %(float_prec)s %(name)s;
void set_%(name)s(%(float_prec)s value) {
    cudaError_t err = cudaMemcpyToSymbol(%(name)s, &value, sizeof(%(float_prec)s), 0, cudaMemcpyHostToDevice);
#ifndef NDEBUG
    std::cout << "set global constant %(name)s = " << value << std::endl;
    if ( err != cudaSuccess )
        std::cerr << cudaGetErrorString(err) << std::endl;
#endif
}"""
                % obj_str
            )

            # TODO: is this really needed, it's overwritten anyways ?
            host_init_code += (
                """
        set_%(name)s(0.0);"""
                % obj_str
            )

        return device_decl_code, host_init_code

    def _body_def_glops(self):
        """
        """
        ops = []
        for pop in NetworkManager().get_network(self._net_id).get_populations():
            for op in pop.global_operations:
                ops.append(op["function"])

        # no global operations
        if ops == []:
            return "", "", ""

        type_def = {"type": ConfigManager().get("precision", self._net_id)}

        # the computation kernel depends on the paradigm
        header = ""
        invoke = ""
        body = ""

        for op in sorted(list(set(ops))):
            header += global_operation_templates_cuda[op]["header"] % type_def
            invoke += global_operation_templates_cuda[op]["invoke"] % type_def
            body += global_operation_templates_cuda[op]["body"] % type_def

        return header, invoke, body

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

        init_tpl = CUDABaseTemplate.host_initialize_template

        return init_tpl % {
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
            return CUDABaseTemplate.run_until_template["default"]

        # a condition has been defined, so we generate corresponding code
        cond_code = ""
        for pop in NetworkManager().get_network(self._net_id).get_populations():
            if pop.stop_condition:
                cond_code += CUDABaseTemplate.run_until_template["single_pop"] % {"id": pop.id}

        return CUDABaseTemplate.run_until_template["body"] % {"run_until": cond_code}

    def _cuda_kernel_config(self):
        """
        Each GPU kernel requires a launch configuration, established in
        the ANNarchyHost.cu code. Until ANNarchy 4.7 we always defined
        the configuration as pre-processor symbol. In context of variations
        required by different formats, we changed the strategy and this
        symbols are only used IF the user wants to overwrite something.

        * 192 threads for psp and synapse update
        * guessed amount of threads for neurons, based on population size
          (see _guess_pop_kernel_config)

        Notice:

            Only related to the CUDA implementation
        """
        if self._cuda_config is None:
            return ""

        # Population config adjust neuron_update
        configuration = "// Populations\n"
        for pop in NetworkManager().get_network(self._net_id).get_populations():
            if pop in self._cuda_config.keys():
                num_blocks = 1
                num_threads = 32

                if "num_threads" in self._cuda_config[pop].keys():
                    num_threads = self._cuda_config[pop]["num_threads"]
                    if "num_blocks" not in self._cuda_config[pop].keys():
                        num_blocks = int(ceil(float(pop.size) / float(num_threads)))

                if "num_blocks" in self._cuda_config[pop].keys():
                    num_blocks = self._cuda_config[pop]["num_blocks"]

                cfg = """#define __pop%(id)s_tpb__ %(nr)s
#define __pop%(id)s_nb__ %(nb)s
"""
                configuration += cfg % {
                    "id": pop.id,
                    "nr": num_threads,
                    "nb": num_blocks,
                }

                if ConfigManager().get("verbose", self._net_id):
                    print(
                        "population",
                        pop.id,
                        " - kernel config: (",
                        num_blocks,
                        ",",
                        num_threads,
                        ")",
                    )

        # Projection config - adjust psp, synapse_local_update, synapse_global_update
        configuration += "\n// Projections\n"
        for proj in NetworkManager().get_network(self._net_id).get_projections():
            if proj in self._cuda_config.keys():
                num_blocks = 1
                num_threads = 192

                if "num_threads" in self._cuda_config[proj].keys():
                    num_threads = self._cuda_config[proj]["num_threads"]
                if "num_blocks" in self._cuda_config[proj].keys():
                    num_blocks = self._cuda_config[proj]["num_blocks"]

                cfg = """#define __proj%(id_proj)s_%(target)s_tpb__ %(nr)s
#define __proj%(id_proj)s_%(target)s_nb__ %(nb)s
"""

                # proj.target can hold a single or multiple targets. We use
                # one configuration for all but need to define single names anyways
                target_list = (
                    proj.target if isinstance(proj.target, list) else [proj.target]
                )
                for target in target_list:
                    configuration += cfg % {
                        "id_proj": proj.id,
                        "target": target,
                        "nr": num_threads,
                        "nb": num_blocks,
                    }

                    if ConfigManager().get("verbose", self._net_id):
                        print(
                            "projection",
                            proj.id,
                            "with target",
                            target,
                            " - kernel config: (",
                            num_blocks,
                            ",",
                            num_threads,
                            ")",
                        )

        return configuration

    def _cuda_stream_config(self):
        """
        With Fermi Nvidia introduced multiple streams respectively concurrent
        kernel execution (requires device with compute compability > 2.x).

        Notice:

            Only related to the CUDA implementation
        """
        if self._cuda_config is None:
            pop_assign = "    // populations\n"
            proj_assign = "    // projections\n"
            max_number_streams = 0
        else:
            # TODO: maybe this should be a parameter too? As one could schedule multiple objects
            #       in one stream, the maximum number is not exploited
            max_number_streams = max(
                len(NetworkManager().get_network(self._net_id).get_populations()),
                len(NetworkManager().get_network(self._net_id).get_projections())
            )

            # HD:
            # the try-except blocks here are a REALLY lazy method.
            # TODO: it should be implemented more carefully in future
            pop_assign = "    // populations\n"
            for pop in NetworkManager().get_network(self._net_id).get_populations():
                try:
                    sid = self._cuda_config[pop]["stream"]
                    pop_assign += """    pop%(pid)s->stream = streams[%(sid)s];
""" % {"pid": pop.id, "sid": sid}
                except KeyError:
                    # default stream, if either no cuda_config at all or
                    # the population is not configured by user
                    pop_assign += """    pop%(pid)s->stream = 0;
""" % {"pid": pop.id}

            proj_assign = "    // projections\n"
            for proj in NetworkManager().get_network(self._net_id).get_projections():
                try:
                    sid = self._cuda_config[proj]["stream"]
                    proj_assign += """    proj%(pid)s->stream = streams[%(sid)s];
""" % {"pid": proj.id, "sid": sid}
                except KeyError:
                    # default stream, if either no cuda_config at all or
                    # the projection is not configured by user
                    proj_assign += """    proj%(pid)s->stream = 0;
""" % {"pid": proj.id}

        # Write config
        stream_config = CUDABaseTemplate.stream_setup % {
            "nbStreams": max_number_streams,
            "pop_assign": pop_assign,
            "proj_assign": proj_assign,
        }

        return stream_config

    def _guess_pop_kernel_config(self, pop):
        """
        Instead of a fixed amount of threads for each kernel, we try
        to guess a good configuration based on the population size.
        """
        max_tpb = 512
        warp_size = 32

        num_neur = pop.size / 2  # at least 2 iterations per thread
        guess = warp_size  # smallest block is 1 warp

        # Simplest case: we have more neurons than
        # available threads per block
        if num_neur > max_tpb:
            guess = max_tpb

        # check which is the closest possible thread amount
        pow_of_2 = [
            2**x for x in range(int(log(warp_size, 2)), int(log(max_tpb, 2)) + 1)
        ]
        for i in range(len(pow_of_2)):
            if pow_of_2[i] < num_neur:
                continue
            else:
                guess = pow_of_2[i]
                break

        return guess

    def _guess_proj_kernel_config(self, proj):
        """
        Instead of a fixed amount of threads for each kernel, we try
        to guess a good configuration based on the pre-synaptic population size.
        """
        max_tpb = 512
        warp_size = 32

        num_neur = proj.pre.size / 4  # at least 1/4 of the neurons are connected
        guess = warp_size  # smallest block is 1 warp

        # Simplest case: we have more neurons than
        # available threads per block
        if num_neur > max_tpb:
            guess = max_tpb

        # check which is the closest possible thread amount
        pow_of_2 = [
            2**x for x in range(int(log(warp_size, 2)), int(log(max_tpb, 2)) + 1)
        ]
        for i in range(len(pow_of_2)):
            if pow_of_2[i] < num_neur:
                continue
            else:
                guess = pow_of_2[i]
                break

        if ConfigManager().get("verbose", self._net_id):
            print("projection", proj.id, " - kernel size:", guess)

        return guess

    def _cuda_common_kernel(self, projections):
        """
        Some sparse matrix formats require additional functions. Which we need to
        define only once.
        """
        fmts = []

        for proj in projections:
            fmts.append(proj._storage_format)

        fmts = list(set(fmts))

        code = ""
        # TODO: generalize!
        if "csr" in fmts:
            from ANNarchy.generator.Projection.CUDA.CSR import (
                additional_global_functions,
            )

            code += additional_global_functions
        elif "csr_vector" in fmts:
            from ANNarchy.generator.Projection.CUDA.CSR_Vector import (
                additional_global_functions,
            )

            code += additional_global_functions

        return code
