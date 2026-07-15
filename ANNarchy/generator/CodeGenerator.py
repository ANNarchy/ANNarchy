"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import time

from ANNarchy.core.PopulationView import PopulationView
from ANNarchy.intern.ConfigManagement import ConfigManager, _check_paradigm
from ANNarchy.intern.NetworkManager import NetworkManager

from ANNarchy.intern import Messages

from ANNarchy.generator.NanoBind.Generator import NanoBindGenerator
from ANNarchy.generator.Monitor.MonitorGenerator import MonitorGenerator
from ANNarchy.generator.SimCore import (
    SingleThreadSimCoreGenerator,
    OpenMPSimCoreGenerator,
    CUDASimCoreGenerator
)
from ANNarchy.generator.Population import (
    SingleThreadGenerator,
    OpenMPGenerator,
    CUDAGenerator,
)
from ANNarchy.generator.Projection import (
    SingleThreadProjectionGenerator,
    OpenMPProjectionGenerator,
    CUDAProjectionGenerator,
)
from ANNarchy.generator.Template.GlobalOperationTemplate import (
    global_operation_templates_st,
    global_operation_templates_openmp,
    global_operation_templates_cuda,
)
from ANNarchy.generator.Template import SingleThreadBaseTemplate, OpenMPBaseTemplate, CUDABaseTemplate
from ANNarchy.generator import Profile


class CodeGenerator(object):
    """
    The CodeGenerator class is responsible to control the code
    generation process.

    Unil now, it implements the code generation for OpenMP
    (including sequential) and CUDA.The decision whether as
    OpenMP or sequential code is dependent on the number of
    threads.
    """

    def __init__(self, annarchy_dir, net_id, cuda_config):
        """
        Constructor initializes the PopulationGenerator and ProjectionGenerator
        class and stores the provided information for later use.

        Parameters:

            * *net_id*: unique id for the current network
            * *annarchy_dir*: unique target directory for the generated code
              files; they are stored in 'generate' sub-folder
            * *cuda_config*: configuration dict for cuda. check the method
              _cuda_kernel_config for more details.
        """
        self.net_id = net_id
        self._network = NetworkManager().get_network(self.net_id)
        self._annarchy_dir = annarchy_dir
        self._populations = self._network.get_populations()
        self._projections = self._network.get_projections()
        self._cuda_config = cuda_config

        # Profiling is optional, but if either Global.config["profiling"] set to True
        # or --profile was added on command line.
        if self._network._profiler is not None:
            if ConfigManager().get("paradigm", self.net_id) == "openmp":
                self._profgen = Profile.CPP11Profile(self._annarchy_dir, net_id)
                self._profgen.generate()
            elif ConfigManager().get("paradigm", self.net_id) == "cuda":
                self._profgen = Profile.CUDAProfile(self._annarchy_dir, net_id)
                self._profgen.generate()
            else:
                paradigm = ConfigManager().get("paradigm", self.net_id)
                Messages.error(
                    f"No ProfileGenerator available for paradigm='{paradigm}'"
                )
        else:
            self._profgen = None

        # Instantiate code generator based on the target platform
        if ConfigManager().get("paradigm", self.net_id) == "openmp":
            if ConfigManager().get("num_threads", self.net_id) == 1:
                self._popgen = SingleThreadGenerator(self._profgen, net_id)
                self._projgen = SingleThreadProjectionGenerator(self._profgen, net_id)
                self._core_gen = SingleThreadSimCoreGenerator(self._profgen, net_id)
            else:
                self._popgen = OpenMPGenerator(self._profgen, net_id)
                self._projgen = OpenMPProjectionGenerator(self._profgen, net_id)
                self._core_gen = OpenMPSimCoreGenerator(self._profgen, net_id)

        elif ConfigManager().get("paradigm", self.net_id) == "cuda":
            self._popgen = CUDAGenerator(
                self._cuda_config["cuda_version"], self._profgen, net_id
            )
            self._projgen = CUDAProjectionGenerator(
                self._cuda_config["cuda_version"], self._profgen, net_id
            )
            self._core_gen = CUDASimCoreGenerator(
                self._cuda_config, self._profgen, net_id
            )

        else:
            paradigm = ConfigManager().get("paradigm", self.net_id)
            Messages.error(f"No PopulationGenerator for paradigm='{paradigm}'")

        # Py-extenstion and RecordGenerator are commonly defined
        self._nb_gen = NanoBindGenerator(annarchy_dir, net_id)
        self._recordgen = MonitorGenerator(annarchy_dir, net_id)

        # Target container for the generated code snippets
        self._pop_desc = []
        self._proj_desc = []

    def generate(self):
        """
        Generate code files and store them in target directory (located at
        self._annarchy_dir/generate). More detailed the following files are
        generated, by this class:

            * *ANNarchyCore[net_id].cpp*: contains main simulation loop, object instantiation
            * *ANNarchyCore[net_id].hpp*: header file for simulation core
            * *ANNarchyWrapper[net_id].cpp*: nanobind interface file, gathering all
              functions/ objects, which should be accessible from Python
            * for each population a seperate header file, contain semantic
              logic of a population respectively neuron object (filename:
              pop<id>).
            * for each projection a seperate header file, contain semantic
              logic of a projection respectively synapse object (filename:
              proj<id>)
        """
        if self._network._profiler is not None:
            t0 = time.time()

        if ConfigManager().get("verbose", self.net_id):
            if ConfigManager().get("paradigm", self.net_id) == "openmp":
                if ConfigManager().get("num_threads", self.net_id) > 1:
                    print("\nGenerate code for OpenMP ...")
                else:
                    print("\nGenerate sequential code ...")
            elif ConfigManager().get("paradigm", self.net_id) == "cuda":
                print("\nGenerate CUDA code ...")
            else:
                raise NotImplementedError

        # Specific populations/projections have an overwritten _generate()
        # method which will populate the self._specific_template dictionary
        for pop in self._populations:
            pop._generate()
        for proj in self._projections:
            proj._generate()

        # Propagate the global operations needed by the projections to the
        # corresponding populations.
        self._propagate_global_ops()

        # Create all populations
        for pop in self._populations:
            self._pop_desc.append(self._popgen.header_struct(pop, self._annarchy_dir))

        # Create all projections
        for proj in self._projections:
            self._proj_desc.append(
                self._projgen.header_struct(proj, self._annarchy_dir)
            )

        # where all source files should take place
        source_dest = self._annarchy_dir + "/generate/net" + str(self.net_id) + "/"

        # Generate SimCore-related files
        self._core_gen.generate_header(
            source_dest, self._pop_desc, self._proj_desc
        )
        self._core_gen.generate_body(
            self._annarchy_dir, source_dest, self._profgen, self._popgen, self._pop_desc, self._proj_desc
        )

        # Generate monitor code for the analysed pops and projs
        self._recordgen.generate()

        # Generate nanobind wrapper code for the analysed pops and projs, as well as the simulation control
        with open(
            source_dest + "ANNarchyWrapper" + str(self.net_id) + ".cpp", "w", encoding='utf-8'
        ) as ofile:
            ofile.write(self._nb_gen.generate())

        self._generate_file_overview(source_dest)

        if self._network._profiler is not None:
            t1 = time.time()
            self._network._profiler.add_entry(t0, t1, "generate", "compile")

    def _generate_file_overview(self, source_dest):
        """
        Generate a logfile, where we log which Population/Projection object is stored in
        which file.

        Parameters:

        * source_dest: path to folder where generated files are stored.
        """
        pop_desc = """pop%(id_pop)s, %(type_pop)s(name ='%(name_pop)s', neuron='%(neuron_type)s')\n"""
        proj_desc = """proj%(id_proj)s, %(type_proj)s(pre='%(pre_name)s', post='%(post_name)s', target='%(target)s', synapse='%(synapse_type)s', name='%(name)s') using connector: %(pattern)s \n"""

        # Equal to target path in CodeGenerator.generate()
        with open(source_dest + "codegen.log", "w", encoding="utf-8") as ofile:
            ofile.write("Filename, Object Description\n")
            for pop in self._populations:
                pop_type = type(pop).__name__
                desc_dict = {
                    "id_pop": pop.id,
                    "name_pop": pop.name,
                    "neuron_type": pop.neuron_type.name,
                    "type_pop": pop_type,
                }
                ofile.write(pop_desc % desc_dict)

            for proj in self._projections:
                proj_type = type(proj).__name__
                desc_dict = {
                    "id_proj": proj.id,
                    "type_proj": proj_type,
                    "pre_name": proj.pre.name,
                    "post_name": proj.post.name,
                    "target": proj.target,
                    "synapse_type": proj.synapse_type.name,
                    "name": proj.name,
                }

                # In case of debug, we print the parameters otherwise not
                if ConfigManager().get("debug", self.net_id):
                    desc_dict.update({"pattern": proj.connector_description})
                else:
                    desc_dict.update(
                        {"pattern": proj.connector_description.split(",")[0]}
                    )

                ofile.write(proj_desc % desc_dict)

    def _propagate_global_ops(self):
        """
        The parser analyses the synapse and neuron definitions and
        store if global operations like min, max or mean are necessary.

        Furthermore for synapses accesses to population variales (e. g. pre.r)
        occure. In this case we need to generate special codes in the PopulationGenerator.
        """
        # Analyse the populations
        for pop in self._populations:
            pop.global_operations = pop.neuron_type.description["global_operations"]
            pop.delayed_variables = []

        # Propagate the global operations from the projections to the populations
        for proj in self._projections:
            for op in proj.synapse_type.description["pre_global_operations"]:
                if isinstance(proj.pre, PopulationView):
                    if not op in proj.pre.population.global_operations:
                        proj.pre.population.global_operations.append(op)
                else:
                    if not op in proj.pre.global_operations:
                        proj.pre.global_operations.append(op)

            for op in proj.synapse_type.description["post_global_operations"]:
                if isinstance(proj.post, PopulationView):
                    if not op in proj.post.population.global_operations:
                        proj.post.population.global_operations.append(op)
                else:
                    if not op in proj.post.global_operations:
                        proj.post.global_operations.append(op)

            if proj.max_delay > 1:
                for var in proj.synapse_type.description["dependencies"]["pre"]:
                    if isinstance(proj.pre, PopulationView):
                        proj.pre.population.delayed_variables.append(var)
                    else:
                        proj.pre.delayed_variables.append(var)

        # Make sure the operations are declared only once
        for pop in self._populations:
            pop.global_operations = [
                dict(y)
                for y in sorted(set(tuple(x.items())) for x in pop.global_operations)
            ]
            pop.delayed_variables = sorted(list(set(pop.delayed_variables)))
