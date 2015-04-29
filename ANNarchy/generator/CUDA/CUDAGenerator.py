"""

    CUDAGenerator.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import ANNarchy.core.Global as Global
from ANNarchy.core.PopulationView import PopulationView
from .PopulationGenerator import PopulationGenerator
from .ProjectionGenerator import ProjectionGenerator

import numpy as np

# TODO: INTERFACE
#
#    as first step I only moved the current implementation to the extra generator classes, in a second step, the code
#    generation could be refined ...
#
#    in general, the ANNarchy.cpp would only contain the the call method instead of the full buisness logic,
#    as a consequence it could be enough to forward the calls to the update_neuron, update_synapse etc. functions.
#    The classes Population-/ProjectionGenerator could be responsible alone for the creation of cuANNarchy.cu / cuANNarchy.h

# TODO: VALIDATE
#
#    the marked functionality is ported towards CUDAGenerator, but need to checked carefully ...

class CUDAGenerator(object):

    def __init__(self, populations, projections):

        self.populations = populations
        self.projections = projections

        self.popgen = PopulationGenerator()
        self.projgen = ProjectionGenerator()

    def generate(self):

        if Global.config['verbose']:
            print('\nGenerate code for CUDA ...')

        # Propagte the global operations needed by the projections to the corresponding populations.
        self.propagate_global_ops()

        # Generate header code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchy.h', 'w') as ofile:
            ofile.write(self.generate_header())
            
        # Generate cpp and cuda code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchy.cu', 'w') as ofile:
            ofile.write(self.generate_body())

        # Generate cython code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchyCore.pyx', 'w') as ofile:
            ofile.write(self.generate_pyx())

    def propagate_global_ops(self):

        # Analyse the populations
        for pop in self.populations:
            pop.global_operations = pop.neuron_type.description['global_operations']

        # Propagate the global operations from the projections to the populations
        for proj in self.projections:
            for op in proj.synapse.description['pre_global_operations']:
                if isinstance(proj.pre, PopulationView):
                    if not op in proj.pre.population.global_operations:
                        proj.pre.population.global_operations.append(op)
                else:
                    if not op in proj.pre.global_operations:
                        proj.pre.global_operations.append(op)

            for op in  proj.synapse.description['post_global_operations']:
                if isinstance(proj.post, PopulationView):
                    if not op in proj.post.population.global_operations:
                        proj.post.population.global_operations.append(op)
                else:
                    if not op in proj.post.global_operations:
                        proj.post.global_operations.append(op)

        # Make sure the operations are declared only once
        for pop in self.populations:
            pop.global_operations = list(np.unique(np.array(pop.global_operations)))



#######################################################################
############## HEADER #################################################
#######################################################################
    def generate_header(self):

        # struct declaration for each population
        pop_struct, pop_ptr = self.header_struct_pop()

        # struct declaration for each projection
        proj_struct, proj_ptr = self.header_struct_proj()

        # Population recorders
        record_classes = self.header_recorder_classes()

        from .HeaderTemplate import header_template
        return header_template % {
            'pop_struct': pop_struct,
            'proj_struct': proj_struct,
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'record_classes': record_classes
        }

    def header_struct_pop(self):
        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""
        for pop in self.populations:
            # Header struct
            pop_struct += self.popgen.header_struct(pop)      
            # Extern pointer
            pop_ptr += """extern PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        return pop_struct, pop_ptr

    def header_struct_proj(self):
        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for proj in self.projections:
            # Header struct
            proj_struct += self.projgen.header_struct(proj)
            # Extern pointer
            proj_ptr += """extern ProjStruct%(id_proj)s proj%(id_proj)s;
"""% {'id_proj': proj.id}

        return proj_struct, proj_ptr

    def header_recorder_classes(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.recorder_class(pop)  
        for proj in self.projections:
            code += self.projgen.recorder_class(proj)  

        return code
        
#######################################################################
############## BODY ###################################################
#######################################################################
    def generate_body(self):
        """
        generate the ANNarchy.cpp file containing initialization codes, host_to_device- and device_to_host
        transfers and kernel call entities next to the files: cuANNarchy.cu (kernel implementation) and
        cuANNarchy.h (kernel prototypes).
        """
        # struct declaration for each population
        pop_ptr = ""
        for pop in self.populations:
            # Declaration of the structure
            pop_ptr += """PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        # struct declaration for each projection
        proj_ptr = ""
        for proj in self.projections:
            # Declaration of the structure
            proj_ptr += """ProjStruct%(id)s proj%(id)s;
"""% {'id': proj.id}

        # Code for the global operations
        glob_ops_header, glob_ops_body = self.body_def_glops()

        # Compute presynaptic sums
        compute_sums_body, compute_sums_header, compute_sums_call = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()

        # Initialize delayed arrays
        delay_init = self.body_init_delay()

        # Initialize populations
        pop_init = self.body_init_population()

        # Initialize projections
        projection_init = self.body_init_projection()

        # Initialize global operations
        globalops_init = self.body_init_globalops()

        # Equations for the neural variables
        update_neuron_body, update_neuron_header, update_neuron_call = self.body_update_neuron()

        # Enque delayed outputs
        delay_code = self.body_delay_neuron()

        # Global operations
        update_globalops = self.body_update_globalops()

        # Equations for the synaptic variables
        update_synapse_body, update_synapse_header, update_synapse_call = self.body_update_synapse()

        # Equations for the post-events
        post_event = self.body_postevent_proj()

        # Structural plasticity
        structural_plasticity = self.body_structural_plasticity()

        # Early stopping
        run_until = self.body_run_until()

        # determine number of threads per kernel and concurrent kernel execution
        threads_per_kernel, stream_setup = self.body_kernel_config()

        # Custom functions
        custom_func = self.body_custom_functions()

        # Initialize device ptr
        device_init = self.body_init_device()

        # host to device and device to host transfers
        host_device_transfer, device_host_transfer = self.body_memory_transfers()

        #TODO: Profiling

        # Generate cpp code for the analysed pops and projs
        from .BodyTemplate import body_template
        return body_template % {
            # host stuff
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'run_until': run_until,
            'compute_sums' : compute_sums_call,
            'update_neuron' : update_neuron_call,
            'update_globalops' : update_globalops,
            'update_synapse' : update_synapse_call,
            'random_dist_init' : rd_init_code,
            'delay_init' : delay_init,
            'delay_code' : delay_code,
            'spike_init' : pop_init,
            'projection_init' : projection_init,
            'globalops_init' : globalops_init,
            'post_event' : post_event,
            'structural_plasticity': structural_plasticity,
            'stream_setup': stream_setup,
            'device_init': device_init,
            'host_device_transfer': host_device_transfer,
            'device_host_transfer': device_host_transfer,
            'kernel_def': update_neuron_header + compute_sums_header + update_synapse_header,
            
            #device stuff
            'kernel_config': threads_per_kernel,
            'pop_kernel': update_neuron_body,
            'psp_kernel': compute_sums_body,
            'syn_kernel': update_synapse_body,
            'glob_ops_kernel': glob_ops_body,
            'custom_func': custom_func            
        }

    def body_update_neuron(self):
        # TODO: INTERFACE
        header = ""
        body = ""
        call = ""

        for pop in self.populations:
            pop_header, pop_body, pop_call = self.popgen.update_neuron(pop)
            
            header += pop_header
            body += pop_body
            call += pop_call

        return header, body, call

    def body_delay_neuron(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.delay_code(pop)
        return code

    def body_computesum_proj(self):
        # TODO: INTERFACE
        header = ""
        body = ""
        call = ""

        for proj in self.projections:
            # Call the right generator depending on type
            if proj.synapse.type == 'rate':
                b, h, c = self.projgen.computesum_rate(proj)
            else:
                b, h, c = self.projgen.computesum_spiking(proj)

            header += h
            body += b
            call += c

        return body, header, call

    def body_postevent_proj(self):
        code = ""
        for proj in self.projections:
            if proj.synapse.type == 'spike':
                code += self.projgen.postevent(proj)

        return code

    def body_update_synapse(self):
        # TODO: INTERFACE
        header = ""
        body = ""
        call = ""

        for proj in self.projections:
            b, h, c = self.projgen.update_synapse(proj)

            header += h
            body += b
            call += c

        return body, header, call

    def body_structural_plasticity(self):
        # Pruning if any
        pruning=""
        creating=""
        if Global.config['structural_plasticity'] :
            for proj in self.projections:
                if 'pruning' in proj.synapse.description.keys():
                    pruning += self.projgen.pruning(proj)
                if 'creating' in proj.synapse.description.keys():
                    creating += self.projgen.creating(proj)

        return creating + pruning

    def body_init_randomdistributions(self):
        code = """
    // Initialize cudaRNG states
"""
        for pop in self.populations:
            code += self.popgen.init_random_distributions(pop)

        for proj in self.projections:
            code += self.projgen.init_random_distributions(proj)

        return code

    def body_init_globalops(self):
        code = """
    // Initialize global operations
"""
        for pop in self.populations:
            code += self.popgen.init_globalops(pop)

        return code

    def body_def_glops(self):
        ops = []
        for pop in self.populations:
            for op in pop.global_operations:
                ops.append( op['function'] )

        if ops == []:
            return "", ""

        from .GlobalOperationTemplate import global_operation_templates
        header = ""
        body = ""

        for op in list(set(ops)):
            header += global_operation_templates[op]['header']
            body += global_operation_templates[op]['body']

        return header, body

    def body_init_delay(self):
        code = ""
        for pop in self.populations:
            if pop.max_delay > 1: # no need to generate the code otherwise
                code += self.popgen.init_delay(pop)

        return code

    def body_init_population(self):
        code = """
    // Initialize populations
"""
        for pop in self.populations:
            code += self.popgen.init_population(pop)

        return code

    def body_init_projection(self):
        code = """
    // Initialize projections
"""
        for proj in self.projections:
            code += self.projgen.init_projection(proj)

        return code
        
    def body_update_globalops(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.update_globalops(pop)
        return code

    def body_run_until(self):
        #TODO: VALIDATE

        # Check if it is useful to generate anything at all
        for pop in self.populations:
            if pop.stop_condition: # a condition has been defined
                break
        else:
            return """
    run(steps);
    return steps;
"""

        # Generate the conditional code
        complete_code = """
    bool stop = false;
    bool pop_stop = false;
    int nb = 0;
    for(int n = 0; n < steps; n++)
    {
        step();
        nb++;
        stop = or_and;
        for(int i=0; i<populations.size();i++)
        {
            // Check all populations
            switch(populations[i]){
%(run_until)s
            }

            // Accumulate the results
            if(or_and)
                stop = stop && pop_stop;
            else
                stop = stop || pop_stop;
        }
        if(stop)
            break;
    }
    return nb;

"""
        code = ""
        for pop in self.populations:
            code += self.popgen.stop_condition(pop)

        return complete_code % {'run_until': code}

    def body_custom_functions(self):
        """
        ATTENTION: the same as OMPGenerator.header_custom_func
        """
        if len(Global._functions) == 0:
            return ""

        Global._error("Not implemented yet: custom functions for GPGPU kernel ...")
        return code


#######################################################################
############## PYX ####################################################
#######################################################################
    def generate_pyx(self):
        # struct declaration for each population
        pop_struct, pop_ptr = self.pyx_struct_pop()

        # struct declaration for each projection
        proj_struct, proj_ptr = self.pyx_struct_proj()

        # Cython wrappers for the populations
        pop_class = self.pyx_wrapper_pop()

        # Cython wrappers for the projections
        proj_class = self.pyx_wrapper_proj()


        from .PyxTemplate import pyx_template
        return pyx_template % {
            'pop_struct': pop_struct, 'pop_ptr': pop_ptr,
            'proj_struct': proj_struct, 'proj_ptr': proj_ptr,
            'pop_class' : pop_class, 'proj_class': proj_class
        }

    def pyx_struct_pop(self):
        pop_struct = ""
        pop_ptr = ""
        for pop in self.populations:
            # Header export
            pop_struct += self.popgen.pyx_struct(pop)
            # Population instance
            pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {
    'id': pop.id,
}
        return pop_struct, pop_ptr

    def pyx_struct_proj(self):
        proj_struct = ""
        proj_ptr = ""
        for proj in self.projections:
            # Header export
            proj_struct += self.projgen.pyx_struct(proj)

            # Projection instance
            proj_ptr += """
    ProjStruct%(id_proj)s proj%(id_proj)s"""% {
    'id_proj': proj.id,
}
        return proj_struct, proj_ptr

    def pyx_wrapper_pop(self):
        # Cython wrappers for the populations
        code = ""
        for pop in self.populations:
            code += self.popgen.pyx_wrapper(pop)
        return code

    def pyx_wrapper_proj(self):
        # Cython wrappers for the projections
        code = ""
        for proj in self.projections:
            code += self.projgen.pyx_wrapper(proj)
        return code


#######################################################################
############## HOST - DEVICE ##########################################
#######################################################################
    def body_memory_transfers(self):
        host_device_transfer = ""
        device_host_transfer = ""

        # transfers for populations
        for pop in self.populations:
            host_device_transfer += """
    // host to device transfers for %(pop_name)s""" % { 'pop_name': pop.name }
            for attr in pop.neuron_type.description['parameters']+pop.neuron_type.description['variables']:
                if attr['name'] in pop.neuron_type.description['local']:
                    host_device_transfer += """
        // %(attr_name)s: local
        if( pop%(id)s.%(attr_name)s_dirty )
        {
            //std::cout << "Transfer pop%(id)s.%(attr_name)s" << std::endl;
            cudaMemcpy(pop%(id)s.gpu_%(attr_name)s, pop%(id)s.%(attr_name)s.data(), pop%(id)s.size * sizeof(%(type)s), cudaMemcpyHostToDevice);
            pop%(id)s.%(attr_name)s_dirty = false;
        }
""" % { 'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype'] }

            device_host_transfer += """
    // device to host transfers for %(pop_name)s\n""" % { 'pop_name': pop.name }
            for attr in pop.neuron_type.description['parameters']+pop.neuron_type.description['variables']:
                if attr['name'] in pop.neuron_type.description['local']:
                    device_host_transfer += """\tcudaMemcpy(pop%(id)s.%(attr_name)s.data(), pop%(id)s.gpu_%(attr_name)s, pop%(id)s.size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
""" % { 'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype'] }

        # transfers for projections
        for proj in self.projections:
            host_device_transfer += """\n    // host to device transfers for proj%(id)s\n""" % { 'id': proj.id }
            for attr in proj.synapse.description['parameters']+proj.synapse.description['variables']:
                if attr['name'] in proj.synapse.description['local']:
                    host_device_transfer += """
        // %(name)s: local
        if ( proj%(id)s.%(name)s_dirty )
        {
            std::vector<double> flat_proj%(id)s_%(name)s = flattenArray<double>(proj%(id)s.%(name)s);
            cudaMemcpy(proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.data(), flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s), cudaMemcpyHostToDevice);
            flat_proj%(id)s_%(name)s.clear();
        }
""" % { 'id': proj.id, 'name': attr['name'], 'type': attr['ctype'] }
                else:
                    host_device_transfer += """
        // %(name)s: global
        if ( proj%(id)s.%(name)s_dirty )
        {
            cudaMemcpy(proj%(id)s.gpu_%(name)s, proj%(id)s.%(name)s.data(), pop%(post)s.size * sizeof(%(type)s), cudaMemcpyHostToDevice);
        }
""" % { 'id': proj.id, 'post': proj.post.id, 'name': attr['name'], 'type': attr['ctype'] }

            device_host_transfer += """
    // device to host transfers for proj%(id)s\n""" % { 'id': proj.id }
            for attr in proj.synapse.description['parameters']+proj.synapse.description['variables']:
                if attr['name'] in proj.synapse.description['local']:
                    device_host_transfer += """
            // %(name)s: local
            std::vector<%(type)s> flat_proj%(id)s_%(name)s = std::vector<%(type)s>(proj%(id)s.overallSynapses, 0);
            cudaMemcpy(flat_proj%(id)s_%(name)s.data(), proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s), cudaMemcpyDeviceToHost);
            proj%(id)s.%(name)s = deFlattenArray<%(type)s>(flat_proj%(id)s_%(name)s, proj%(id)s.flat_idx);
            flat_proj%(id)s_%(name)s.clear();
""" % { 'id': proj.id, 'name': attr['name'], 'type': attr['ctype'] }
                else:
                    device_host_transfer += """
            // %(name)s: global
            cudaMemcpy( proj%(id)s.%(name)s.data(), proj%(id)s.gpu_%(name)s, pop%(post)s.size * sizeof(%(type)s), cudaMemcpyDeviceToHost);
""" % { 'id': proj.id, 'post': proj.post.id, 'name': attr['name'], 'type': attr['ctype'] }

        return host_device_transfer, device_host_transfer

    def body_init_device(self):
        code = ""

        dev_id = 0 # default cuda device
        if 'device' in Global.cuda_config.keys():
            dev_id = Global.cuda_config['device']

        code += """
    // set active cuda device
    cudaError_t status = cudaSetDevice(%(id)s);
    if ( status != cudaSuccess )
        std::cerr << "Error on setting cuda device ... " << std::endl;

    // initialize cuda-api
    cudaFree(0);
""" % { 'id': dev_id }

        for pop in self.populations:
            code += """\n\t// Initialize device memory for %(pop_name)s\n""" % { 'pop_name': pop.name }
            for attr in pop.neuron_type.description['parameters']+pop.neuron_type.description['variables']:
                if attr['name'] in pop.neuron_type.description['local']:
                    code += """\tcudaMalloc((void**)&pop%(id)s.gpu_%(attr_name)s, pop%(id)s.size * sizeof(%(type)s));
        pop%(id)s.%(attr_name)s_dirty = true;
""" % { 'id': pop.id, 'attr_name': attr['name'], 'type': attr['ctype'] }
            for target in pop.neuron_type.description['targets']:
                code += """\tcudaMalloc((void**)&pop%(id)s.gpu_sum_%(target)s, pop%(id)s.size * sizeof(double));
""" % { 'id': pop.id, 'target': target }

        for proj in self.projections:
            from .cuBodyTemplate import proj_basic_data

            # basic variables: post_rank, nb_synapses, off_synapses, pre_rank
            code += proj_basic_data % { 'id': proj.id }

            # other variables, parameters
            for attr in proj.synapse.description['parameters']+proj.synapse.description['variables']:
                if attr['name'] in proj.synapse.description['local']:
                    code += """
        // %(name)s
        std::vector<double> flat_proj%(id)s_%(name)s = flattenArray<double>(proj%(id)s.%(name)s);
        cudaMalloc((void**)&proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s));
        cudaMemcpy(proj%(id)s.gpu_%(name)s, flat_proj%(id)s_%(name)s.data(), flat_proj%(id)s_%(name)s.size() * sizeof(%(type)s), cudaMemcpyHostToDevice);
        flat_proj%(id)s_%(name)s.clear();
        proj%(id)s.%(name)s_dirty = false;
""" % { 'id': proj.id, 'name': attr['name'], 'type': attr['ctype'] }
                else:
                    code += """
        // %(name)s
        cudaMalloc((void**)&proj%(id)s.gpu_%(name)s, pop%(post)s.size * sizeof(%(type)s));
        cudaMemcpy(proj%(id)s.gpu_%(name)s, proj%(id)s.%(name)s.data(), pop%(post)s.size * sizeof(%(type)s), cudaMemcpyHostToDevice);
        proj%(id)s.%(name)s_dirty = false;
""" % { 'id': proj.id, 'post': proj.post.id, 'name': attr['name'], 'type': attr['ctype'] }

        return code

    def body_kernel_config(self):
        cu_config = Global.cuda_config

        code = "// Population config\n"
        for pop in self.populations:
            num_threads = 32
            if pop in cu_config.keys():
                num_threads = cu_config[pop]['num_threads']

            code+= """#define __pop%(id)s__ %(nr)s\n""" % { 'id': pop.id, 'nr': num_threads }

        code += "\n// Population config\n"
        for proj in self.projections:
            num_threads = 192
            if proj in cu_config.keys():
                num_threads = cu_config[proj]['num_threads']

            code+= """#define __pop%(pre)s_pop%(post)s_%(target)s__ %(nr)s\n""" % { 'pre': proj.pre.id, 'post': proj.post.id, 'target': proj.target, 'nr': num_threads }

        pop_assign = "    // populations\n"
        for pop in self.populations:
            if pop in Global.cuda_config.keys():
                pop_assign += """    pop%(pid)s.stream = streams[%(sid)s];
""" % {'pid': pop.id, 'sid': Global.cuda_config[pop]['stream'] }
            else:
                # default stream
                pop_assign += """    pop%(pid)s.stream = 0;
""" % {'pid': pop.id }

        proj_assign = "    // populations\n"
        for proj in self.projections:
            if proj in Global.cuda_config.keys():
                proj_assign += """    proj%(pid)s.stream = streams[%(sid)s];
""" % {'pid': proj.id, 'sid': Global.cuda_config[proj]['stream'] }
            else:
                # default stream
                proj_assign += """    proj%(pid)s.stream = 0;
""" % {'pid': proj.id }

        from .cuBodyTemplate import stream_setup
        stream_config = stream_setup % {
            'nbStreams': 2,
            'pop_assign': pop_assign,
            'proj_assign': proj_assign
        }

        return code, stream_config
