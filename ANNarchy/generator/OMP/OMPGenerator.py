import ANNarchy.core.Global as Global
from ANNarchy.core.PopulationView import PopulationView
from .PopulationGenerator import PopulationGenerator
from .ProjectionGenerator import ProjectionGenerator

import numpy as np

class OMPGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

        self.popgen = PopulationGenerator()
        self.projgen = ProjectionGenerator()

    def generate(self):
        if Global.config['verbose']:
            print('\nGenerate code for OpenMP ...')

        # Propagte the global operations needed by the projections to the corresponding populations.
        self.propagate_global_ops()

        # Generate header code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchy.h', 'w') as ofile:
            ofile.write(self.generate_header())
            
        # Generate cpp code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchy.cpp', 'w') as ofile:
            ofile.write(self.generate_body())

        # Generate cython code for the analysed pops and projs
        with open(Global.annarchy_dir+'/generate/ANNarchyCore.pyx', 'w') as ofile:
            ofile.write(self.generate_pyx())

    def propagate_global_ops(self):

        # Analyse the populations
        for pop in self.populations:
            pop.global_operations = pop.neuron_type.description['global_operations']
            pop.delayed_variables = []

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
            if proj._synapses.max_delay > 1:
                for var in proj.synapse.description['dependencies']['pre']:
                    proj.pre.delayed_variables.append(var)

        # Make sure the operations are declared only once
        for pop in self.populations:
            pop.global_operations = list(np.unique(np.array(pop.global_operations)))
            pop.delayed_variables = list(set(pop.delayed_variables))



#######################################################################
############## HEADER #################################################
#######################################################################
    def generate_header(self):

        # struct declaration for each population
        pop_struct, pop_ptr = self.header_struct_pop()

        # struct declaration for each projection
        proj_struct, proj_ptr = self.header_struct_proj()

        # Custom functions
        custom_func = self.header_custom_functions()

        # Include OMP
        include_omp = "#include <omp.h>" if Global.config['num_threads'] > 1 else ""

        from .HeaderTemplate import header_template
        return header_template % {
            'pop_struct': pop_struct,
            'proj_struct': proj_struct,
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'custom_func': custom_func,
            'include_omp': include_omp
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

    def header_custom_functions(self):

        if len(Global._functions) == 0:
            return ""

        code = ""
        from ANNarchy.parser.Extraction import extract_functions
        for func in Global._functions:
            code +=  extract_functions(func, local_global=True)[0]['cpp'] + '\n'

        return code

#######################################################################
############## BODY ###################################################
#######################################################################
    def generate_body(self):
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
        glop_definition = self.body_def_glops()

        # Reset presynaptic sums
        reset_sums = self.body_resetcomputesum_pop()

        # Compute presynaptic sums
        compute_sums = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()
        rd_update_code = self.body_update_randomdistributions()

        # Initialize populations
        pop_init = self.body_init_population()

        # Initialize projections
        projection_init = self.body_init_projection()

        # Initialize global operations
        globalops_init = self.body_init_globalops()

        # Equations for the neural variables
        update_neuron = self.body_update_neuron()

        # Enque delayed outputs
        delay_code = self.body_delay_neuron()

        # Global operations
        update_globalops = self.body_update_globalops()

        # Equations for the synaptic variables
        update_synapse = self.body_update_synapse()

        # Equations for the post-events
        post_event = self.body_postevent_proj()

        # Structural plasticity
        structural_plasticity = self.body_structural_plasticity()

        # Record
        record = self.body_record()

        # Early stopping
        run_until = self.body_run_until()

        # Number threads
        number_threads = "omp_set_num_threads(threads);" if Global.config['num_threads'] > 1 else ""

        #Profiling
        from ..Profile.Template import profile_generator_omp_template
        prof_include = "" if not Global.config["profiling"] else profile_generator_omp_template['include']
        prof_init = "" if not Global.config["profiling"] else profile_generator_omp_template['init']
        prof_step_pre = "" if not Global.config["profiling"] else profile_generator_omp_template['step_pre']
        prof_run_pre = "" if not Global.config["profiling"] else profile_generator_omp_template['run_pre']
        prof_run_post = "" if not Global.config["profiling"] else profile_generator_omp_template['run_post']

        # Generate cpp code for the analysed pops and projs
        from .BodyTemplate import body_template
        return body_template % {
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'glops_def': glop_definition,
            'run_until': run_until,
            'compute_sums' : compute_sums,
            'reset_sums' : reset_sums,
            'update_neuron' : update_neuron,
            'update_globalops' : update_globalops,
            'update_synapse' : update_synapse,
            'random_dist_init' : rd_init_code,
            'random_dist_update' : rd_update_code,
            'delay_code' : delay_code,
            'pop_init' : pop_init,
            'projection_init' : projection_init,
            'globalops_init' : globalops_init,
            'post_event' : post_event,
            'structural_plasticity': structural_plasticity,
            'record' : record,
            'set_number_threads' : number_threads,
            'prof_include': prof_include,
            'prof_init': prof_init,
            'prof_step_pre': prof_step_pre,
            'prof_run_pre': prof_run_pre,
            'prof_run_post': prof_run_post
        }

    def body_update_neuron(self):
        code = ""
        for pop in self.populations:
            code  += self.popgen.update_neuron(pop)

        return code

    def body_delay_neuron(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.delay_code(pop)
        return code

    def body_computesum_proj(self):
        code = ""
        # Sum over all synapses 
        for proj in self.projections:
            # Is it a specific projection?
            if proj.generator['omp']['body_compute_psp']:
                code += proj.generator['omp']['body_compute_psp'] 
                continue
            # Call the right generator depending on type
            if proj.synapse.type == 'rate':
                code += self.projgen.computesum_rate(proj)
            else:
                code += self.projgen.computesum_spiking(proj)

        return code

    def body_resetcomputesum_pop(self):
        code = ""
        for pop in self.populations:
            if pop.neuron_type.type == 'rate':
                code += self.popgen.reset_computesum(pop)
        
        return code

    def body_postevent_proj(self):
        code = ""
        for proj in self.projections:
            if proj.synapse.type == 'spike':
                code += self.projgen.postevent(proj)

        return code


    def body_update_synapse(self):
        # Reset code
        code = ""
        # Iterate over all synapses 
        for proj in self.projections:
            code += self.projgen.update_synapse(proj)

        return code


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
    // Initialize random distribution objects
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
            return ""

        from .GlobalOperationTemplate import global_operation_templates
        code = ""
        for op in list(set(ops)):
            code += global_operation_templates[op]

        return code

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
        
    def body_update_randomdistributions(self):
        code = "" 
        for pop in self.populations:
            code += self.popgen.update_random_distributions(pop)

        for proj in self.projections:
            code += self.projgen.update_random_distributions(proj)

        return code

    def body_update_globalops(self):
        code = ""
        for pop in self.populations:
            code += self.popgen.update_globalops(pop)
        return code

    def body_record(self):
        code = ""
        # Populations
        for pop in self.populations:
           code += self.popgen.record(pop)

        # Projections
        for proj in self.projections:
           code += self.projgen.record(proj)

        return code

    def body_run_until(self):
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
