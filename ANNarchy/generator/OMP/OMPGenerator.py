import ANNarchy.core.Global as Global

import numpy as np

class OMPGenerator(object):

    def __init__(self, populations, projections):
        
        self.populations = populations
        self.projections = projections

    def generate(self):

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

        for name, proj in self.projections.iteritems():
            for op in proj.synapse.description['pre_global_operations']:
                proj.pre.global_operations.append(op)
            for op in  proj.synapse.description['post_global_operations']:
                proj.post.global_operations.append(op)

    def generate_header(self):

        # struct declaration for each population
        pop_struct, pop_ptr = self.header_struct_pop()

        # struct declaration for each projection
        proj_struct, proj_ptr = self.header_struct_proj()

        from .HeaderTemplate import header_template
        return header_template % {
            'pop_struct': pop_struct,
            'proj_struct': proj_struct,
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr
        }

    def generate_body(self):

        # struct declaration for each population
        pop_ptr = ""
        for name, pop in self.populations.iteritems():
            # Declaration of the structure
            pop_ptr += """
PopStruct%(id)s pop%(id)s;
"""% {'id': pop.id}

        # struct declaration for each projection
        proj_ptr = ""
        for name, proj in self.projections.iteritems():
            # Declaration of the structure
            proj_ptr += """
ProjStruct%(id)s proj%(id)s;
"""% {'id': proj.id}

        # Compute presynaptic sums
        compute_sums = self.body_computesum_proj()

        # Initialize random distributions
        rd_init_code = self.body_init_randomdistributions()
        rd_update_code = self.body_update_randomdistributions()


        # Equations for the neural variables
        update_neuron = self.body_update_neuron()

        from .BodyTemplate import body_template
        return body_template % {
            'pop_ptr': pop_ptr,
            'proj_ptr': proj_ptr,
            'compute_sums' : compute_sums,
            'update_neuron' : update_neuron,
            'random_dist_init' : rd_init_code,
            'random_dist_update' : rd_update_code
        }

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


#######################################################################
############## HEADER #################################################
#######################################################################

    def header_struct_pop(self):
        # struct declaration for each population
        pop_struct = ""
        pop_ptr = ""
        for name, pop in self.populations.iteritems():
            code = """
struct PopStruct%(id)s{
    int size;
"""
            # Parameters
            for var in pop.neuron.description['parameters']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in pop.neuron.description['variables']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
    // Global variable %(name)s
    %(type)s  %(name)s ;
    std::vector< %(type)s > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Arrays for the presynaptic sums
            code += """
    // Targets
"""
            for target in pop.neuron.description['targets']:
                code += """    std::vector<double> sum_%(target)s;
""" % {'target' : target}

            # Arrays for the random numbers
            code += """
    // Random numbers
"""
            for rd in pop.neuron.description['random_distributions']:
                code += """    std::vector<double> %(rd_name)s;
    std::uniform_real_distribution<double> dist_%(rd_name)s;
""" % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}


            code += """
};    
""" 
            pop_struct += code % {'id': pop.id}

            pop_ptr += """
extern PopStruct%(id)s pop%(id)s;
"""% {
    'id': pop.id,
}

        return pop_struct, pop_ptr

    def header_struct_proj(self):
        # struct declaration for each projection
        proj_struct = ""
        proj_ptr = ""
        for name, proj in self.projections.iteritems():
            code = """
struct ProjStruct%(id)s{
    int size;
    std::vector<int> post_rank ;
    std::vector< std::vector< int > > pre_rank ;
    std::vector< std::vector< int > > delay ;
"""
            # Parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    // Local parameter %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    //std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    //bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
    // Global variable %(name)s
    %(type)s  %(name)s ;
    //std::vector< %(type)s > recorded_%(name)s ;
    //bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            code += """
};    
""" 
            proj_struct += code % {'id': proj.id}

            proj_ptr += """
extern ProjStruct%(id)s proj%(id)s;
"""% {
    'id': proj.id,
}

        return proj_struct, proj_ptr


#######################################################################
############## BODY ###################################################
#######################################################################

    def body_update_neuron(self):
        update_neuron = ""
        for name, pop in self.populations.iteritems():
            # Neural update
            from ..Utils import generate_equation_code
            eqs = generate_equation_code(pop.id, pop.neuron.description, 'global') % {'pop': 'pop' + str(pop.id)}
            update_neuron += """
    // Updating the global variables of population %(id)s
%(eqs)s
""" % {'id': pop.id, 'eqs': eqs}
            eqs = generate_equation_code(pop.id, pop.neuron.description, 'local') % {'pop': 'pop' + str(pop.id)}
            update_neuron += """
    // Updating the local variables of population %(id)s
    //start = omp_get_wtime();
    #pragma omp for
    for(int i = 0; i < pop%(id)s.size; i++){
%(eqs)s
    }
    //std::cout << "Updating pop%(id)s took " << (omp_get_wtime() - start) << std::endl;
""" % {'id': pop.id, 'eqs': eqs}
        return update_neuron

    def body_computesum_proj(self):
        # Reset code
        code = """
    // Rate-coded: reset the sum arrays to 0.0"""
        for name, pop in self.populations.iteritems():
            if pop.neuron.type=='rate':
                for target in pop.targets:
                    code += """    
    //memset( pop%(id)s.sum_%(target)s.data(), 0, pop%(id)s.sum_%(target)s.size() * sizeof(double));
""" %{'id' : pop.id, 'target': target}

        # Sum over all synapses 
        for name, proj in self.projections.iteritems():
            code+= """
    // proj%(id_proj)s: pop%(id_pre)s -> pop%(id_post)s with target %(target)s
    //start = omp_get_wtime();
    #pragma omp parallel for private(sum)
    for(int i = 0; i < proj%(id_proj)s.post_rank.size(); i++){
        sum = 0.0;
        for(int j = 0; j < proj%(id_proj)s.pre_rank[i].size(); j++){
            sum += proj%(id_proj)s.w[i][j] * pop%(id_pre)s.r[proj%(id_proj)s.pre_rank[i][j]];
        }
        pop%(id_post)s.sum_%(target)s[proj%(id_proj)s.post_rank[i]] = sum;
    }
    //std::cout << "Compute_sum of proj %(id_proj)s took " << (omp_get_wtime() - start) << std::endl;
"""%{'id_proj' : proj.id, 'target': proj.target, 'id_post': proj.post.id, 'id_pre': proj.pre.id}

        return code

    def body_init_randomdistributions(self):
        code = """
    // Initialize random distribution objects
"""
        for name, pop in self.populations.iteritems():
            for rd in pop.neuron.description['random_distributions']:
                code += """    pop%(id)s.%(rd_name)s = std::vector<double>(pop%(id)s.size, 0.0);
    pop%(id)s.dist_%(rd_name)s = %(rd_init)s;
""" % {'id': pop.id, 'rd_name': rd['name'], 'rd_init': rd['definition']}

        return code

    def body_update_randomdistributions(self):
        code = """
    // Compute random distributions""" 
        for name, pop in self.populations.iteritems():
            if len(pop.neuron.description['random_distributions']) > 0:
                code += """
    // RD of pop%(id)s
    #pragma omp parallel for
    for(int i = 0; i < pop%(id)s.size; i++)
    {
"""% {'id': pop.id}
                for rd in pop.neuron.description['random_distributions']:
                    code += """
        pop%(id)s.%(rd_name)s[i] = pop%(id)s.dist_%(rd_name)s(rng[omp_get_thread_num()]);
""" % {'id': pop.id, 'rd_name': rd['name']}

                code += """
    }
"""
        return code


#######################################################################
############## PYX ####################################################
#######################################################################
    def pyx_struct_pop(self):
        pop_struct = ""
        pop_ptr = ""
        for name, pop in self.populations.iteritems():
            code = """
    cdef struct PopStruct%(id)s :
        int size
"""
            # Parameters
            for var in pop.neuron.description['parameters']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
        # Local parameter %(name)s
        vector[%(type)s] %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
        # Global parameter %(name)s
        %(type)s  %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in pop.neuron.description['variables']:
                if var['name'] in pop.neuron.description['local']:
                    code += """
        # Local variable %(name)s
        vector[%(type)s] %(name)s 
        vector[vector[%(type)s]] recorded_%(name)s 
        bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in pop.neuron.description['global']:
                    code += """
        # Global variable %(name)s
        %(type)s  %(name)s 
        vector[%(type)s] recorded_%(name)s
        bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Arrays for the presynaptic sums
            code += """
        # Targets
"""
            for target in pop.neuron.description['targets']:
                code += """        vector[double] sum_%(target)s
""" % {'target' : target}

            # Finalize the code
            pop_struct += code % {'id': pop.id}

            # Population instance
            pop_ptr += """
    PopStruct%(id)s pop%(id)s"""% {
    'id': pop.id,
}
        return pop_struct, pop_ptr

    def pyx_struct_proj(self):
        proj_struct = ""
        proj_ptr = ""
        for name, proj in self.projections.iteritems():
            code = """
    cdef struct ProjStruct%(id)s :
        int size
        vector[int] post_rank
        vector[vector[int]] pre_rank
        vector[vector[int]] delay
"""
            # Parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
        # Local parameter %(name)s
        vector[vector[%(type)s]] %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global parameter %(name)s
        %(type)s  %(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    code += """
        # Local variable %(name)s
        vector[vector[%(type)s]] %(name)s 
        #vector[vector[%(type)s]] recorded_%(name)s 
        #bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}
                elif var['name'] in proj.synapse.description['global']:
                    code += """
        # Global variable %(name)s
        %(type)s  %(name)s 
        #vector[%(type)s] recorded_%(name)s
        #bool record_%(name)s 
""" % {'type' : var['ctype'], 'name': var['name']}


            # Finalize the code
            proj_struct += code % {'id': proj.id}

            # Population instance
            proj_ptr += """
    ProjStruct%(id)s proj%(id)s"""% {
    'id': proj.id,
}
        return proj_struct, proj_ptr

    def pyx_wrapper_pop(self):
        # Cython wrappers for the populations
        pop_class = ""
        for name, pop in self.populations.iteritems():
            # Init
            pop_class += """
cdef class pop%(id)s_wrapper :

    def __cinit__(self, size):
        pop%(id)s.size = size"""% {'id': pop.id}
            for var in pop.neuron.description['parameters']:
                if var['name'] in pop.neuron.description['local']:
                    if isinstance(var['init'], np.ndarray):
                        init = 0.0 if var['ctype'] == 'double' else 0
                    else:
                        init = var['init']
                    pop_class += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            for var in pop.neuron.description['variables']:
                if var['name'] in pop.neuron.description['local']:
                    if isinstance(var['init'], np.ndarray):
                        init = 0.0 if var['ctype'] == 'double' else 0
                    else:
                        init = var['init']
                    pop_class += """
        pop%(id)s.%(name)s = vector[%(type)s](size, %(init)s)""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            for target in pop.neuron.description['targets']:
                pop_class += """
        pop%(id)s.sum_%(target)s = vector[double](size, 0.0)""" %{'id': pop.id, 'target': target}

            # Size property
            pop_class += """

    property size:
        def __get__(self):
            return pop%(id)s.size
""" % {'id': pop.id}

            # Parameters
            for var in pop.neuron.description['parameters']:
                if var['name'] in pop.neuron.description['local']:
                    pop_class += """
    # Local parameter %(name)s
    property %(name)s:
        def __get__(self):
            return pop%(id)s.%(name)s
        def __set__(self, value):
            pop%(id)s.%(name)s = value
    def get_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    def set_%(name)s(self, int rank, %(type)s value):
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}
                elif var['name'] in pop.neuron.description['global']:
                    pop_class += """
    # Global parameter %(name)s
    property %(name)s:
        def __get__(self):
            return pop%(id)s.%(name)s
        def __set__(self, value):
            pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name']}

            # Variables
            for var in pop.neuron.description['variables']:
                if var['name'] in pop.neuron.description['local']:
                    pop_class += """
    # Local variable %(name)s
    property %(name)s:
        def __get__(self):
            return pop%(id)s.%(name)s
        def __set__(self, value):
            pop%(id)s.%(name)s = value
    def get_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    def set_%(name)s(self, int rank, %(type)s value):
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}
                elif var['name'] in pop.neuron.description['global']:
                    pop_class += """
    # Global variable %(name)s
    property %(name)s:
        def __get__(self):
            return pop%(id)s.%(name)s
        def __set__(self, value):
            pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name']}

        return pop_class

    def pyx_wrapper_proj(self):
        # Cython wrappers for the projections
        proj_class = ""
        for name, proj in self.projections.iteritems():
            # Init
            proj_class += """
cdef class proj%(id)s_wrapper :

    def __cinit__(self, synapses):

        cdef CSR syn = synapses
        cdef int size = syn.size
        
        proj%(id)s.size = size
        proj%(id)s.post_rank = syn.post_rank
        proj%(id)s.pre_rank = syn.pre_rank
        proj%(id)s.w = syn.w
        proj%(id)s.delay = syn.delay
"""% {'id': proj.id}

            for var in proj.synapse.description['parameters']:
                if var['name'] == 'w':
                    continue
                if var['name'] in proj.synapse.description['local']:
                    if isinstance(var['init'], np.ndarray):
                        init = 0.0 if var['ctype'] == 'double' else 0
                    else:
                        init = var['init']
                    proj_class += """
        proj%(id)s.%(name)s = vector[%(type)s](size, %(init)s)""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}
            for var in proj.synapse.description['variables']:
                if var['name'] == 'w':
                    continue
                if var['name'] in proj.synapse.description['local']:
                    if isinstance(var['init'], np.ndarray):
                        init = 0.0 if var['ctype'] == 'double' else 0
                    else:
                        init = var['init']
                    proj_class += """
        proj%(id)s.%(name)s = vector[%(type)s](size, %(init)s)""" %{'id': proj.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            # Size property
            proj_class += """

    property size:
        def __get__(self):
            return proj%(id)s.size
""" % {'id': proj.id}

            # Parameters
            for var in proj.synapse.description['parameters']:
                if var['name'] in proj.synapse.description['local']:
                    proj_class += """
    # Local parameter %(name)s
    property %(name)s:
        def __get__(self):
            return proj%(id)s.%(name)s
        def __set__(self, value):
            proj%(id)s.%(name)s = value
    def get_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.%(name)s[rank] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}
                elif var['name'] in pop.neuron.description['global']:
                    proj_class += """
    # Global parameter %(name)s
    property %(name)s:
        def __get__(self):
            return proj%(id)s.%(name)s
        def __set__(self, value):
            proj%(id)s.%(name)s = value
""" % {'id' : proj.id, 'name': var['name']}

            # Variables
            for var in proj.synapse.description['variables']:
                if var['name'] in proj.synapse.description['local']:
                    proj_class += """
    # Local variable %(name)s
    property %(name)s:
        def __get__(self):
            return proj%(id)s.%(name)s
        def __set__(self, value):
            proj%(id)s.%(name)s = value
    def get_%(name)s(self, int rank):
        return proj%(id)s.%(name)s[rank]
    def set_%(name)s(self, int rank, vector[%(type)s] value):
        proj%(id)s.%(name)s[rank] = value
""" % {'id' : proj.id, 'name': var['name'], 'type': var['ctype']}
                elif var['name'] in proj.synapse.description['global']:
                    proj_class += """
    # Global variable %(name)s
    property %(name)s:
        def __get__(self):
            return proj%(id)s.%(name)s
        def __set__(self, value):
            proj%(id)s.%(name)s = value
""" % {'id' : proj.id, 'name': var['name']}

        return proj_class