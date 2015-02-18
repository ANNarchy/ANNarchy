"""

    PopulationGenerator.py

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

class PopulationGenerator(object):

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, pop):
        """
        Generate the c-style struct definition for a population object.
        """
        # Is it a specific population?
        if pop.generator['omp']['header_pop_struct']:
            return pop.generator['omp']['header_pop_struct'] % {'id': pop.id}

        # Generate the structure for the population.
        code = """
struct PopStruct%(id)s{
    // Number of neurons
    int size;
    // Active
    bool _active;
"""
        # Spiking neurons have aditional data
        if pop.neuron_type.type == 'spike':
            code += """
    // Spiking population
    std::vector<long int> last_spike;
    std::vector<int> spiked;
    std::vector<int> refractory;
    std::vector<int> refractory_remaining;
    bool record_spike;
    std::vector<std::vector<long> > recorded_spike;
"""

        # Record
        code+="""
    // Record parameter
    int record_period;
    long int record_offset;
"""

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local parameter %(name)s
    std::vector< %(type)s > %(name)s;
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    // Global parameter %(name)s
    %(type)s  %(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local variable %(name)s
    std::vector< %(type)s > %(name)s ;
    std::vector< std::vector< %(type)s > > recorded_%(name)s ;
    bool record_%(name)s ;
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
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
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets']+pop.targets)):
                code += """    std::vector<double> _sum_%(target)s;
""" % {'target' : target}

        # Global operations
        code += """
    // Global operations
"""
        for op in pop.global_operations:
            code += """    double _%(op)s_%(var)s;
""" % {'op': op['function'], 'var': op['variable']}

        # Arrays for the random numbers
        code += """
    // Random numbers (STL implementation)
"""
        for rd in pop.neuron_type.description['random_distributions']:
            code += """    std::vector<double> %(rd_name)s;
    %(template)s dist_%(rd_name)s;
""" % {'rd_name' : rd['name'], 'type': rd['dist'], 'template': rd['template']}


        # Delays
        if pop.max_delay > 1:
            if pop.neuron_type.type == "rate":
                code += """
    // Delays for rate-coded population
    std::deque< std::vector<double> > _delayed_r;
"""
            else:
                code += """
    // Delays for spike population
    std::deque< std::vector<int> > _delayed_spike;
"""

        # Local functions
        if len(pop.neuron_type.description['functions'])>0:
            code += """
    // Local functions
"""
            for func in pop.neuron_type.description['functions']:
                code += ' '*4 + func['cpp'] + '\n'

        # Finish the structure
        code += """
};
"""
        return code % {'id': pop.id}

#######################################################################
############## BODY ###################################################
#######################################################################

    def update_neuron(self, pop):
        """
        generate omp update code.
        """
        # Is it a specific population?
        if pop.generator['omp']['body_update_neuron']:
            return pop.generator['omp']['body_update_neuron'] %{'id': pop.id}

        # Is there any variable?
        if len(pop.neuron_type.description['variables']) == 0:
            # even if there are no variables we still need to do something for profiling
            if Global.config['profiling']:
                return "\n    //nothing to do for pop%(id)s, but increase counter\n    rc++;\n" % {'id': pop.id}
            else:
                return ""

        if pop.neuron_type.type == 'rate':
            return self.update_rate_neuron(pop)
        else:
            return self.update_spike_neuron(pop)


    def update_rate_neuron(self, pop):
        # Neural update
        from ..Utils import generate_equation_code
        code = ""

        # Global variables
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global') % {'id': pop.id}
        if eqs.strip() != "":
            code += """
    // Updating the global variables of population %(id)s (%(name)s)
%(eqs)s
""" % {'id': pop.id, 'name' : pop.name, 'eqs': eqs}

        # Local variables, evaluated in parallel
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local') % {'id': pop.id}
        code += """
        // Updating the local variables of population %(id)s (%(name)s)
        #pragma omp parallel for firstprivate(dt)
        for(int i = 0; i < %(size)s; i++){
%(eqs)s
        }
""" % {'id': pop.id, 'size': pop.size, 'name' : pop.name, 'eqs': eqs}

        # if profiling enabled, annotate with profiling code
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            pGen = ProfileGenerator(Global._populations, Global._projections)
            code = pGen.annotate_update_neuron_omp(code)

        # finish code
        return """
    if(pop%(id)s._active){
%(code)s
    } // active
""" % {'id': pop.id, 'code': code }


    def update_spike_neuron(self, pop):
        # Neural update
        from ..Utils import generate_equation_code
        code = ""

        # Global variables
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'global') % {'id': pop.id}
        if eqs.strip() != "":
            code += """
    // Updating the global variables of population %(id)s (%(name)s)
%(eqs)s
""" % {'id': pop.id, 'name' : pop.name, 'eqs': eqs}

        # Local variables, evaluated in parallel
        eqs = generate_equation_code(pop.id, pop.neuron_type.description, 'local') % {'id': pop.id}
        code += """
        // Updating the local variables of population %(id)s (%(name)s)
        pop%(id)s.spiked.clear();
        #pragma omp parallel for firstprivate(dt)
        for(int i = 0; i < %(size)s; i++){
%(eqs)s
""" % {'id': pop.id, 'size': pop.size, 'name' : pop.name, 'eqs': eqs}

        # if profiling enabled, annotate with profiling code
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            pGen = ProfileGenerator(Global._populations, Global._projections)
            code = pGen.annotate_update_neuron_omp(code)

        # Process the condition
        cond =  pop.neuron_type.description['spike']['spike_cond'] % {'id': pop.id}
        reset = ""; refrac = ""
        # reset equations
        for eq in pop.neuron_type.description['spike']['spike_reset']:
            reset += """
                %(reset)s
""" % {'reset': eq['cpp'] % {'id': pop.id}}
            if not 'unless_refractory' in eq['constraint']:
                refrac += """
                %(refrac)s
""" % {'refrac': eq['cpp'] % {'id': pop.id} }

        # Is there a refractory period?
        if pop.neuron_type.refractory or pop.refractory:
            refrac_period = """if(pop%(id)s.refractory_remaining[i] > 0){ // Refractory period
%(refrac)s
                pop%(id)s.refractory_remaining[i]--;
            }
            else """ %  {'id': pop.id, 'refrac': refrac}
            refrac_inc = "pop%(id)s.refractory_remaining[i] = pop%(id)s.refractory[i];" %  {'id': pop.id}
        else:
            refrac_period = ""
            refrac_inc = ""

        # Main code
        spike_gather = """
            %(refrac_period)sif(%(condition)s){ // Emit a spike
%(reset)s        
                #pragma omp critical
                {
                    pop%(id)s.spiked.push_back(i);
                    if(pop%(id)s.record_spike){
                        pop%(id)s.recorded_spike[i].push_back(t);
                    }
                }
                pop%(id)s.last_spike[i] = t;
                %(refrac_inc)s
            }
        }
"""% {'id': pop.id, 'name': pop.name, 'size': pop.size, 'condition' : cond, 'reset': reset, 'refrac_period': refrac_period, 'refrac_inc': refrac_inc} 

        # if profiling enabled, annotate with profiling code
        if Global.config['profiling']:
            from ..Profile.ProfileGenerator import ProfileGenerator
            pGen = ProfileGenerator(Global._populations, Global._projections)
            spike_gather = pGen.annotate_spike_propagation_omp(spike_gather)

        code += spike_gather

        # finish code
        return """
    if(pop%(id)s._active){
%(code)s
    } // active
""" % {'id': pop.id, 'code': code }


    def delay_code(self, pop):
        # No delay
        if pop.max_delay <= 1:
            return ""

        # Is it a specific population?
        if pop.generator['omp']['body_delay_code']:
            return pop.generator['omp']['body_delay_code'] % {'id': pop.id}

        code = ""
        if pop.neuron_type.type == 'rate':
            code += """
    // Enqueuing outputs of pop%(id)s (%(name)s)
    if ( pop%(id)s._active ) {
        pop%(id)s._delayed_r.push_front(pop%(id)s.r);
        pop%(id)s._delayed_r.pop_back();
    }
""" % {'id': pop.id, 'name' : pop.name }

        else:
            code += """
    // Enqueuing outputs of pop%(id)s (%(name)s)
    if (pop%(id)s._active){
        pop%(id)s._delayed_spike.push_front(pop%(id)s.spiked);
        pop%(id)s._delayed_spike.pop_back();
    }
""" % {'id': pop.id, 'name' : pop.name }

        return code

    def init_globalops(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['body_globalops_init']:
            return pop.generator['omp']['body_globalops_init'] %{'id': pop.id}

        code = ""
        for op in pop.global_operations:
            code += """    pop%(id)s._%(op)s_%(var)s = 0.0;
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}
        return code

    def init_random_distributions(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['body_random_dist_init']:
            return pop.generator['omp']['body_random_dist_init'] %{'id': pop.id}

        code = ""
        for rd in pop.neuron_type.description['random_distributions']:
            code += """    pop%(id)s.%(rd_name)s = std::vector<double>(pop%(id)s.size, 0.0);
    pop%(id)s.dist_%(rd_name)s = %(rd_init)s;
""" % {'id': pop.id, 'rd_name': rd['name'], 'rd_init': rd['definition']% {'id': pop.id}}
        return code

    def init_delay(self, pop):
        code = "    // Delays from pop%(id)s (%(name)s)\n" % {'id': pop.id, 'name': pop.name}

        # Is it a specific population?
        if pop.generator['omp']['body_delay_init']:
            code += pop.generator['omp']['body_delay_init'] %{'id': pop.id, 'delay': pop.max_delay}
            return code

        if pop.neuron_type.type == 'rate':
            code += """    pop%(id)s._delayed_r = std::deque< std::vector<double> >(%(delay)s, std::vector<double>(pop%(id)s.size, 0.0));
""" % {'id': pop.id, 'delay': pop.max_delay}
        else: # SPIKE
            code += """    pop%(id)s._delayed_spike = std::deque< std::vector<int> >(%(delay)s, std::vector<int>());
""" % {'id': pop.id, 'delay': pop.max_delay}

        return code

    def init_population(self, pop):
        # active is true by default
        code = """
    /////////////////////////////
    // Population %(id)s
    /////////////////////////////
    pop%(id)s._active = true;
    pop%(id)s.record_period = 1;
    pop%(id)s.record_offset = 0;
""" % {'id': pop.id}

        # Is it a specific population?
        if pop.generator['omp']['body_spike_init']:
            code += pop.generator['omp']['body_spike_init'] %{'id': pop.id}
            return code

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local parameter %(name)s
    pop%(id)s.%(name)s = std::vector<%(type)s>(pop%(id)s.size, %(init)s);
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += """
    // Global parameter %(name)s
    pop%(id)s.%(name)s = %(init)s;
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Variables
        for var in pop.neuron_type.description['variables']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    // Local variable %(name)s
    pop%(id)s.%(name)s = std::vector<%(type)s>(pop%(id)s.size, %(init)s);
    pop%(id)s.recorded_%(name)s = std::vector<std::vector<%(type)s> >(0, std::vector<%(type)s>(0,%(init)s));
    pop%(id)s.record_%(name)s = false;
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += """
    // Global variable %(name)s
    pop%(id)s.%(name)s = %(init)s;
    pop%(id)s.recorded_%(name)s = std::vector<%(type)s>(0, %(init)s);
    pop%(id)s.record_%(name)s = false;
""" %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Targets
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                code += """
    pop%(id)s._sum_%(target)s = std::vector<double>(pop%(id)s.size, 0.0);""" %{'id': pop.id, 'target': target}

        if pop.neuron_type.type == 'spike':
            code += """
    // Spiking neuron
    pop%(id)s.refractory = std::vector<int>(pop%(id)s.size, 0);
    pop%(id)s.record_spike = false;
    pop%(id)s.recorded_spike = std::vector<std::vector<long int> >();
    for(int i = 0; i < pop%(id)s.size; i++)
        pop%(id)s.recorded_spike.push_back(std::vector<long int>());
    pop%(id)s.spiked = std::vector<int>(0, 0);
    pop%(id)s.last_spike = std::vector<long int>(pop%(id)s.size, -10000L);
    pop%(id)s.refractory_remaining = std::vector<int>(pop%(id)s.size, 0);
""" % {'id': pop.id}

        return code

    def reset_computesum(self, pop):
        code = ""
        for target in pop.targets:
            code += """
    if (pop%(id)s._active)
        memset( pop%(id)s._sum_%(target)s.data(), 0.0, pop%(id)s._sum_%(target)s.size() * sizeof(double));
""" % {'id': pop.id, 'target': target}
        return code

    def update_globalops(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['body_update_globalops']:
            return pop.generator['omp']['body_update_globalops'] %{'id': pop.id}

        code = ""
        if len(pop.global_operations) > 0:
            code += """
    if (pop%(id)s._active){
"""% {'id': pop.id}

            for op in pop.global_operations:
                code += """    pop%(id)s._%(op)s_%(var)s = %(op)s_value(pop%(id)s.%(var)s.data(), pop%(id)s.size);
""" % {'id': pop.id, 'op': op['function'], 'var': op['variable']}

            code += """
    }
"""

        return code

    def record(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['body_record']:
            return pop.generator['omp']['body_record'] %{'id': pop.id}

        code = ""
        for var in pop.neuron_type.description['variables']:
            code += """
    if(pop%(id)s.record_%(name)s && ( (t - pop%(id)s.record_offset) %(mod)s pop%(id)s.record_period == 0 ) )
        pop%(id)s.recorded_%(name)s.push_back(pop%(id)s.%(name)s) ;
""" % {'id': pop.id, 'type' : var['ctype'], 'name': var['name'], 'mod': '%' }

        return code

    def stop_condition(self, pop):
        if not pop.stop_condition: # no stop condition has been defined
            return """
                case %(id)s:
                    pop_stop = false;
                    break;
""" % {'id': pop.id}

        pop.neuron_type.description['stop_condition'] = {'eq': pop.stop_condition}

        from ANNarchy.parser.Extraction import extract_stop_condition
        from ANNarchy.parser.SingleAnalysis import pattern_omp, pattern_cuda

        # Find the paradigm OMP or CUDA
        if Global.config['paradigm'] == 'cuda':
            pattern = pattern_cuda
        else:
            pattern = pattern_omp

        extract_stop_condition(pop.neuron_type.description, pattern)

        if pop.neuron_type.description['stop_condition']['type'] == 'any':
            stop_code = """
                    pop_stop = false;
                    for(int i=0; i<pop%(id)s.size; i++)
                    {
                        if(%(condition)s)
                            pop_stop = true;
                    }
    """ % {'id': pop.id, 'condition': pop.neuron_type.description['stop_condition']['cpp']% {'id': pop.id}}
        else:
            stop_code = """
                    pop_stop = true;
                    for(int i=0; i<pop%(id)s.size; i++)
                    {
                        if(!(%(condition)s))
                            pop_stop = false;
                    }
    """ % {'id': pop.id, 'condition': pop.neuron_type.description['stop_condition']['cpp']% {'id': pop.id}}

        return """
                case %(id)s:
%(stop_code)s
                    break;
""" % {'id': pop.id, 'stop_code': stop_code}

    def update_random_distributions(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['body_random_dist_update']:
            return pop.generator['omp']['body_random_dist_update'] %{'id': pop.id}

        code = ""
        if len(pop.neuron_type.description['random_distributions']) > 0:
            code += """
    // RD of pop%(id)s
    if (pop%(id)s._active){
        for(int i = 0; i < pop%(id)s.size; i++)
        {
"""% {'id': pop.id}

            for rd in pop.neuron_type.description['random_distributions']:
                code += """
            pop%(id)s.%(rd_name)s[i] = pop%(id)s.dist_%(rd_name)s(rng);
""" % {'id': pop.id, 'rd_name': rd['name']}

            code += """
        }
    }
"""

        return code

#######################################################################
############## PYX ####################################################
#######################################################################

    def pyx_struct(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['pyx_pop_struct']:
            return pop.generator['omp']['pyx_pop_struct'] %{'id': pop.id}

        code = """
    # Population %(id)s (%(name)s)
    cdef struct PopStruct%(id)s :
        int size
        bool _active
"""
        # Spiking neurons have aditional data
        if pop.neuron_type.type == 'spike':
            code += """
        vector[int] refractory
        bool record_spike
        vector[vector[long]] recorded_spike
"""

        # Record parameter
        code += """
        # Record parameter
        long int record_offset
        int record_period
"""

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
        # Local parameter %(name)s
        vector[%(type)s] %(name)s
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
        # Global parameter %(name)s
        %(type)s  %(name)s
""" % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
        # Local variable %(name)s
        vector[%(type)s] %(name)s
        vector[vector[%(type)s]] recorded_%(name)s
        bool record_%(name)s
""" % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
        # Global variable %(name)s
        %(type)s  %(name)s
        vector[%(type)s] recorded_%(name)s
        bool record_%(name)s
""" % {'type' : var['ctype'], 'name': var['name']}

        # Arrays for the presynaptic sums of rate-coded neurons
        if pop.neuron_type.type == 'rate':
            code += """
        # Targets
"""
            for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                code += """        vector[double] _sum_%(target)s
""" % {'target' : target}

        # Finalize the code
        return code % {'id': pop.id, 'name': pop.name}

    def pyx_wrapper(self, pop):
        # Is it a specific population?
        if pop.generator['omp']['pyx_pop_class']:
            return pop.generator['omp']['pyx_pop_class'] %{'id': pop.id}

        # Init
        code = """
# Population %(id)s (%(name)s)
cdef class pop%(id)s_wrapper :

    def __cinit__(self, size):
        pop%(id)s.size = size
"""% {'id': pop.id, 'name': pop.name}

        # Size property
        code += """

    property size:
        def __get__(self):
            return pop%(id)s.size
""" % {'id': pop.id}

        # Activate population
        code += """

    def activate(self, bool val):
        pop%(id)s._active = val
""" % {'id': pop.id}

        # Spiking neurons have aditional data
        if pop.neuron_type.type == 'spike':
            code += """
    # Spiking neuron
    cpdef np.ndarray get_refractory(self):
        return pop%(id)s.refractory
    cpdef set_refractory(self, np.ndarray value):
        pop%(id)s.refractory = value

    def start_record_spike(self):
        pop%(id)s.record_spike = True
    def stop_record_spike(self):
        pop%(id)s.record_spike = False
    def get_record_spike(self):
        cdef vector[vector[long]] tmp = pop%(id)s.recorded_spike
        for i in xrange(self.size):
            pop%(id)s.recorded_spike[i].clear()
        return tmp

""" % {'id': pop.id}

        # Record parameter
        code += """
    # Record parameter
    cpdef set_record_period( self, int period, long int t ):
        pop%(id)s.record_period = period
        pop%(id)s.record_offset = t
""" % {'id': pop.id}

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    # Local parameter %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.%(name)s[rank] = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    # Global parameter %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    # Local variable %(name)s
    cpdef np.ndarray get_%(name)s(self):
        return np.array(pop%(id)s.%(name)s)
    cpdef set_%(name)s(self, np.ndarray value):
        pop%(id)s.%(name)s = value
    cpdef %(type)s get_single_%(name)s(self, int rank):
        return pop%(id)s.%(name)s[rank]
    cpdef set_single_%(name)s(self, int rank, value):
        pop%(id)s.%(name)s[rank] = value
    def start_record_%(name)s(self):
        pop%(id)s.record_%(name)s = True
    def stop_record_%(name)s(self):
        pop%(id)s.record_%(name)s = False
    def get_record_%(name)s(self):
        cdef vector[vector[%(type)s]] tmp = pop%(id)s.recorded_%(name)s
        for i in xrange(tmp.size()):
            pop%(id)s.recorded_%(name)s[i].clear()
        pop%(id)s.recorded_%(name)s.clear()
        return tmp
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += """
    # Global variable %(name)s
    cpdef %(type)s get_%(name)s(self):
        return pop%(id)s.%(name)s
    cpdef set_%(name)s(self, %(type)s value):
        pop%(id)s.%(name)s = value
    def start_record_%(name)s(self):
        pop%(id)s.record_%(name)s = True
    def stop_record_%(name)s(self):
        pop%(id)s.record_%(name)s = False
    def get_record_%(name)s(self):
        cdef vector[%(type)s] tmp = pop%(id)s.recorded_%(name)s
        pop%(id)s.recorded_%(name)s.clear()
        return tmp
""" % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

        return code
