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
import PopulationTemplate as PopTemplate

class PopulationGenerator(object):

#######################################################################
############## HEADER #################################################
#######################################################################
    def header_struct(self, pop):
        """
        Generate the c-style struct definition for a population object.

        Used templates:

            header_struct, parameter_decl, parameter_acc, variable_decl, variable_acc
        """
        # Is it a specific population?
        if pop.generator['omp']['header_pop_struct']:
            return pop.generator['omp']['header_pop_struct'] % {'id': pop.id}

        # Pick basic template based on neuron type
        base_template = PopTemplate.header_struct[pop.neuron_type.type]

        code = "" # member declarations
        accessors = "" # export member functions

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.parameter_decl['local'] % {'type' : var['ctype'], 'name': var['name']}
                accessors += PopTemplate.parameter_acc['local'] % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += PopTemplate.parameter_decl['global'] % {'type' : var['ctype'], 'name': var['name']}
                accessors += PopTemplate.parameter_acc['global'] % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.variable_decl['local'] % {'type' : var['ctype'], 'name': var['name']}
                accessors += PopTemplate.variable_acc['local'] % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += PopTemplate.variable_decl['global'] % {'type' : var['ctype'], 'name': var['name']}
                accessors += PopTemplate.variable_acc['global'] % {'type' : var['ctype'], 'name': var['name']}

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
    // Delayed variables"""
                for var in pop.delayed_variables:
                    if var in pop.neuron_type.description['local']:
                        code += """
    std::deque< std::vector<double> > _delayed_%(var)s; """ % {'var': var}
                    else:
                        code += """
    std::deque< double > _delayed_%(var)s; """ % {'var': var}
            else: # Spiking networks should only exchange spikes
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

        init = self.init_population(pop)
        update = self.update_neuron(pop).replace("pop"+str(pop.id)+".", "") #TODO: adjust prefixes in parser

        return base_template % { 'id': pop.id, 
                                 'additional': code, 
                                 'accessor': accessors, 
                                 'init': init,
                                 'update': update 
                                }

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
        omp_code = "#pragma omp parallel for" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""
        code += """
        // Updating the local variables of population %(id)s (%(name)s)
        %(omp_code)s
        for(int i = 0; i < %(size)s; i++){
%(eqs)s
        }
""" % {'id': pop.id, 'size': pop.size, 'name' : pop.name, 'eqs': eqs, 'omp_code': omp_code}

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
        omp_code = "#pragma omp parallel for" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""
        code += """
        // Updating the local variables of population %(id)s (%(name)s)
        pop%(id)s.spiked.clear();
        %(omp_code)s
        for(int i = 0; i < %(size)s; i++){
%(eqs)s
""" % {'id': pop.id, 'size': pop.size, 'name' : pop.name, 'eqs': eqs, 'omp_code': omp_code}

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
        omp_critical_code = "#pragma omp critical" if (Global.config['num_threads'] > 1 and pop.size > Global.OMP_MIN_NB_NEURONS) else ""
        spike_gather = """
            %(refrac_period)sif(%(condition)s){ // Emit a spike
%(reset)s        
                %(omp_critical_code)s
                {
                    pop%(id)s.spiked.push_back(i);
                }
                pop%(id)s.last_spike[i] = t;
                %(refrac_inc)s
            }
        }
"""% {  'id': pop.id, 'name': pop.name, 'size': pop.size, 
        'condition' : cond, 'reset': reset, 
        'refrac_period': refrac_period, 'refrac_inc': refrac_inc,
        'omp_critical_code': omp_critical_code} 

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
    if ( pop%(id)s._active ) {""" % {'id': pop.id, 'name' : pop.name }

            for var in pop.delayed_variables:
                code += """
        pop%(id)s._delayed_%(var)s.push_front(pop%(id)s.%(var)s);
        pop%(id)s._delayed_%(var)s.pop_back();""" % {'id': pop.id, 'var' : var}

            code += """
    }
""" 
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
        code = """
    // Delayed variables
"""
        # Is it a specific population?
        if pop.generator['omp']['body_delay_init']:
            code += pop.generator['omp']['body_delay_init'] %{'id': pop.id, 'delay': pop.max_delay}
            return code

        if pop.neuron_type.type == 'rate':
            for var in pop.delayed_variables:
                if var in pop.neuron_type.description['local']:
                    code += """
    _delayed_%(var)s = std::deque< std::vector<double> >(%(delay)s, std::vector<double>(size, 0.0)); """ % {'id': pop.id, 'delay': pop.max_delay, 'var': var}
                else:
                    code += """
    _delayed_%(var)s = std::deque< double >(%(delay)s, 0.0); """ % {'id': pop.id, 'delay': pop.max_delay, 'var': var}


        else: # SPIKE
            code += """
    _delayed_spike = std::deque< std::vector<int> >(%(delay)s, std::vector<int>()); """ % {'id': pop.id, 'delay': pop.max_delay}

        return code

    def init_population(self, pop):
        # active is true by default
        code = """
        size = %(size)s;
        _active = true;
        record_period = 1;
        record_offset = 0;
        record_ranks = std::vector<int>(1, -1);
""" % { 'id': pop.id, 'size': pop.size }

        # Is it a specific population?
        if pop.generator['omp']['body_spike_init']:
            code += pop.generator['omp']['body_spike_init'] %{'id': pop.id}
            if pop.max_delay > 1:
                code += self.init_delay(pop)
            return code

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.parameter_cpp_init['local'] % {'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += PopTemplate.parameter_cpp_init['global'] %{'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Variables
        for var in pop.neuron_type.description['variables']:
            init = 0.0 if var['ctype'] == 'double' else 0
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.variable_cpp_init['local'] % {'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

            else: # global
                code += PopTemplate.variable_cpp_init['global'] % {'id': pop.id, 'name': var['name'], 'type': var['ctype'], 'init': init}

        # Targets
        if pop.neuron_type.type == 'rate':
            for target in list(set(pop.neuron_type.description['targets'] + pop.targets)):
                code += PopTemplate.model_specific_init['rate_psp'] % {'id': pop.id, 'target': target}

        # Spike event and refractory
        if pop.neuron_type.type == 'spike':
            code += PopTemplate.model_specific_init['spike_event'] % {'id': pop.id}

        # Delays
        if pop.max_delay > 1:
            code += self.init_delay(pop)

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
            if var['name'] in pop.neuron_type.description['local']:
                code += """
    if(pop%(id)s.record_%(name)s && ( (t - pop%(id)s.record_offset) %(mod)s pop%(id)s.record_period == 0 ) ){
        if(pop%(id)s.record_ranks[0]==-1){ // Record everything
            pop%(id)s.recorded_%(name)s.push_back(pop%(id)s.%(name)s) ;
        }
        else{
            std::vector<%(type)s> tmp = std::vector<%(type)s>();
            for(int i=0; i<pop%(id)s.record_ranks.size(); i++){
                tmp.push_back(pop%(id)s.%(name)s[pop%(id)s.record_ranks[i]]);
            }
            pop%(id)s.recorded_%(name)s.push_back(tmp) ;
            tmp.clear();
        }
    }
""" % {'id': pop.id, 'type' : var['ctype'], 'name': var['name'], 'mod': '%' }
            else:
                code += """
    if(pop%(id)s.record_%(name)s && ( (t - pop%(id)s.record_offset) %(mod)s pop%(id)s.record_period == 0 ) ){
        pop%(id)s.recorded_%(name)s.push_back(pop%(id)s.%(name)s) ;
    }
""" % {'id': pop.id, 'type' : var['ctype'], 'name': var['name'], 'mod': '%' }

        # Special case for spike
        if pop.neuron_type.type == 'spike':
            code += """
    // Recording spikes
    if(pop%(id)s.record_spike){
        for(int i=0; i<pop%(id)s.spiked.size(); i++){
            if(pop%(id)s.record_ranks[0]==-1){ // Record everything
                pop%(id)s.recorded_spike[pop%(id)s.spiked[i]].push_back(t);
            }
            else{
                if(std::find(pop%(id)s.record_ranks.begin(), pop%(id)s.record_ranks.end(), pop%(id)s.spiked[i]) != pop%(id)s.record_ranks.end()){
                    pop%(id)s.recorded_spike[pop%(id)s.spiked[i]].push_back(t);
                }
            }
        }
    }
""" % {'id': pop.id}

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
        int get_size()
        bool is_active()
        void set_active(bool)
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
        void set_record_period( int, long int)
        void set_record_ranks(vector[int])
"""

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.parameter_cpp_export['local'] % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += PopTemplate.parameter_cpp_export['global'] % {'type' : var['ctype'], 'name': var['name']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.variable_cpp_export['local'] % {'type' : var['ctype'], 'name': var['name']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += PopTemplate.variable_cpp_export['global'] % {'type' : var['ctype'], 'name': var['name']}

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
        pass
"""% {'id': pop.id, 'name': pop.name}

        # Size property
        code += """

    property size:
        def __get__(self):
            return pop%(id)s.get_size()
""" % {'id': pop.id}

        # Activate population
        code += """

    def activate(self, bool val):
        pop%(id)s.set_active( val )
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
        pop%(id)s.set_record_period( period, t )
    cpdef set_record_ranks( self, list ranks ):
        pop%(id)s.set_record_ranks( ranks )
""" % {'id': pop.id}

        # Parameters
        for var in pop.neuron_type.description['parameters']:
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.parameter_pyx_wrapper['local'] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += PopTemplate.parameter_pyx_wrapper['global'] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

        # Variables
        for var in pop.neuron_type.description['variables']:
            if var['name'] in pop.neuron_type.description['local']:
                code += PopTemplate.variable_pyx_wrapper['local'] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

            elif var['name'] in pop.neuron_type.description['global']:
                code += PopTemplate.variable_pyx_wrapper['global'] % {'id' : pop.id, 'name': var['name'], 'type': var['ctype']}

        return code
