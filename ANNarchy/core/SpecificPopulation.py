from ANNarchy.core.Population import Population
from ANNarchy.generator.Templates import pop_generator_template
from ANNarchy.core.Neuron import Neuron
import ANNarchy.core.Global as Global
import numpy as np

class PoissonPopulation(Population):
    """ 
    Population of spiking neurons following a Poisson distribution.

    Each neuron of the population will randomly emit spikes, with a mean firing rate defined by the *rates* argument.

    The mean firing rate in Hz can be a fixed value for all neurons::

        pop = PoissonPopulation(geometry=100, rates=100.0)

    but it can be modified later as a normal parameter::

        pop.rates = np.linspace(10, 150, 100)

    It is also possible to define a temporal equation for the rates, by passing a string to the argument::

        pop = PoissonPopulation(geometry=100, rates="100.0 * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0")

    The syntax of this equation follows the same structure as neural variables.

    It is also possible to add parameters to the population which can be used in the equation of *rates*::

        pop = PoissonPopulation( 
            geometry=100, 
            parameters = '''
                amp = 100.0
                frequency = 1.0
            ''',
            rates="amp * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0"
        )

    .. note::

        The preceding definition is fully equivalent to the definition of this neuron::

            poisson = Neuron(
                parameters = '''
                    amp = 100.0
                    frequency = 1.0
                ''',
                equations = '''
                    rates = amp * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0
                    p = Uniform(0.0, 1.0) * 1000.0 / dt
                ''',
                spike = '''
                    p < rates
                '''
            )

    The refractory period can also be set, so that a neuron can not emit two spikes too close from each other.

    """

    def __init__(self, geometry, name=None, rates=10.0, parameters=None, refractory=None):
        """        
        *Parameters*:
        
            * *geometry*: population geometry as tuple. 

            * *name*: unique name of the population (optional).

            * *rates*: mean firing rate of each neuron (default: 10.0 Hz). It can be a single value (e.g. 10.0) or an equation (as string).

            * *parameters*: additional parameters which can be used in the *rates* equation.
        """  
        if isinstance(rates, str):
            poisson_neuron = Neuron(
                parameters = """
                %(params)s
                """ % {'params': parameters if parameters else ''},
                equations = """
                rates = %(rates)s
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """ % {'rates': rates},
                spike = """
                    p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution."
            )

        elif isinstance(rates, np.ndarray):
            poisson_neuron = Neuron(
                parameters = """
                rates = 10.0
                """,
                equations = """
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike = """
                p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution."
            )
        else:
            poisson_neuron = Neuron(
                parameters = """
                rates = %(rates)s
                """ % {'rates': rates},
                equations = """
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                """,
                spike = """
                p < rates
                """,
                refractory=refractory,
                name="Poisson",
                description="Spiking neuron with spikes emitted according to a Poisson distribution."
            )
        Population.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name)
        
        if isinstance(rates, np.ndarray):
            self.rates = rates

class SpikeSourceArray(Population):
    """
    Spike source generating spikes at the times given in the spike_times array.

    Depending on the initial array provided, the population will have one or several neurons, but the geoemtry can only be one-dimensional.

    *Parameters*:

    * **spike_times** : a list of times at which a spike should be emitted if the population has 1 neuron, a list of lists otherwise. Times are defined in milliseconds, and will be rounded to the closest multiple of the discretization time step dt.
    * **name**: optional name for the population.
    """
    def __init__(self, spike_times, name=None):

        if not isinstance(spike_times, list):
            Global._error('in SpikeSourceArray, spike_times must be a Python list.')
            exit(0)

        if isinstance(spike_times[0], list): # several neurons
            nb_neurons = len(spike_times)
        else: # a single Neuron
            nb_neurons = 1
            spike_times = [ spike_times ]

        # Create a fake neuron just to be sure the description has the correct parameters
        neuron = Neuron(
            parameters="""
                spike_times = 0.0
            """,
            equations="",
            spike=" t == spike_times",
            reset="",
            name="Spike source",
            description="Spikes source array."
        )

        Population.__init__(self, geometry=nb_neurons, neuron=neuron, name=name)

        # Do some sorting to save C++ complexity
        times = []
        for neur_times in spike_times:
            times.append(sorted(list(set(neur_times)))) # suppress doublons and sort

        self.init['spike_times'] = times

        # Skip the normal code generation process
        self._specific = True

        self.generator['omp']['pyx_pop_struct'] = """
    cdef struct PopStruct%(id)s :
        int size
        bool _active

        # Local parameter spike_times
        vector[vector[double]] spike_times
"""

        self.generator['omp']['pyx_pop_class'] = """
cdef class pop%(id)s_wrapper :
    def __cinit__(self, size, times):
        pop%(id)s.size = size
        pop%(id)s.spike_times = times

    property size:
        def __get__(self):
            return pop%(id)s.size

    def activate(self, bool val):
        pop%(id)s._active = val

    # Local parameter spike_times
    cpdef get_spike_times(self):
        return pop%(id)s.spike_times
    cpdef set_spike_times(self, value):
        pop%(id)s.spike_times = value
"""

    def generate(self):
        # Generate the Code
        code = """#pragma once

extern long int t;
extern double dt;

// Spike array
struct PopStruct%(id)s{
    // Number of neurons
    int size;
    bool _active;

    // Spiking population
    std::deque< std::vector<int> > _delayed_spike;
    std::vector<long int> last_spike;
    std::vector<int> spiked;

    // Local parameter spike_times
    std::vector< std::vector< double > > spike_times ;
    std::vector< double >  next_spike ;
    std::vector< int > idx_next_spike;

    int get_size() { return size; }

    bool is_active() { return _active; }
    void set_active(bool value) { _active = value; }

    void init_population() {
        size = %(size)s;

        spiked = std::vector<int>();
        last_spike = std::vector<long int>(size, -10000L);
        next_spike = std::vector<double>(size, -10000.0);
        for(int i=0; i< size; i++){
            if(!spike_times[i].empty())
                next_spike[i] = spike_times[i][0];
        }
        idx_next_spike = std::vector<int>(size, 0);
        _delayed_spike = std::deque< std::vector<int> >(%(delay)s, std::vector<int>());
    }

    void update() {
        // Updating the local variables of SpikeArray population %(id)s
        if(_active){
            spiked.clear();
            for(int i = 0; i < size; i++){
                // Emit spike
                if( t == (long int)(next_spike[i]/dt) ){
                    last_spike[i] = t;
                    idx_next_spike[i]++ ;
                    if(idx_next_spike[i] < spike_times[i].size())
                        next_spike[i] = spike_times[i][idx_next_spike[i]];
                    spiked.push_back(i);
                }
            }
        }
    }
};
""" % { 'id': self.id, 'size': self.size, 'delay': self.max_delay }

        return code
        
    def _instantiate(self, module):
        # Create the Cython instance 
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.init['spike_times'])

    def __setattr__(self, name, value):
        if name == 'spike_times':
            if self.initialized:
                self.cyInstance.set_spike_times(value)
            else:
                object.__setattr__(self, name, value)
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'spike_times':
            if self.initialized:
                return self.cyInstance.get_spike_times()
            else:
                return object.__getattribute__(self, name)
        else:
            return Population.__getattribute__(self, name)
            