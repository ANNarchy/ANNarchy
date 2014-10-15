from ANNarchy.core.Population import Population, pop_generator_template
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

    """

    def __init__(self, geometry, name=None, rates=10.0, parameters=None):
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
                """
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
                """
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
                """
            )
        Population.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name)
        
        if isinstance(rates, np.ndarray):
            self.rates = rates

class SpikeSourceArray(Population):
    """
    Spike source generating spikes at the times given in the spike_times array.

    Depending on the initial array provided, the population will have one or several neurons, but the geoemtry can only be one-dimensional.

    *Parameters*:

<<<<<<< HEAD
    * **spike_times** : a list of times at which a spike should be emitted if the population has 1 neuron, a list of lists otherwise.
=======
    * **spike_times** : a list of times at which a spike should be emitted if the population has 1 neuron, a list of lists otherwise. Times are defined in milliseconds, and will be rounded to the closest multiple of the discretization time step dt.
>>>>>>> v43/master

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
            reset=""
        )

        Population.__init__(self, geometry=nb_neurons, neuron=neuron, name=name)

        # Do some sorting to save C++ complexity
        times = []
        for neur_times in spike_times:
            times.append(sorted(list(set(neur_times))) ) # suppress doublons and sort

        self.init['spike_times'] = times

        # Generate the Code
        self.generator['omp']['header_pop_struct'] = """
// Spike array
struct PopStruct%(id)s{
    // Number of neurons
    int size;
    // Spiking population
    std::vector<bool> spike;
    std::deque< std::vector<bool> > _delayed_spike;
    std::vector<long int> last_spike;
    std::vector<int> spiked;
    bool record_spike;
    std::vector<std::vector<long> > recorded_spike;
    // Local parameter spike_times
    std::vector< std::vector< double > > spike_times ;
    std::vector< double >  next_spike ;
    std::vector< int > idx_next_spike;
}; 
"""
        self.generator['omp']['body_spike_init'] = """    
    pop%(id)s.spike = std::vector<bool>(pop%(id)s.size, false);
    pop%(id)s.spiked = std::vector<int>(0, 0);
    pop%(id)s.last_spike = std::vector<long int>(pop%(id)s.size, -10000L);
    pop%(id)s.next_spike = std::vector<double>(pop%(id)s.size, -10000L);
    for(int i=0; i<pop%(id)s.size; i++){
        pop%(id)s.next_spike[i] = pop%(id)s.spike_times[i][0];  
    }
    pop%(id)s.idx_next_spike = std::vector<int>(pop%(id)s.size, 0);
"""
        self.generator['omp']['body_update_neuron'] = """ 
    // Updating the local variables of SpikeArray population %(id)s
    #pragma omp parallel for
    for(int i = 0; i < pop%(id)s.size; i++){
        // Emit spike 
        if( (t >= (long int)(pop%(id)s.next_spike[i]/dt)) && (t < (long int)(pop%(id)s.next_spike[i]/dt) +1 ) ){
            pop%(id)s.spike[i] = true;
            pop%(id)s.last_spike[i] = t;
            pop%(id)s.idx_next_spike[i]++ ;
            if(pop%(id)s.idx_next_spike[i] < pop%(id)s.spike_times[i].size())
                pop%(id)s.next_spike[i] = pop%(id)s.spike_times[i][pop%(id)s.idx_next_spike[i]];
        }
        else{
            pop%(id)s.spike[i] = false;
        }
    }
    // Gather spikes
    pop%(id)s.spiked.clear();
    for(int i=0; i< pop%(id)s.size; i++){
        if(pop%(id)s.spike[i]){
            pop%(id)s.spiked.push_back(i);
            if(pop%(id)s.record_spike){
                pop%(id)s.recorded_spike[i].push_back(t);
            }
        }
    }
"""   
        self.generator['omp']['pyx_pop_struct'] = """
    cdef struct PopStruct%(id)s :
        int size

        bool record_spike
        vector[vector[long]] recorded_spike

        # Local parameter spike_times
        vector[vector[double]] spike_times 
"""
        self.generator['omp']['pyx_pop_class'] = """
cdef class pop%(id)s_wrapper :
    def __cinit__(self, size):
        pop%(id)s.size = size
        # Spiking neuron
        pop%(id)s.record_spike = False
        pop%(id)s.recorded_spike = vector[vector[long]]()
        for i in xrange(pop%(id)s.size):
            pop%(id)s.recorded_spike.push_back(vector[long]())

        pop%(id)s.spike_times = vector[vector[double]](size, vector[double]())

    property size:
        def __get__(self):
            return pop%(id)s.size

    # Spiking neuron
    def start_record_spike(self):
        pop%(id)s.record_spike = True
    def stop_record_spike(self):
        pop%(id)s.record_spike = False
    def get_record_spike(self):
        cdef vector[vector[long]] tmp = pop%(id)s.recorded_spike
        for i in xrange(self.size):
            pop%(id)s.recorded_spike[i].clear()
        return tmp


    # Local parameter spike_times
    cpdef get_spike_times(self):
        return pop%(id)s.spike_times
    cpdef set_spike_times(self, value):
        pop%(id)s.spike_times = value
"""


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
            