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
        poisson_neuron = Neuron(
            parameters = """
            rates = 0.0
            """ ,
            equations = """
            p = Uniform(0.0, 1.0)
            """,
            spike = """
            p <= rates
            """,
            reset=""
        )
            
        Population.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name)
        
        if isinstance(rates, np.ndarray):
            self.rates = rates


        # Code generation
        self.generator['omp']['body_update_neuron'] = """
    // Updating the Poisson population %(id)s
    #pragma omp parallel for
    for(int i = 0; i < pop%(id)s.size; i++){           
        if(1000.0*pop%(id)s.rand_0[i] <= dt*pop%(id)s.rates[i]){
            pop%(id)s.spike[i] = true;
            pop%(id)s.last_spike[i] = t;
        }
        else{
            pop%(id)s.spike[i] = false;
        }
    }
    pop%(id)s.spiked.clear();
    for(int i=0; i< (int)pop%(id)s.size; i++){
        if(pop%(id)s.spike[i]){
            pop%(id)s.spiked.push_back(i);
            if(pop%(id)s.record_spike){
                pop%(id)s.recorded_spike[i].push_back(t);
            }
        }
    }
"""


class SpikeSourceArray(Population):
    """
    Spike source generating spikes at the times given in the spike_times array.

    Depending on the initial array provided, the population will have one or several neurons, but the geoemtry can only be one-dimensional.

    *Parameters*:

    * **spike_times** : a list of times at which a spike should be emitted if the population has 1 neuron, a list of lists otherwise.

    * **name**: optional name for the population.
    """
    def __init__(self, spike_times, name=None):

        if not isinstance(spike_times, list):
            Global._error('in SpikeSourceArray, spike_times must be a Python list.')
            exit(0)