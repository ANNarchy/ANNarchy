from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron
import ANNarchy.core.Global as Global

class Spike2RatePopulation(Population):
    """
    Converts a population of spiking neurons into a population of rate-coded neurons.

    When building a hybrid network, one need to convert spike trains into an instantaneous firing. Creating a ``Spike2RatePopulation`` allows to get a rate-coded population of the same size as the spiking population.

    Each neuron collects the spikes of corresponding spiking neuron over the last milliseconds (defined by the parameter ``window``), and computes the average firing rate in Hz over this sliding window.

    The firing rate ``r`` of each neuron represents by default the firing rate in Hz. The output can be scaled with the parameter ``scaling``. For example, if you want that ``r = 1.0`` represents a firing rate of 100Hz, you can set ``scaling`` to 100.0. The default is 1.0.

    By definition, the firing rate varies abruptly each time a new spike is perceived. The output can be smoothed with a low-pass filter of time constant ``smooth``.

    .. code-block:: python

        from ANNarchy.extensions.hybrid import Spike2RatePopulation

        pop2 = Spike2RatePopulation(
            population=pop1, 
            name='rate-coded', 
            window=50.0, 
            smooth=100.0, 
            scaling=100.0
        )
    """
    def __init__(self, population, name=None, window = 100.0, scaling=1.0, smooth=1.0):
        """
        *Parameters*:

        * **population**: the Population to convert. Its neuron type must be rate-coded.
        * **name**: the (optional) name of the hybrid population.
        * **window**: the extent of the sliding window (in ms) used to compute the firing rate (default: 100.0 ms).
        * **scaling**: the scaling of the firing rate. Defines what a firing rate of 1 Hz outputs (default: 1.0).
        * **smooth**: time constant (in ms) of the low-pass filter used to smooth the firing rate (default: 1 ms, i.e no smoothing)
        """
        self.population = population
        if not self.population.neuron_type.description['type'] == 'spike':
            Global._error('the population ' + self.population.name + ' must contain spiking neurons.')
            exit(0)

        # Create the description, but it will not be used for generation
        Population.__init__(
            self, 
            geometry = self.population.geometry, 
            name=name, 
            neuron = Neuron(
                parameters="""
                    window = %(window)s : population
                    scaling = %(scaling)s : population
                    smooth = %(smooth)s : population
                """ % {'window': window, 'scaling': scaling, 'smooth': smooth} ,
                equations="r = 0.0"
            ) 
        )

        # Generate specific code

        self.generator['omp']['header_pop_struct'] = """ 
struct PopStruct%(id)s{
    // Number of neurons
    int size;

    // Global parameter window
    double  window ;

    // Global parameter scaling
    double  scaling ;

    // Global parameter smooth
    double  smooth ;

    // Local variable r
    std::vector< double > r ;
    std::vector< std::vector< double > > recorded_r ;
    bool record_r ;

    // Store the last spikes
    std::vector< std::deque<long int> > last_spikes;
}; 
""" % {'id' : self.id}


        self.generator['omp']['body_spike_init'] = """ 
    pop%(id)s.last_spikes = std::vector< std::deque<long int> >(pop%(id)s.size, std::deque<long int>());
"""
        self.generator['omp']['body_update_neuron'] = """ 
    // Updating the local variables of Spike2Rate population %(id)s
    #pragma omp parallel for
    for(int i = 0; i < pop%(id)s.size; i++){
        // Increase when spiking
        if (pop%(id_pre)s.last_spike[i] == t-1){
           pop%(id)s.last_spikes[i].push_front(t-1);
        }
        int nb = 0;
        for(int j=0; j < pop%(id)s.last_spikes[i].size(); j++){
            if(pop%(id)s.last_spikes[i][j] > t - int(pop%(id)s.window*dt) ){
                nb++;
            }
            else{
                pop%(id)s.last_spikes[i].erase(pop%(id)s.last_spikes[i].begin()+j);
            }
        }
        pop%(id)s.r[i] += dt*(1000.0/pop%(id)s.scaling / pop%(id)s.window * nb - pop%(id)s.r[i] ) / pop%(id)s.smooth;
    }
"""  % {'id' : self.id, 'id_pre': self.population.id}


class Rate2SpikePopulation(Population):
    """
    Converts a population of rate-coded neurons into a population of spiking neurons.

    This class allows to generate spike trains based on the computations of a rate-coded network (for example doing visual pre-processing). Creating a ``Rate2SpikePopulation`` allows to get a spiking population of the same size as the rate-coded population.

    The firing rate ``r`` of the rate-coded population represents by default the desired firing rate in Hz. This value can be scaled with the parameter ``scaling``. For example, if you want that ``r = 1.0`` represents a firing rate of 100Hz, you can set ``scaling`` to 100.0.

    .. code-block:: python

        from ANNarchy.extensions.hybrid import Rate2SpikePopulation
        pop2 = Rate2SpikePopulation(
            population=pop1, 
            name='spiking', 
            scaling=100.0
        )
    """
    def __init__(self, population, name=None, scaling=1.0):
        """
        *Parameters*:

        * **population**: the Population to convert. Its neuron type must be spiking.
        * **name**: the (optional) name of the hybrid population.
        * **scaling**: the scaling of the firing rate. Defines what a rate ``r`` of 1.0 means in Hz (default: 1.0).
        """
        self.population = population

        if not self.population.neuron_type.description['type'] == 'rate':
            Global._error('the population ' + self.population.name + ' must contain rate-coded neurons.')
            exit(0)

        # Create the description, but it will not be used for generation
        Population.__init__(
            self, 
            geometry = self.population.geometry, 
            name=name, 
            neuron = Neuron(
                parameters="""
                    scaling = %(scaling)s : population
                """ % {'scaling': scaling} ,
                equations="""
                    p = Uniform(0.0, 1.0)
                    rates = p
                """,
                spike="rates>p"
            ) 
        )

        # Generate the code
        self.generator['omp']['body_update_neuron'] = """ 
    // Updating the local variables of population %(id)s (Rate2SpikePopulation)
    #pragma omp parallel for
    for(int i = 0; i < pop%(id)s.size; i++){

        pop%(id)s.rates[i] = pop%(id_pre)s.r[i] * pop%(id)s.scaling;

        if(pop%(id)s.rates[i] > pop%(id)s.rand_0[i]*1000.0/dt){
            pop%(id)s.spike[i] = true;
            pop%(id)s.last_spike[i] = t;
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
""" % {'id' : self.id, 'id_pre': self.population.id}

