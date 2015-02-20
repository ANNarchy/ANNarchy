from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron
import ANNarchy.core.Global as Global

class Spike2RatePopulation(Population):
    """
    Converts a population of spiking neurons into a population of rate-coded neurons.

    When building a hybrid network, one may need to convert spike trains into an instantaneous firing. Creating a ``Spike2RatePopulation`` allows to get a rate-coded population of the same size as the spiking population.

    Two modes are available:

    * ``window``: at each each time step t, the number of spikes in the last T milliseconds (defined by the parameter ``window``) is counted and divided by the interval.

    * ``isi``: the inter-spike interval is computed for each new spike, and filtered out through an asymmetrical kernel function. 


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
    def __init__(self, population, name=None, mode='window', window = 100.0, scaling=1.0, smooth=1.0, cut=3.0):
        """
        *Parameters*:

        * **population**: the Population to convert. Its neuron type must be rate-coded.
        * **name**: the (optional) name of the hybrid population.
        * **mode**: mode of computation of the firing rate. ``'window'`` (default) or ``'isi'``.
        * **window**: the extent of the sliding window (in ms) used to compute the firing rate in the 'window' mode(default: 100.0 ms).
        * **cut**: cutting frequency of the ``'isi'`` kernel. default 2.0.
        * **scaling**: the scaling of the firing rate. Defines what a firing rate of 1 Hz outputs (default: 1.0).
        * **smooth**: time constant (in ms) of the low-pass filter used to smooth the firing rate (default: 1 ms, i.e no smoothing)
        """
        self.population = population
        self.name = name
        self.mode = mode
        self.scaling = scaling
        self.window = window
        self.smooth = smooth
        self.cut = cut

        if not self.population.neuron_type.description['type'] == 'spike':
            Global._error('the population ' + self.population.name + ' must contain spiking neurons.')
            exit(0)

        if self.mode == 'window':
            self._create_window()
        elif self.mode == 'adaptive':
            self._create_adaptive()
        elif self.mode == 'isi':
            self._create_isi()
        else:
            Global._error('Spike2RatePopulation: Unknown method ' + self.mode)
            exit(0)

    def _create_isi(self):

        # Create the description, but it will not be used for generation
        Population.__init__(
            self, 
            geometry = self.population.geometry, 
            name=self.name, 
            neuron = Neuron(
                parameters="""
                    cut = %(cut)s : population
                    scaling = %(scaling)s : population
                    smooth = %(smooth)s : population
                """ % {'cut': self.cut, 'scaling': self.scaling, 'smooth': self.smooth} ,
                equations="r = 0.0"
            ) 
        )

        # Generate specific code

        self.generator['omp']['header_pop_struct'] = """ 
struct PopStruct%(id)s{
    // Number of neurons
    int size;
    bool _active;

    // Record parameter
    int record_period;
    long int record_offset;

    // Global parameter cut
    double  cut ;

    // Global parameter scaling
    double  scaling ;

    // Global parameter smooth
    double  smooth ;

    // Local variable r
    std::vector< double > r ;
    std::vector< std::vector< double > > recorded_r ;
    bool record_r ;

    // Store the last spike
    std::vector< long int > last_spike;
    std::vector< double > isi;
    std::vector< double > support;
}; 
""" % {'id' : self.id}



        self.generator['omp']['body_spike_init'] = """ 
    pop%(id)s.last_spike = std::vector< long int >(pop%(id)s.size, -10000L);
    pop%(id)s.isi = std::vector< double >(pop%(id)s.size, 10000.0);
    pop%(id)s.support = std::vector< double >(pop%(id)s.size, 10000.0);
"""

        omp_code = "#pragma omp parallel for" if Global.config['num_threads'] > 1 else ""

        self.generator['omp']['body_update_neuron'] = """ 
    // Updating the local variables of Spike2Rate population %(id)s
    if(pop%(id)s._active){
        %(omp_code)s
        for(int i = 0; i < pop%(id)s.size; i++){
            // Increase when spiking
            if (pop%(id_pre)s.last_spike[i] == t-1){
               pop%(id)s.isi[i] = double(t-1 - pop%(id)s.last_spike[i]);
               pop%(id)s.support[i] = double(t-1 - pop%(id)s.last_spike[i]);
               pop%(id)s.last_spike[i] = t-1;
            }
            else if( double(t - pop%(id)s.last_spike[i]) <= pop%(id)s.isi[i]){
                    // do nothing
            }
            else if( double(t - pop%(id)s.last_spike[i]) <= pop%(id)s.cut*pop%(id)s.isi[i]){
                    pop%(id)s.support[i] += 1.0 ;
            }
            else{
                pop%(id)s.support[i] = 10000.0 ;    
            }
            
            pop%(id)s.r[i] += dt*(1000.0/pop%(id)s.scaling/pop%(id)s.support[i]/dt - pop%(id)s.r[i])/pop%(id)s.smooth;
        }
    }
"""  % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code}



    def _create_window(self):

        # Create the description, but it will not be used for generation
        Population.__init__(
            self, 
            geometry = self.population.geometry, 
            name=self.name, 
            neuron = Neuron(
                parameters="""
                    window = %(window)s : population
                    scaling = %(scaling)s : population
                    smooth = %(smooth)s : population
                """ % {'window': self.window, 'scaling': self.scaling, 'smooth': self.smooth} ,
                equations="r = 0.0"
            ) 
        )

        # Generate specific code

        self.generator['omp']['header_pop_struct'] = """ 
struct PopStruct%(id)s{
    // Number of neurons
    int size;
    bool _active;

    // Record parameter
    int record_period;
    long int record_offset;

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
    std::vector< std::vector<long int> > last_spikes;
}; 
""" % {'id' : self.id}


        self.generator['omp']['body_spike_init'] = """ 
    pop%(id)s.last_spikes = std::vector< std::vector<long int> >(pop%(id)s.size, std::vector<long int>());
"""
        omp_code = "#pragma omp parallel for private(pop%(id)s_nb, pop%(id)s_out)" if Global.config['num_threads'] > 1 else ""

        self.generator['omp']['body_update_neuron'] = """ 
    // Updating the local variables of Spike2Rate population %(id)s
    if(pop%(id)s._active){
        int pop%(id)s_nb, pop%(id)s_out;
        %(omp_code)s
        for(int i = 0; i < pop%(id)s.size; i++){
            // Increase when spiking
            if (pop%(id_pre)s.last_spike[i] == t-1){
               pop%(id)s.last_spikes[i].push_back(t-1);
            }
            pop%(id)s_nb = 0;
            pop%(id)s_out = -1;
            for(int j=0; j < pop%(id)s.last_spikes[i].size(); j++){
                if(pop%(id)s.last_spikes[i][j] >= t -1 - (long int)(pop%(id)s.window/dt) ){
                    pop%(id)s_nb++;
                }
                else{
                    pop%(id)s_out = j;
                }
            }

            if (pop%(id)s_out > -1){
                pop%(id)s.last_spikes[i].erase(pop%(id)s.last_spikes[i].begin());
            }

            pop%(id)s.r[i] += dt*(1000.0/pop%(id)s.scaling / pop%(id)s.window * double(pop%(id)s_nb) - pop%(id)s.r[i] ) / pop%(id)s.smooth;
        }
    }
"""  % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code}


    def _create_adaptive(self):

        # Create the description, but it will not be used for generation
        Population.__init__(
            self, 
            geometry = self.population.geometry, 
            name=self.name, 
            neuron = Neuron(
                parameters="""
                    window = %(window)s : population
                    scaling = %(scaling)s : population
                    smooth = %(smooth)s : population
                """ % {'window': self.window, 'scaling': self.scaling, 'smooth': self.smooth} ,
                equations="r = 0.0"
            ) 
        )

        # Generate specific code

        self.generator['omp']['header_pop_struct'] = """ 
struct PopStruct%(id)s{
    // Number of neurons
    int size;
    bool _active;

    // Record parameter
    int record_period;
    long int record_offset;

    // Local parameter window
    double  window ;
    std::vector< double >  ad_window ;

    // Global parameter scaling
    double  scaling ;

    // Global parameter smooth
    double  smooth ;

    // Local variable r
    std::vector< double > r ;
    std::vector< std::vector< double > > recorded_r ;
    bool record_r ;

    // Store the last spikes
    std::vector< std::vector<long int> > last_spikes;
    std::vector< double > isi;
}; 
""" % {'id' : self.id}


        omp_code = "#pragma omp parallel for private(pop%(id)s_nb, pop%(id)s_out)" if Global.config['num_threads'] > 1 else ""

        self.generator['omp']['body_spike_init'] = """ 
    pop%(id)s.last_spikes = std::vector< std::vector<long int> >(pop%(id)s.size, std::vector<long int>());
    pop%(id)s.ad_window = std::vector< double >(pop%(id)s.size, pop%(id)s.window);
    pop%(id)s.isi = std::vector< double >(pop%(id)s.size, 10000.0);
"""
        self.generator['omp']['body_update_neuron'] = """ 
    // Updating the local variables of Spike2Rate population %(id)s
    if(pop%(id)s._active){
        int pop%(id)s_nb, pop%(id)s_out;
        %(omp_code)s
        for(int i = 0; i < pop%(id)s.size; i++){
            // Increase when spiking
            if (pop%(id_pre)s.last_spike[i] == t-1){
                if(pop%(id)s.last_spikes[i].size() > 0)
                    pop%(id)s.isi[i] = double(t-1 - pop%(id)s.last_spikes[i][pop%(id)s.last_spikes[i].size()-1]);
                else
                    pop%(id)s.isi[i] = 10000.0;
                pop%(id)s.last_spikes[i].push_back(t-1);
            }
            pop%(id)s_nb = 0;
            pop%(id)s_out = -1;
            for(int j=0; j < pop%(id)s.last_spikes[i].size(); j++){
                if(pop%(id)s.last_spikes[i][j] >= t -1 - (long int)(pop%(id)s.ad_window[i]/dt) ){
                    pop%(id)s_nb++;
                }
                else{
                    pop%(id)s_out = j;
                }
            }

            if (pop%(id)s_out > -1){
                pop%(id)s.last_spikes[i].erase(pop%(id)s.last_spikes[i].begin(), pop%(id)s.last_spikes[i].begin()+pop%(id)s_out);
            }

            pop%(id)s.r[i] += dt*(1000.0/pop%(id)s.scaling / pop%(id)s.ad_window[i] * double(pop%(id)s_nb) - pop%(id)s.r[i] ) / pop%(id)s.smooth;

            //pop%(id)s.ad_window[i] = clip(5.0*pop%(id)s.isi[i], 20.0*dt, pop%(id)s.window) ;
            pop%(id)s.ad_window[i] += dt * (clip(5.0*pop%(id)s.isi[i], 20.0*dt, pop%(id)s.window) - pop%(id)s.ad_window[i])/100.0;
            if (i==0)
                std::cout << pop%(id)s.ad_window[i] << " " << pop%(id)s.isi[i] << std::endl;

        }
    }
"""  % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code}


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
    def __init__(self, population, name=None, scaling=1.0, refractory=None):
        """
        *Parameters*:

        * **population**: the Population to convert. Its neuron type must be spiking.
        * **name**: the (optional) name of the hybrid population.
        * **scaling**: the scaling of the firing rate. Defines what a rate ``r`` of 1.0 means in Hz (default: 1.0).
        * **refractory**: a refractory period in ms to ensure the ISI is not too high (default: None)
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
                spike="rates>p",
                refractory=refractory
            ) 
        )

        omp_code = "#pragma omp parallel for" if Global.config['num_threads'] > 1 else ""

        # Generate the code
        self.generator['omp']['body_update_neuron'] = """ 
    // Updating the local variables of population %(id)s (Rate2SpikePopulation)
    if(pop%(id)s._active){
        %(omp_code)s
        for(int i = 0; i < pop%(id)s.size; i++){

            pop%(id)s.rates[i] = pop%(id_pre)s.r[i] * pop%(id)s.scaling;

                
            if(pop%(id)s.refractory_remaining[i] > 0){ // Refractory period

                pop%(id)s.refractory_remaining[i]--;
            }
            else if(pop%(id)s.rates[i] > pop%(id)s.rand_0[i]*1000.0/dt){
                #pragma omp critical
                {
                    pop%(id)s.spiked.push_back(i);
                    if(pop%(id)s.record_spike){
                        pop%(id)s.recorded_spike[i].push_back(t);
                    }
                }
                pop%(id)s.last_spike[i] = t;
                pop%(id)s.refractory_remaining[i] = pop%(id)s.refractory[i];
            }
        }
    }
""" % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code}

