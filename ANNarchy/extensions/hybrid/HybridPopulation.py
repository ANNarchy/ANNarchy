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
    def __init__(self, population, name=None, mode='window', window = 100.0, scaling=1.0, smooth=1.0, cut=3.0, copied=False):
        """
        :param population: the Population to convert. Its neuron type must be rate-coded.
        :param name: the (optional) name of the hybrid population.
        :param mode: mode of computation of the firing rate. ``'window'`` (default) or ``'isi'``.
        :param window: the extent of the sliding window (in ms) used to compute the firing rate in the 'window' mode(default: 100.0 ms).
        :param cut: cutting frequency of the ``'isi'`` kernel. default 2.0.
        :param scaling: the scaling of the firing rate. Defines what a firing rate of 1 Hz outputs (default: 1.0).
        :param smooth: time constant (in ms) of the low-pass filter used to smooth the firing rate (default: 1 ms, i.e no smoothing)
        """
        self.population = population
        self.name = name
        self.mode = mode
        self.scaling = scaling
        self.window = window
        self.smooth = smooth
        self.cut = cut
        self.copied = copied

        if not self.population.neuron_type.description['type'] == 'spike':
            Global._error('the population ' + self.population.name + ' must contain spiking neurons.')
            
        if self.mode == 'window':
            self._code = self._create_window()
        elif self.mode == 'adaptive':
            self._code = self._create_adaptive()
        elif self.mode == 'isi':
            self._code = self._create_isi()
        else:
            Global._error('Spike2RatePopulation: Unknown method ' + self.mode)
            

        self._specific = True

    def _copy(self):
        "Returns a copy of the population when creating networks. Internal use only."
        return Spike2RatePopulation(population=self.population, name=self.name, mode=self.mode, window=self.window, scaling=self.scaling, smooth=self.smooth, cut=self.cut, copied=True)
    
    def generate(self):
        """
        return the corresponding code template
        """
        return self._code

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
            ),
            copied=self.copied
        )

        # Generate specific code
        omp_code = "#pragma omp parallel for" if Global.config['num_threads'] > 1 else ""
        code = """#pragma once

#include "pop%(id_pre)s.hpp"
extern PopStruct%(id_pre)s pop%(id_pre)s;

struct PopStruct%(id)s{
    // Number of neurons
    int size;
    bool _active;

    // Global parameter cut
    %(float_prec)s cut;

    // Global parameter scaling
    %(float_prec)s  scaling;

    // Global parameter smooth
    %(float_prec)s  smooth ;

    // Local variable r
    std::vector< %(float_prec)s > r ;

    // Store the last spike
    std::vector< long int > last_spike;
    std::vector< %(float_prec)s > isi;
    std::vector< %(float_prec)s > support;

    int get_size() { return size; }
    void set_size(int value) { size = value; }

    bool is_active() { return _active; }
    void set_active(bool value) { _active = value; }

    %(float_prec)s get_cut() { return cut; }
    void set_cut(%(float_prec)s value) { cut = value; }

    %(float_prec)s get_scaling() { return scaling; }
    void set_scaling(%(float_prec)s value) { scaling = value; }

    %(float_prec)s get_smooth() { return smooth; }
    void set_smooth(%(float_prec)s value) { smooth = value; }

    std::vector< %(float_prec)s > get_r() { return r; }
    void set_r(std::vector< %(float_prec)s > value) { r = value; }
    %(float_prec)s get_single_r(int rk) { return r[rk]; }
    void set_single_r(int rk, %(float_prec)s value) { r[rk] = value; }

    void init_population() {
        size = %(size)s;
        _active = true;

        last_spike = std::vector< long int >(size, -10000L);
        isi = std::vector< %(float_prec)s >(size, 10000.0);
        support = std::vector< %(float_prec)s >(size, 10000.0);
    }

    void update_rng() {

    }

    void update() {
        // Updating the local variables of Spike2Rate population %(id)s
        if(_active){
            %(omp_code)s
            for(int i = 0; i < size; i++){
                // Increase when spiking
                if (pop%(id_pre)s.last_spike[i] == t-1){
                   isi[i] = %(float_prec)s(t-1 - last_spike[i]);
                   support[i] = %(float_prec)s(t-1 - last_spike[i]);
                   last_spike[i] = t-1;
                }
                else if( %(float_prec)s(t - last_spike[i]) <= isi[i]){
                        // do nothing
                }
                else if( %(float_prec)s(t - last_spike[i]) <= cut*isi[i]){
                        support[i] += 1.0 ;
                }
                else{
                    support[i] = 10000.0 ;
                }

                r[i] += dt*(1000.0/scaling/support[i]/dt - r[i])/smooth;
            }
        }
    }
};
""" % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code, 'size': self.size, 'float_prec': Global.config['precision'] }

        return code

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
            ) ,
            copied=self.copied
        )

        # Generate specific code
        omp_code = "#pragma omp parallel for" if Global.config['num_threads'] > 1 else ""
        code = """#pragma once

#include "pop%(id_pre)s.hpp"
extern PopStruct%(id_pre)s pop%(id_pre)s;

struct PopStruct%(id)s{
    // Number of neurons
    int size;
    bool _active;

    // Global parameter window
    %(float_prec)s  window ;

    // Global parameter scaling
    %(float_prec)s  scaling ;

    // Global parameter smooth
    %(float_prec)s  smooth ;

    // Local variable r
    std::vector< %(float_prec)s > r ;

    // Store the last spikes
    std::vector< std::vector<long int> > last_spikes;

    int get_size() { return size; }
    void set_size(int value) { size = value; }

    bool is_active() { return _active; }
    void set_active(bool value) { _active = value; }

    %(float_prec)s get_window() { return window; }
    void set_window(%(float_prec)s value) { window = value; }

    %(float_prec)s get_scaling() { return scaling; }
    void set_scaling(%(float_prec)s value) { scaling = value; }

    %(float_prec)s get_smooth() { return smooth; }
    void set_smooth(%(float_prec)s value) { smooth = value; }

    std::vector< %(float_prec)s > get_r() { return r; }
    void set_r(std::vector< %(float_prec)s > value) { r = value; }
    %(float_prec)s get_single_r(int rk) { return r[rk]; }
    void set_single_r(int rk, %(float_prec)s value) { r[rk] = value; }

    void init_population() {
        size = %(size)s;
        _active = true;

        last_spikes = std::vector< std::vector<long int> >(size, std::vector<long int>());
    }

    void update_rng() {

    }

    void update() {
        // Updating the local variables of Spike2Rate population %(id)s
        if(_active){
            int pop%(id)s_nb, pop%(id)s_out;
            %(omp_code)s
            for(int i = 0; i < size; i++){
                // Increase when spiking
                if (pop%(id_pre)s.last_spike[i] == t-1){
                   last_spikes[i].push_back(t-1);
                }
                pop%(id)s_nb = 0;
                pop%(id)s_out = -1;
                for(int j=0; j < last_spikes[i].size(); j++){
                    if(last_spikes[i][j] >= t -1 - (long int)(window/dt) ){
                        pop%(id)s_nb++;
                    }
                    else{
                        pop%(id)s_out = j;
                    }
                }

                if (pop%(id)s_out > -1){
                    last_spikes[i].erase(last_spikes[i].begin());
                }

                r[i] += dt*(1000.0/scaling / window * %(float_prec)s(pop%(id)s_nb) - r[i] ) / smooth;
            }
        }
    }
};
""" % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code, 'size': self.size, 'float_prec': Global.config['precision']}

        return code

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
            ) ,
            copied=self.copied
        )

        # Generate specific code
        omp_code = "#pragma omp parallel for private(pop%(id)s_nb, pop%(id)s_out)" if Global.config['num_threads'] > 1 else ""
        code = """#pragma once

#include "pop%(id_pre)s.hpp"
extern PopStruct%(id_pre)s pop%(id_pre)s;

struct PopStruct%(id)s{
    // Number of neurons
    int size;
    bool _active;

    // Local parameter window
    %(float_prec)s  window ;
    std::vector< %(float_prec)s >  ad_window ;

    // Global parameter scaling
    %(float_prec)s  scaling ;

    // Global parameter smooth
    %(float_prec)s  smooth ;

    // Local variable r
    std::vector< %(float_prec)s > r ;
    std::vector< std::vector< %(float_prec)s > > recorded_r ;
    bool record_r ;

    // Store the last spikes
    std::vector< std::vector<long int> > last_spikes;
    std::vector< %(float_prec)s > isi;

    int get_size() { return size; }
    void set_size(int value) { size = value; }

    bool is_active() { return _active; }
    void set_active(bool value) { _active = value; }

    %(float_prec)s get_window() { return window; }
    void set_window(%(float_prec)s value) { window = value; }

    %(float_prec)s get_scaling() { return scaling; }
    void set_scaling(%(float_prec)s value) { scaling = value; }

    %(float_prec)s get_smooth() { return smooth; }
    void set_smooth(%(float_prec)s value) { smooth = value; }

    std::vector< %(float_prec)s > get_r() { return r; }
    void set_r(std::vector< %(float_prec)s > value) { r = value; }
    %(float_prec)s get_single_r(int rk) { return r[rk]; }
    void set_single_r(int rk, %(float_prec)s value) { r[rk] = value; }

    void init_population() {
        size = %(size)s;
        _active = true;

        last_spikes = std::vector< std::vector<long int> >(size, std::vector<long int>());
        ad_window = std::vector< %(float_prec)s >(size, window);
        isi = std::vector< %(float_prec)s >(size, 10000.0);
    }

    void update_rng() {

    }

    void update() {
        // Updating the local variables of Spike2Rate population %(id)s
        if(_active){
            int pop%(id)s_nb, pop%(id)s_out;
            %(omp_code)s
            for(int i = 0; i < size; i++){
                // Increase when spiking
                if (pop%(id_pre)s.last_spike[i] == t-1){
                    if(last_spikes[i].size() > 0)
                        isi[i] = %(float_prec)s(t-1 - last_spikes[i][last_spikes[i].size()-1]);
                    else
                        isi[i] = 10000.0;
                    last_spikes[i].push_back(t-1);
                }
                pop%(id)s_nb = 0;
                pop%(id)s_out = -1;
                for(int j=0; j < last_spikes[i].size(); j++){
                    if(last_spikes[i][j] >= t -1 - (long int)(ad_window[i]/dt) ){
                        pop%(id)s_nb++;
                    }
                    else{
                        pop%(id)s_out = j;
                    }
                }

                if (pop%(id)s_out > -1){
                    last_spikes[i].erase(last_spikes[i].begin(), last_spikes[i].begin()+pop%(id)s_out);
                }

                r[i] += dt*(1000.0/scaling / ad_window[i] * %(float_prec)s(pop%(id)s_nb) - r[i] ) / smooth;

                //ad_window[i] = clip(5.0*isi[i], 20.0*dt, window) ;
                ad_window[i] += dt * (clip(5.0*isi[i], 20.0*dt, window) - ad_window[i])/100.0;
                if (i==0)
                    std::cout << ad_window[i] << " " << isi[i] << std::endl;
            }
        }
    }
};
""" % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code, 'size': self.size, 'float_prec': Global.config['precision']}

        return code

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
    def __init__(self, population, name=None, scaling=1.0, refractory=None, copied=False):
        """
        :param population: the Population to convert. Its neuron type must be spiking.
        :param name: the (optional) name of the hybrid population.
        :param scaling: the scaling of the firing rate. Defines what a rate ``r`` of 1.0 means in Hz (default: 1.0).
        :param refractory: a refractory period in ms to ensure the ISI is not too high (default: None)
        """
        self.population = population
        self.refractory_init = refractory

        if not self.population.neuron_type.description['type'] == 'rate':
            Global._error('the population ' + self.population.name + ' must contain rate-coded neurons.')
            

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
            ) ,
            copied=self.copied
        )

        self._specific = True

    def _copy(self):
        "Returns a copy of the population when creating networks. Internal use only."
        return Rate2SpikePopulation(population=self.population, name=self.name, scaling=self.scaling, refractory=self.refractory_init, copied=True)

    def generate(self):
        omp_code = "#pragma omp parallel for" if Global.config['num_threads'] > 1 else ""
        omp_critical = "#pragma omp critical" if Global.config['num_threads'] > 1 else ""

        # Generate the code
        code = """#pragma once

#include "pop%(id_pre)s.hpp"
extern PopStruct%(id_pre)s pop%(id_pre)s;

struct PopStruct1{
    // Number of neurons
    int size;
    bool _active;

    // Global parameter scaling
    %(float_prec)s  scaling ;

    // Local variable p
    std::vector< %(float_prec)s > p;

    // Local variable rates
    std::vector< %(float_prec)s > rates;

    // Random numbers (STL implementation)
    std::vector< %(float_prec)s > rand_0;
    std::uniform_real_distribution< %(float_prec)s > dist_rand_0;

    // Spiking events
    std::vector<long int> last_spike;
    std::vector<int> spiked;
    std::vector<int> refractory;
    std::vector<int> refractory_remaining;
    bool record_spike;
    std::vector<std::vector<long> > recorded_spike;

    int get_size() { return size; }
    bool is_active() { return _active; }
    bool set_active(bool val) { _active = val; }

    // Global parameter scaling
    %(float_prec)s get_scaling() { return scaling; }
    void set_scaling(%(float_prec)s val) { scaling = val; }

    // Local variable p
    std::vector< %(float_prec)s > get_p() { return p; }
    %(float_prec)s get_single_p(int rk) { return p[rk]; }
    void set_p(std::vector< %(float_prec)s > val) { p = val; }
    void set_single_p(int rk, %(float_prec)s val) { p[rk] = val; }

    // Local variable rates
    std::vector< %(float_prec)s > get_rates() { return rates; }
    %(float_prec)s get_single_rates(int rk) { return rates[rk]; }
    void set_rates(std::vector< %(float_prec)s > val) { rates = val; }
    void set_single_rates(int rk, %(float_prec)s val) { rates[rk] = val; }

    void init_population() {
        size = %(size)s;
        _active = true;

        refractory = std::vector<int>(size, 0);
        spiked = std::vector<int>(0, 0);
        last_spike = std::vector<long int>(size, -10000L);
        refractory_remaining = std::vector<int>(size, 0);

        rand_0 = std::vector< %(float_prec)s >(size, 0.0);
        dist_rand_0 = std::uniform_real_distribution< %(float_prec)s >(0.0, 1.0);
    }

    void update_rng() {
        for(int i = 0; i < size; i++)
        {
            rand_0[i] = dist_rand_0(rng);
        }
    }

    void update() {
        // Updating the local variables of population %(id)s (Rate2SpikePopulation)
        if(_active){
            spiked.clear();
            %(omp_code)s
            for(int i = 0; i < size; i++){
                rates[i] = pop%(id_pre)s.r[i] * scaling;

                if(refractory_remaining[i] > 0){ // Refractory period
                    refractory_remaining[i]--;
                }
                else if(rates[i] > rand_0[i]*1000.0/dt){
                    %(omp_critical)s
                    {
                        spiked.push_back(i);
                    }
                    last_spike[i] = t;
                    refractory_remaining[i] = refractory[i];
                }
            }
        }
    }
};
""" % {'id' : self.id, 'id_pre': self.population.id, 'omp_code': omp_code, 'omp_critical': omp_critical, 'size': self.size, 'float_prec': Global.config['precision'] }

        return code