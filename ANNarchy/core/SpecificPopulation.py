#===============================================================================
#
#     SpecificPopulation.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron, RateNeuron
import ANNarchy.core.Global as Global

import numpy as np
from scipy.special import erf
from scipy.optimize import newton

class SpecificPopulation(Population):
    """
    Interface class for user-defined definition of Population objects. An inheriting
    class need to override the implementator functions _generate_[paradigm], otherwise
    a NotImplementedError exception will be thrown.

    *Parameters*:

        * geometry *:
        * neuron *:
        * name *:
    """
    def __init__(self, geometry, neuron, name=None):
        Population.__init__(self, geometry, neuron, name)

    def _generate(self):
        if Global.config['paradigm'] == "openmp":
            self._generate_omp()
        elif Global.config['paradigm'] == "cuda":
            self._generate_cuda()
        else:
            raise NotImplementedError

    def _generate_omp(self):
        " Overridden by child class "
        raise NotImplementedError

    def _generate_cuda(self):
        " Overridden by child class "
        raise NotImplementedError

class PoissonPopulation(SpecificPopulation):
    """ 
    Population of spiking neurons following a Poisson distribution.

    **Case 1:** Input population

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

    **Case 2:** Hybrid population

    If the ``rates`` argument is not set, the population can be used as an interface from a rate-coded population. 

    The ``target`` argument specifies which incoming projections will be summed to determine the instantaneous firing rate of each neuron.

    See the example in ``examples/hybrid/Hybrid.py`` for a usage.

    """

    def __init__(self, geometry, name=None, rates=None, target=None, parameters=None, refractory=None):
        """        
        *Parameters*:
        
        * **geometry**: population geometry as tuple. 

        * **name**: unique name of the population (optional).

        * **rates**: mean firing rate of each neuron. It can be a single value (e.g. 10.0) or an equation (as string).

        * **target**: the mean firing rate will be the weighted sum of inputs having this target name (e.g. "exc").

        * **parameters**: additional parameters which can be used in the *rates* equation.

        * **refractory**: refractory period in ms.
        """  
        if rates is None and target is None:
            Global._error('A PoissonPopulation must define either rates or target.')
            

        if target is not None: # hybrid population
            # Create the neuron
            poisson_neuron = Neuron(
                parameters = """
                %(params)s
                """ % {'params': parameters if parameters else ''},
                equations = """
                rates = sum(%(target)s)
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                _sum_%(target)s = 0.0
                """ % {'target': target},
                spike = """
                    p < rates
                """,
                refractory=refractory,
                name="Hybrid",
                description="Hybrid spiking neuron emitting spikes according to a Poisson distribution at a frequency determined by the weighted sum of inputs."
            )


        elif isinstance(rates, str):
            # Create the neuron
            poisson_neuron = Neuron(
                parameters = """
                %(params)s
                """ % {'params': parameters if parameters else ''},
                equations = """
                rates = %(rates)s
                p = Uniform(0.0, 1.0) * 1000.0 / dt
                _sum_exc = 0.0
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
        SpecificPopulation.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name)
        
        if isinstance(rates, np.ndarray):
            self.rates = rates

    def _generate_omp(self):
        " Nothing special to do here. "
        pass

    def _generate_cuda(self):
        " Nothing special to do here. "
        pass

class TimedArray(SpecificPopulation):
    """
    Timed array source setting provided firing rates at the times given in the schedule array.

    * Parameters *:

        *geometry*: population geometry
        *schedule*: either a scalar or a set of time points where inputs should be set.
        *values*: inputs to be set.
        *periodic*: if the simulation time exceeds the last point of the schedule plan, the state of firing rates need to be defined.
        By default, the last set firing rate will remain. If periodic is set to true, the schedule will be applied from the beginning
        again. The same applies for the values.

    * Note *:

        Until now, this specific population is not available for CUDA ( will be changed soon ).
    """
    def __init__(self, geometry, schedule, values, periodic=False, name=None):
        neuron = Neuron(
            parameters=" r ",
            equations="",
            name="Timed Array",
            description="Timed array source."
        )

        if isinstance(schedule, (int, float)):
            schedule = [ schedule for i in xrange(values.shape[0])]

        if len(schedule) != values.shape[0]:
            Global._error('TimedArray: length of schedule parameter and 1st dimension of values parameter should be the same')

        SpecificPopulation.__init__(self, geometry=geometry, neuron=neuron, name=name)

        self.init['schedule'] = schedule
        self.init['values'] = values
        self.init['periodic'] = periodic

    def _generate_omp(self):
        " adjust code templates for the specific population "
        self._specific_template['declare_additional'] = """
    // Custom local parameter timed array
    std::vector< int > _schedule;
    std::vector< std::vector< double > > _buffer;
    bool _periodic;
    int _curr_slice;
    int _curr_cnt;
"""
        self._specific_template['access_additional'] = """
    // Custom local parameter timed array
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< double > > buffer) { _buffer = buffer; r = _buffer[0]; }
    std::vector< std::vector< double > > get_buffer() { return _buffer; }
    void set_periodic(bool periodic) { _periodic = periodic; }
    bool get_periodic() { return _periodic; }
"""
        self._specific_template['init_additional'] = """
        // counters
        _curr_slice = 0;
        _curr_cnt = 1;
        _periodic = false;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters timed array
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[double]])
        vector[vector[double]] get_buffer()
        void set_periodic(bool)
        bool get_periodic()
"""
        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters timed array
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_values( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_values( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_periodic( self, periodic ):
        pop%(id)s.set_periodic(periodic)
    cpdef bool get_periodic(self):
        return pop%(id)s.get_periodic()
""" % { 'id': self.id }
        self._specific_template['update_variables'] = """
        if ( _curr_slice == -1 )
            return;

        if ( _curr_cnt < _schedule[_curr_slice] ) {
            _curr_cnt++;
        } else {
            if ( ++_curr_slice == _schedule.size() ) {
                if ( _periodic ) {
                    _curr_slice = 0;
                } else {
                    _curr_slice = -1;
                    return;
                }
            }
            _curr_cnt=1;
            r = _buffer[_curr_slice];
        }
"""

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size)

    def __setattr__(self, name, value):
        if name == 'schedule':
            if self.initialized:
                self.cyInstance.set_schedule( np.array(value) / Global.config['dt'] )
            else:
                self.init['schedule'] = value
        elif name == 'values':
            if self.initialized:
                if len(value.shape) > 2:
                    # we need to flatten the provided data
                    flat_values = np.zeros( (value.shape[0], self.size) )
                    for x in xrange(value.shape[0]):
                        flat_values[x] = np.reshape( value[x], self.size )
                    self.cyInstance.set_values( flat_values )
                else:
                    self.cyInstance.set_values( value )
            else:
                self.init['values'] = value
        elif name == "periodic":
            if self.initialized:
                self.cyInstance.set_periodic( value )
            else:
                self.init['periodic'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                return Global.config['dt'] * self.cyInstance.get_schedule()
            else:
                return self.init['schedule']
        elif name == 'values':
            if self.initialized:
                if len(self.geometry) > 1:
                    # unflatten the data
                    flat_values = self.cyInstance.get_values()
                    values = np.zeros( tuple( [len(self.schedule)] + list(self.geometry) ) )
                    for x in range(len(self.schedule)):
                        values[x] = np.reshape( flat_values[x], self.geometry)
                    return values
                else:
                    return self.cyInstance.get_values()
            else:
                return self.init['values']
        elif name == 'periodic':
            if self.initialized:
                return self.cyInstance.get_periodic()
            else:
                return self.init['periodic']
        else:
            return Population.__getattribute__(self, name)

class SpikeSourceArray(SpecificPopulation):
    """
    Spike source generating spikes at the times given in the spike_times array.

    Depending on the initial array provided, the population will have one or several neurons, but the geometry can only be one-dimensional.

    *Parameters*:

    * **spike_times** : a list of absolute times at which a spike should be emitted if the population has 1 neuron, a list of lists otherwise. 
    Times are defined in milliseconds, and will be rounded to the closest multiple of the discretization time step dt.
    * **name**: optional name for the population.

    You can later modify the spike_times attribute of the population, but it must have the same size as the initial one::
    """
    def __init__(self, spike_times, name=None):

        if not isinstance(spike_times, list):
            Global._error('In a SpikeSourceArray, spike_times must be a Python list.')
            

        if isinstance(spike_times[0], list): # several neurons
            nb_neurons = len(spike_times)
        else: # a single Neuron
            nb_neurons = 1
            spike_times = [ spike_times ]

        # Create a fake neuron just to be sure the description has the correct parameters
        neuron = Neuron(
            parameters="""
                spike_times = 0.0 : int
            """,
            equations="",
            spike=" t == spike_times",
            reset="",
            name="Spike source",
            description="Spikes source array."
        )

        Population.__init__(self, geometry=nb_neurons, neuron=neuron, name=name)

        self.init['spike_times'] = spike_times

    def _sort_spikes(self, spike_times):
        "Sort, unify the spikes and transform them intosteps."
        return [sorted(list(set([round(t/Global.config['dt']) for t in neur_times]))) for neur_times in spike_times]

    def _generate_omp(self):
        "Code generation"
        # Do not generate default parameters and variables
        self._specific_template['declare_parameters_variables'] = """
    // Custom local parameter spike_times
    std::vector< double > r ;
    std::vector< std::vector< long int > > spike_times ;
    std::vector< long int >  next_spike ;
    std::vector< int > idx_next_spike;
"""
        self._specific_template['declare_additional'] = """
    // Recompute the spike times
    void recompute_spike_times(){
        std::fill(next_spike.begin(), next_spike.end(), -10000);
        std::fill(idx_next_spike.begin(), idx_next_spike.end(), 0);
        for(int i=0; i< size; i++){
            if(!spike_times[i].empty()){
                int idx = 0;
                // Find the first spike time which is not in the past
                while(spike_times[i][idx] < t){
                    idx++;
                }
                // Set the next spike
                if(idx < spike_times[i].size())
                    next_spike[i] = spike_times[i][idx];
                else
                    next_spike[i] = -10000;
            }
        }
    }
"""
        self._specific_template['access_parameters_variables'] = ""

        self._specific_template['init_parameters_variables'] ="""
        r = std::vector<double>(size, 0.0);
        next_spike = std::vector<long int>(size, -10000);
        idx_next_spike = std::vector<int>(size, 0);
        this->recompute_spike_times();
"""

        self._specific_template['reset_additional'] ="""
        this->recompute_spike_times();
"""

        self._specific_template['update_variables'] ="""
        if(_active){
            spiked.clear();
            for(int i = 0; i < %(size)s; i++){
                // Emit spike
                if( t == next_spike[i] ){
                    last_spike[i] = t;
                    /* 
                    while(++idx_next_spike[i]< spike_times[i].size()){
                        if(spike_times[i][idx_next_spike[i]] > t)
                            break;
                    }
                    */
                    idx_next_spike[i]++ ;
                    if(idx_next_spike[i] < spike_times[i].size()){
                        next_spike[i] = spike_times[i][idx_next_spike[i]];
                    }
                    spiked.push_back(i);
                }
            }
        }
""" % {'size': self.size}

        self._specific_template['export_parameters_variables'] ="""
        vector[vector[long]] spike_times
        vector[double] r
        void recompute_spike_times()
"""

        self._specific_template['wrapper_args'] = "size, times"
        self._specific_template['wrapper_init'] = "        pop%(id)s.spike_times = times" % {'id': self.id}

        self._specific_template['wrapper_access_parameters_variables'] = """
    # Local parameter spike_times
    cpdef get_spike_times(self):
        return pop%(id)s.spike_times
    cpdef set_spike_times(self, value):
        pop%(id)s.spike_times = value
        pop%(id)s.recompute_spike_times()
    # Mean firing rate
    cpdef get_r(self):
        return pop%(id)s.r
    cpdef set_r(self, value):
        pop%(id)s.r = value
""" % {'id': self.id}

        
    def _instantiate(self, module):
        # Create the Cython instance 
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.init['spike_times'])

    def __setattr__(self, name, value):
        if name == 'spike_times':
            if not isinstance(value[0], list): # several neurons
                value = [ value ]
            if not len(value) == self.size:
                Global._error('SpikeSourceArray: the size of the spike_times attribute must match the number of neurons in the population.')

            if self.initialized:
                self.cyInstance.set_spike_times(self._sort_spikes(value))
            else:
                self.init['spike_times'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'spike_times':
            if self.initialized:
                return [ [Global.config['dt']*time for time in neur] for neur in self.cyInstance.get_spike_times()]
            else:
                return self.init['spike_times']
        else:
            return Population.__getattribute__(self, name)
            

class HomogeneousCorrelatedSpikeTrains(SpecificPopulation):
    """ 
    Population of spiking neurons following a homogeneous distribution with correlated spike trains.

    The method describing the generation of homogeneous correlated spike trains is described in:

    Brette, R. (2009). Generation of correlated spike trains. <http://audition.ens.fr/brette/papers/Brette2008NC.html>

    The implementation is based on the one provided by `Brian <http://briansimulator.org>`_.

    To generate correlated spike trains, the population rate of the group of Poisson-like spiking neurons varies following a stochastic differential equation:

    .. math::

        dx/dt = (mu - x)/tau + sigma * Xi / sqrt(tau)

    where Xi is a random variable. Basically, x will randomly vary around mu over time, with an amplitude determined by sigma and a speed determined by tau. 

    This doubly stochastic process is called a Cox process or Ornstein-Uhlenbeck process. 

    To avoid that x becomes negative, the values of mu and sigma are computed from a rectified Gaussian distribution, parameterized by the desired population rate **rates**, the desired correlation strength **corr** and the time constant **tau**. See Brette's paper for details.

    In short, you should only define the parameters ``rates``, ``corr` and ``tau``, and let the class compute mu and sigma for you. Changing ``rates``, ``corr` or ``tau`` after initialization automatically recomputes mu and sigma.

    Example:

    .. code-block:: python

        from ANNarchy import *
        setup(dt=0.1)

        pop_poisson = PoissonPopulation(200, rates=10.)
        pop_corr    = HomogeneousCorrelatedSpikeTrains(200, rates=10., corr=0.3, tau=10.)

        compile()

        simulate(1000.)

        pop_poisson.rates=30.
        pop_corr.rates=30.

        simulate(1000.)

    """

    def __init__(self, geometry, rates, corr, tau, name=None, refractory=None):
        """        
        *Parameters*:
        
        * **geometry**: population geometry as tuple. 

        * **rates**: rate in Hz of the population (must be a positive float)

        * **corr**: total correlation strength (float in [0, 1])

        * **tau**: correlation time constant in ms. 

        * **name**: unique name of the population (optional).

        * **refractory**: refractory period in ms (careful: may break the correlation)
        """  
        # Store parameters
        self.rates = float(rates)
        self.corr = corr

        # Correction of mu and sigma
        mu, sigma = self._rectify(self.rates, self.corr, tau)

        # Create the neuron
        corr_neuron = Neuron(
            parameters = """
                tau = %(tau)s : population
                mu = %(mu)s : population
                sigma = %(sigma)s : population
            """ % {'tau': tau, 'mu': mu, 'sigma': sigma},
            equations = """
                x += dt*(mu - x)/tau + sqrt(dt/tau) * sigma * Normal(0., 1.) : population, init=%(mu)s
                p = Uniform(0.0, 1.0) * 1000.0 / dt
            """ % {'mu': mu},
            spike = "p < x",
            refractory=refractory,
            name="HomogeneousCorrelated",
            description="Homogeneous correlated spike trains."
        )

        Population.__init__(self, geometry=geometry, neuron=corr_neuron, name=name)
    
    def __setattr__(self, name, value):
        " Method called when setting an attribute."
        Population.__setattr__(self, name, value) 
        if name in ['rates', 'corr', 'tau'] and hasattr(self, 'initialized'):
            # Correction of mu and sigma everytime r, c or tau is changed
            self.mu, self.sigma = self._rectify(self.rates, self.corr, self.tau)

    def _rectify(self, mu, corr, tau):
        """
        Rectifies mu and sigma to ensure the rates are positive.

        This part of the code is adapted from Brian's source code:

        Copyright ENS, INRIA, CNRS
        Authors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
        Licence: CeCILL
        """
        def _rectified_gaussian(mu, sigma):
            """
            Calculates the mean and standard deviation for a rectified Gaussian distribution.
            mu, sigma: parameters of the original distribution
            Returns mur,sigmar: parameters of the rectified distribution
            """
            a = 1. + erf(mu / (sigma * (2 ** .5)))
            mur = (sigma / (2. * np.pi) ** .5) * np.exp(-0.5 * (mu / sigma) ** 2) + .5 * mu * a
            sigmar = ((mu - mur) * mur + .5 * sigma ** 2 * a) ** .5
            return (mur, sigmar)

        mur = mu
        sigmar = (corr * mu / (2. * tau/1000.)) ** .5
        if sigmar == 0 * sigmar: # for unit consistency
            return (mur, sigmar)
        x0 = mur / sigmar
        ratio = lambda u, v:u / v
        f = lambda x:ratio(*_rectified_gaussian(x, 1.)) - x0
        y = newton(f, x0 * 1.1) # Secant method
        new_sigma = mur / (np.exp(-0.5 * y ** 2) / ((2. * np.pi) ** .5) + .5 * y * (1. + erf(y * (2 ** (-.5)))))
        new_mu = y * new_sigma
        return (new_mu, new_sigma)

    def _generate_omp(self):
        " Nothing special to do here. "
        pass

    def _generate_cuda(self):
        " Nothing special to do here. "
        pass
