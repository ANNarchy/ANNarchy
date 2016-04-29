"""

    SpecificPopulation.py

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
from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron
import ANNarchy.core.Global as Global

import numpy as np
from scipy.special import erf
from scipy.optimize import newton

class PoissonPopulation(Population):
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
            Global._error('In a SpikeSourceArray, spike_times must be a Python list.')
            

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

    def _generate(self):
        "Code generation"
        # Do not generate default parameters and variables
        self._specific_template['declare_parameters_variables'] = """
    // Custom local parameter spike_times
    std::vector< double > r ;
    std::vector< std::vector< double > > spike_times ;
    std::vector< double >  next_spike ;
    std::vector< int > idx_next_spike;
"""
        self._specific_template['access_parameters_variables'] = ""

        self._specific_template['init_parameters_variables'] ="""
        r = std::vector<double>(size, 0.0);
        next_spike = std::vector<double>(size, -10000.0);
        for(int i=0; i< size; i++){
            if(!spike_times[i].empty())
                next_spike[i] = spike_times[i][0];
        }
        idx_next_spike = std::vector<int>(size, 0);
"""

        self._specific_template['reset_additional'] ="""
        next_spike = std::vector<double>(size, -10000.0);
        for(int i=0; i< size; i++){
            if(!spike_times[i].empty())
                next_spike[i] = spike_times[i][0];
        }
        idx_next_spike = std::vector<int>(size, 0);
"""

        self._specific_template['update_variables'] ="""
        if(_active){
            spiked.clear();
            for(int i = 0; i < %(size)s; i++){
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
""" % {'size': self.size}

        self._specific_template['export_parameters_variables'] ="""
        vector[vector[double]] spike_times
        vector[double] r
"""

        self._specific_template['wrapper_args'] = "size, times"
        self._specific_template['wrapper_init'] = "        pop%(id)s.spike_times = times" % {'id': self.id}

        self._specific_template['wrapper_access_parameters_variables'] = """
    # Local parameter spike_times
    cpdef get_spike_times(self):
        return pop%(id)s.spike_times
    cpdef set_spike_times(self, value):
        pop%(id)s.spike_times = value
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
            

class HomogeneousCorrelatedSpikeTrains(Population):
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