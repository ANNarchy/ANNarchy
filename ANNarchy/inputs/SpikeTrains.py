"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.intern.SpecificPopulation import SpecificPopulation
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages
from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron

def _rectify(mu, corr, tau):
    """
    Rectifies mu and sigma to ensure the rates are positive.

    This part of the code is adapted from Brian's source code:

    Copyright ENS, INRIA, CNRS
    Authors: Romain Brette (brette@di.ens.fr) and Dan Goodman (goodman@di.ens.fr)
    Licence: CeCILL
    """

    from scipy.special import erf #pylint: disable=no-name-in-module
    from scipy.optimize import newton

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

class HomogeneousCorrelatedSpikeTrains(SpecificPopulation):
    r"""
    Population of spiking neurons following a homogeneous distribution with correlated spike trains.

    The method describing the generation of homogeneous correlated spike trains is described in:

    > Brette, R. (2009). Generation of correlated spike trains. Neural Computation 21(1). <http://romainbrette.fr/WordPress3/wp-content/uploads/2014/06/Brette2008NC.pdf>

    The implementation is based on the one provided by Brian <http://briansimulator.org>.

    To generate correlated spike trains, the population rate of the group of Poisson-like spiking neurons varies following a stochastic differential equation:

    $$\frac{dx}{dt} = \frac{\mu - x}{\tau} + \sigma \, \frac{\xi}{\sqrt{\tau}}$$

    where $\xi$ is a random sample from the standard normal distribution. In short, $x$ will randomly vary around $\mu$ over time, with an amplitude determined by $\sigma$ and a speed determined by $\tau$.

    This doubly stochastic process is called a Cox process or Ornstein-Uhlenbeck process.

    To avoid that x becomes negative, the values of mu and sigma are computed from a rectified Gaussian distribution, parameterized by the desired population rate **rates**, the desired correlation strength **corr** and the time constant **tau**. See Brette's paper for details.

    In short, you should only define the parameters ``rates``, ``corr`` and ``tau``, and let the class compute mu and sigma for you. Changing ``rates``, ``corr`` or ``tau`` after initialization automatically recomputes mu and sigma.

    Example:

    ```python
    import ANNarchy as ann

    net = ann.Network(dt=0.1)

    pop_corr = net.create(
        ann.HomogeneousCorrelatedSpikeTrains(
            geometry=200, 
            rates=10., 
            corr=0.3, 
            tau=10.)
    )

    net.compile()

    net.simulate(1000.)

    pop_corr.rates=30.

    net.simulate(1000.)
    ```

    Alternatively, a schedule can be provided to change automatically the value of `rates` and `corr` (but not `tau`) at the required times (as in `TimedArray` or TimedPoissonPopulation):

    ```python
    pop_corr = net.create(
        ann.HomogeneousCorrelatedSpikeTrains(
            geometry = 200, 
            rates = [10., 30.], 
            corr = [0.3, 0.5], 
            tau = 10.,
            schedule = [0., 1000.]
        )
    )
    ```

    Even when using a schedule, ``corr`` accepts a single constant value. The first value of ``schedule`` must be 0. ``period`` specifies when the schedule "loops" back to its initial value. 

    :param geometry: population geometry as tuple.
    :param rates: rate in Hz of the population (must be a float or a list of float)
    :param corr: total correlation strength (float in [0, 1], or a list)
    :param tau: correlation time constant in ms.
    :param schedule: list of times where new values of ``rates``and ``corr``will be used to computre mu and sigma.
    :param period: time when the array will be reset and start again, allowing cycling over the schedule. Default: no cycling (-1.)
    :param name: unique name of the population (optional).
    :param refractory: refractory period in ms (careful: may break the correlation)
    """
    def __init__(self, 
            geometry: int | tuple[int], 
            rates:float | list[float], 
            corr:float | list[float], 
            tau:float, 
            schedule:list[float] = None, 
            period:float = -1., 
            name:str = None, 
            refractory:float = None, 
            copied:bool = False,
            net_id:int = 0,
        ):

        # schedule should be a list of onset times. 
        if schedule is None:
            schedule = [0.0]
        else:
            schedule = list(schedule)
        
        # rates should be a list with one value per schedule
        if isinstance(rates, (list, np.ndarray)):
            rates = list(rates)
            if len(rates) != len(schedule): 
                Messages._error("TimedHomogeneousCorrelatedSpikeTrains: the rates argument must be of the same size as schedule when provided as a list.")
        else:
            rates = [float(rates) for _ in range(len(schedule))]

        # corr also
        if isinstance(corr, (list, np.ndarray)):
            if len(corr) != len(schedule): 
                Messages._error("TimedHomogeneousCorrelatedSpikeTrains: the corr argument must be of the same size as schedule when provided as a list.")
            corr = list(corr)
        else:
            corr = [float(corr) for _ in range(len(schedule))]

        # Store the arguments
        self.schedule = schedule
        self.rates = rates
        self.corr = corr
        self.tau = tau
        
        # Store refractory
        self.refractory_init = refractory

        # Correction of mu and sigma
        self.mu_list, self.sigma_list = self._correction(self.rates, self.corr, self.tau)

        # Create the neuron
        corr_neuron = Neuron(
            parameters = """
                tau = %(tau)s : population
                mu = %(mu)s : population
                sigma = %(sigma)s : population
            """ % {'tau': tau, 'mu': self.mu_list[0], 'sigma': self.sigma_list[0]},
            equations = """
                x += dt*(mu - x)/tau + sqrt(dt/tau) * sigma * Normal(0., 1.) : population, init=%(mu)s
                p = Uniform(0.0, 1.0) * 1000.0 / dt
            """ % {'mu': self.mu_list[0]},
            spike = "p < x",
            refractory=refractory,
            name="HomogeneousCorrelated",
            description="Homogeneous correlated spike trains."
        )

        SpecificPopulation.__init__(
            self, 
            geometry=geometry, 
            neuron=corr_neuron, 
            name=name, 
            copied=copied, 
            net_id=net_id
        )

        # Initial values
        self.init['schedule'] = schedule
        self.init['rates'] = rates
        self.init['corr'] = corr
        self.init['tau'] = tau
        self.init['period'] = period
        self.init['mu_list'] = self.mu_list
        self.init['sigma_list'] = self.sigma_list
        self.init['mu'] = self.mu_list[0]
        self.init['sigma'] = self.sigma_list[0]


    def _copy(self, net_id=None):
        "Returns a copy of the population when creating networks."
        return HomogeneousCorrelatedSpikeTrains(
            geometry=self.geometry, 
            rates=self.init['rates'], 
            corr=self.init['corr'], 
            tau=self.init['tau'], 
            schedule=self.init['schedule'], 
            period=self.init['period'], 
            name=self.name, 
            refractory=self.refractory_init, 
            copied=True, 
            net_id=self.net_id if net_id is None else net_id)

    def _correction(self, rates, corr, tau):

        # Correction of mu and sigma
        mu_list = []
        sigma_list = []

        for i in range(len(rates)):
            mu, sigma = _rectify(rates[i], corr[i], tau)
            mu_list.append(mu)
            sigma_list.append(sigma)

        return mu_list, sigma_list

    def _generate_st(self):
        """
        adjust code templates for the specific population for single thread.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a HomogeneousCorrelatedSpikeTrains
    std::vector< int > _schedule; // List of times where new inputs should be set
    
    std::vector< %(float_prec)s > _mu; // buffer holding the data
    std::vector< %(float_prec)s > _sigma; // buffer holding the data
    
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['access_additional'] = """
    // Custom local parameters of a HomogeneousCorrelatedSpikeTrains
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }

    void set_mu_list(std::vector< %(float_prec)s > buffer) { _mu = buffer; mu = _mu[0]; }
    std::vector< %(float_prec)s > get_mu_list() { return _mu; }

    void set_sigma_list(std::vector< %(float_prec)s > buffer) { _sigma = buffer; sigma = _sigma[0]; }
    std::vector< %(float_prec)s > get_sigma_list() { return _sigma; }

    void set_period(int period) { _period = period; }
    int get_period() { return _period; }

""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['update_variables'] = """
        if(_active){
            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                mu = _mu[_block];
                sigma = _sigma[_block];
                // Move to the next block
                _block++;
                // If was the last block, go back to the first block
                if (_block == _schedule.size()){
                    _block = 0;
                }
            }

            // If the timedarray is periodic, check if we arrive at that point
            if(_period > -1 && (_t == _period-1)){
                // Reset the counters
                _block=0;
                _t = -1;
            }

            // Always increment the internal time
            _t++;

            //std::cout << "t: " << _t << " mu: " << mu << " sigma: " << sigma << std::endl;
            
            // Generate spikes
            spiked.clear();

            // x += dt*(mu - x)/tau + sqrt(dt/tau) * sigma * Normal(0., 1.)
            x += dt*(mu - x)/tau + rand_0*sigma*sqrt(dt/tau);

            %(float_prec)s _step = 1000.0/dt;

            #pragma omp simd
            for(int i = 0; i < size; i++){

                // p = Uniform(0.0, 1.0) * 1000.0 / dt
                p[i] = _step*rand_1[i];

            }
        } // active
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        
        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        

        # Nanobind
        self._specific_template['wrapper'] = f"""
    // TimedArray
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)

        // Attributes
		.def_rw("r", &PopStruct{self.id}::r)
		.def_rw("tau", &PopStruct{self.id}::tau)
		.def_rw("mu", &PopStruct{self.id}::mu)
		.def_rw("sigma", &PopStruct{self.id}::sigma)
		.def_rw("x", &PopStruct{self.id}::x)
		.def_rw("p", &PopStruct{self.id}::p)

        // Access methods
        .def("set_schedule", &PopStruct{self.id}::set_schedule)
        .def("get_schedule", &PopStruct{self.id}::get_schedule)

        .def("set_mu_list", &PopStruct{self.id}::set_mu_list)
        .def("get_mu_list", &PopStruct{self.id}::get_mu_list)

        .def("set_sigma_list", &PopStruct{self.id}::set_sigma_list)
        .def("get_sigma_list", &PopStruct{self.id}::get_sigma_list)

        .def("set_period", &PopStruct{self.id}::set_period)
        .def("get_period", &PopStruct{self.id}::get_period)

        // Other methods
		.def("compute_firing_rate", &PopStruct{self.id}::compute_firing_rate)
        .def("activate", &PopStruct{self.id}::set_active)
        .def("reset", &PopStruct{self.id}::reset)
        .def("clear", &PopStruct{self.id}::clear);
        """

    def _generate_omp(self):
        """
        adjust code templates for the specific population for openMP.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a HomogeneousCorrelatedSpikeTrains
    std::vector< int > _schedule; // List of times where new inputs should be set
    
    std::vector< %(float_prec)s > _mu; // buffer holding the data
    std::vector< %(float_prec)s > _sigma; // buffer holding the data
    
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['access_additional'] = """
    // Custom local parameters of a HomogeneousCorrelatedSpikeTrains
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }

    void set_mu_list(std::vector< %(float_prec)s > buffer) { _mu = buffer; mu = _mu[0]; }
    std::vector< %(float_prec)s > get_mu_list() { return _mu; }

    void set_sigma_list(std::vector< %(float_prec)s > buffer) { _sigma = buffer; sigma = _sigma[0]; }
    std::vector< %(float_prec)s > get_sigma_list() { return _sigma; }

    void set_period(int period) { _period = period; }
    int get_period() { return _period; }

""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}


        scheduling_block = """
        if(_active){
            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                mu = _mu[_block];
                sigma = _sigma[_block];
                // Move to the next block
                _block++;
                // If was the last block, go back to the first block
                if (_block == _schedule.size()){
                    _block = 0;
                }
            }

            // If the timedarray is periodic, check if we arrive at that point
            if(_period > -1 && (_t == _period-1)){
                // Reset the counters
                _block=0;
                _t = -1;
            }

            // Always increment the internal time
            _t++;
        }
        """

        update_block = """
        if( _active ) {
            #pragma omp single
            {
                spiked.clear();

                // x += dt*(mu - x)/tau + sqrt(dt/tau) * sigma * Normal(0., 1.)
                x += dt*(mu - x)/tau + rand_0*sigma*sqrt(dt/tau);

                %(float_prec)s _step = 1000.0/dt;

                #pragma omp simd
                for(int i = 0; i < size; i++){

                    // p = Uniform(0.0, 1.0) * 1000.0 / dt
                    p[i] = _step*rand_1[i];

                }
            }
        } // active
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

        if self._has_schedule:
            self._specific_template['update_variables'] = scheduling_block + update_block
        else:
            self._specific_template['update_variables'] = update_block
        
        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}

    def _generate_cuda(self):
        """
        Code generation if the CUDA paradigm is set.
        """
        #
        # Code for handling the buffer and schedule parameters
        self._specific_template['declare_additional'] = """
    // Custom local parameter HomogeneousCorrelatedSpikeTrains
    std::vector< int > _schedule;

    std::vector<%(float_prec)s> mu_buffer;      // buffer
    std::vector<%(float_prec)s> sigma_buffer;   // buffer

    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': ConfigManager().get('precision', self.net_id)}
        
        self._specific_template['access_additional'] = """
    // Custom local parameter HomogeneousCorrelatedSpikeTrains
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }

    void set_mu_list(std::vector< %(float_prec)s > buffer) { mu_buffer = buffer; }
    void set_sigma_list(std::vector< %(float_prec)s > buffer) { sigma_buffer = buffer; }
    std::vector< %(float_prec)s > get_mu_list() { return mu_buffer; }
    std::vector< %(float_prec)s > get_sigma_list() { return sigma_buffer; }

    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': ConfigManager().get('precision', self.net_id), 'id': self.id}
        
        self._specific_template['init_additional'] = """
        // counters
        _t = 0;
        _block = 0;
        _period = -1;
"""

        self._specific_template['reset_additional'] = """
        // counters
        _t = 0;
        _block = 0;
"""

        if not self._has_schedule:
            # we can use the normal code generation for GPU kernels
            pass

        else:
            self._specific_template['update_variables'] = """
        if(_active) {
            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                mu = mu_buffer[_block];
                sigma = sigma_buffer[_block];
                // Move to the next block
                _block++;
                // If was the last block, go back to the first block
                if ( _block == _schedule.size() ) {
                    _block = 0;
                }
            }

            // If the timedarray is periodic, check if we arrive at that point
            if( (_period > -1) && (_t == _period-1) ) {
                // Reset the counters
                _block=0;
                _t = -1;
            }

            // Always increment the internal time
            _t++;
        }
"""

            self._specific_template['update_variable_body'] = """
// Updating global variables of population %(id)s
__global__ void cuPop%(id)s_global_step( const long int t, const double dt, const double tau, double mu, double* x, curandState* rand_0, double sigma )
{
    // x += dt*(mu - x)/tau + sqrt(dt/tau) * sigma * Normal(0., 1.)
    x[0] += dt*(mu - x[0])/tau + curand_normal_double( &rand_0[0] )*sigma*sqrt(dt/tau);
}

// Updating local variables of population %(id)s
__global__ void cuPop%(id)s_local_step( const long int t, const double dt, curandState* rand_1, double* x, unsigned int* num_events, int* spiked, long int* last_spike )
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    %(float_prec)s step = 1000.0/dt;

    while ( i < %(size)s )
    {
        // p = Uniform(0.0, 1.0) * 1000.0 / dt
        %(float_prec)s p = curand_uniform_double( &rand_1[i] ) * step;

        if (p < x[0]) {
            int pos = atomicAdd ( num_events, 1);
            spiked[pos] = i;
            last_spike[i] = t;
        }

        i += blockDim.x;
    }

    __syncthreads();
}
""" % {
    'id': self.id,
    'size': self.size,
    'float_prec': ConfigManager().get('precision', self.net_id)
}

            self._specific_template['update_variable_header'] = """__global__ void cuPop%(id)s_global_step( const long int t, const double dt, const double tau, double mu, double* x, curandState* rand_0, double sigma );
__global__ void cuPop%(id)s_local_step( const long int t, const double dt, curandState* rand_1, double* x, unsigned int* num_events, int* spiked, long int* last_spike );
""" % {'id': self.id}

            # Please notice, that the GPU kernels can be launched only with one block. Otherwise, the
            # atomicAdd which is called inside the kernel is not working correct (HD: April 1st, 2021)
            self._specific_template['update_variable_call'] = """
    if (pop%(id)s->_active) {
        // Update the scheduling
        pop%(id)s->update();

        // Reset old events
        clear_num_events<<< 1, 1, 0, pop%(id)s->stream >>>(pop%(id)s->gpu_spike_count);
    #ifdef _DEBUG
        cudaError_t err_clear_num_events_%(id)s = cudaGetLastError();
        if (err_clear_num_events_%(id)s != cudaSuccess)
            std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_clear_num_events_%(id)s) << std::endl;
    #endif

        // compute the value of x based on mu/sigma
        cuPop%(id)s_global_step<<< 1, 1, 0, pop%(id)s->stream >>>(
            t, dt,
            pop%(id)s->tau,
            pop%(id)s->mu,
            pop%(id)s->gpu_x,
            pop%(id)s->gpu_rand_0,
            pop%(id)s->sigma 
        );
        #ifdef _DEBUG
            cudaError_t err_pop%(id)s_global_step = cudaGetLastError();
            if( err_pop%(id)s_global_step != cudaSuccess) {
                std::cout << "pop%(id)s_step: " << cudaGetErrorString(err_pop%(id)s_global_step) << std::endl;
                exit(0);
            }
        #endif

        // Generate new spike events
        cuPop%(id)s_local_step<<< 1, pop%(id)s->_threads_per_block, 0, pop%(id)s->stream >>>(
            t, dt,
            pop%(id)s->gpu_rand_1,
            pop%(id)s->gpu_x,
            pop%(id)s->gpu_spike_count,
            pop%(id)s->gpu_spiked,
            pop%(id)s->gpu_last_spike
        );
    #ifdef _DEBUG
        cudaError_t err_pop_spike_gather_%(id)s = cudaGetLastError();
        if(err_pop_spike_gather_%(id)s != cudaSuccess) {
            std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_pop_spike_gather_%(id)s) << std::endl;
            exit(0);
        }
    #endif

        // transfer back the spike counter (needed by record)
        cudaMemcpy( &pop%(id)s->spike_count, pop%(id)s->gpu_spike_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    #ifdef _DEBUG
        cudaError_t err_pop%(id)s_async_copy = cudaGetLastError();
        if ( err_pop%(id)s_async_copy != cudaSuccess ) {
            std::cout << "record_spike_count: " << cudaGetErrorString(err_pop%(id)s_async_copy) << std::endl;
            exit(0);
        }
    #endif

        // transfer back the spiked array (needed by record)
        if (pop%(id)s->spike_count > 0) {
            cudaMemcpy( pop%(id)s->spiked.data(), pop%(id)s->gpu_spiked, pop%(id)s->spike_count*sizeof(int), cudaMemcpyDeviceToHost);
        #ifdef _DEBUG
            cudaError_t err_pop%(id)s_async_copy2 = cudaGetLastError();
            if ( err_pop%(id)s_async_copy2 != cudaSuccess ) {
                std::cout << "record_spike: " << cudaGetErrorString(err_pop%(id)s_async_copy2) << std::endl;
                exit(0);
            }
        #endif
        }
    }
""" % {'id': self.id}

        self._specific_template['size_in_bytes'] = "//TODO: "

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)

    def __setattr__(self, name, value):

        if not hasattr(self, 'initialized'):
            Population.__setattr__(self, name, value)

        elif name == 'schedule':
            if self.initialized:
                # Cast to steps (integer)
                schedule_list = [int(val / ConfigManager().get('dt', self.net_id)) for val in value]
                self.cyInstance.set_schedule( schedule_list )
            else:
                self.init['schedule'] = value

        elif name == 'mu':
            if self.initialized:
                self.cyInstance.mu = value
            else:
                self.init['mu'] = value

        elif name == 'mu_list':
            if self.initialized:
                self.cyInstance.set_mu_list( value )
            else:
                self.init['mu_list'] = value

        elif name == 'sigma':
            if self.initialized:
                self.cyInstance.sigma = value
            else:
                self.init['sigma'] = value

        elif name == 'sigma_list':
            if self.initialized:
                self.cyInstance.set_sigma_list( value )
            else:
                self.init['sigma_list'] = value

        elif name == "period":
            if self.initialized:
                self.cyInstance.set_period(int(value /ConfigManager().get('dt', self.net_id)))
            else:
                self.init['period'] = value

        elif name == 'rates': 
            # Create a list of rates, even if only one value is provided
            if isinstance(value, (np.ndarray, list)):
                value = list(value)
            else:
                value = [float(value)]
            if not len(value) == len(self.schedule):
                Messages._error("HomogeneousCorrelatedSpikeTrains: rates must have the same length as schedule.")

            if self.initialized:
                # Set the attribute
                Population.__setattr__(self, name, value)
                
                # Correction of mu and sigma everytime rates, corr or tau is changed
                try:
                    mu_list, sigma_list = self._correction(value, self.corr, self.tau)
                    self.cyInstance.set_mu_list( mu_list )
                    self.cyInstance.set_sigma_list( sigma_list )

                except Exception as e:
                    print(e)
            else:
                self.init[name] = value

        elif name == 'corr': 
            # Create a list of corrs, even if only one value is provided
            if isinstance(value, (np.ndarray, list)):
                value = list(value)
            else:
                value = [float(value)]
            if not len(value) == len(self.schedule):
                Messages._error("HomogeneousCorrelatedSpikeTrains: corr must have the same length as schedule.")

            if self.initialized:
                # Set the attribute
                Population.__setattr__(self, name, value)
                
                # Correction of mu and sigma everytime rates, corr or tau is changed
                try:
                    mu_list, sigma_list = self._correction(self.rates, value, self.tau)
                    self.cyInstance.set_mu_list( mu_list )
                    self.cyInstance.set_sigma_list( sigma_list )
                except Exception as e:
                    print(e)
            else:
                self.init[name] = value

        elif name == 'tau': 
            if self.initialized:
                # Set the attribute
                Population.__setattr__(self, name, value)
                
                # Correction of mu and sigma everytime rates, corr or tau is changed
                try:
                    mu_list, sigma_list = self._correction(self.rates, self.corr, value)
                    self.cyInstance.set_mu_list( mu_list )
                    self.cyInstance.set_sigma_list( sigma_list )
                except Exception as e:
                    print(e)
            else:
                self.init[name] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):

        if name == 'schedule':
            if self.initialized:
                return [ConfigManager().get('dt', self.net_id) * val for val in self.cyInstance.get_schedule() ]
            else:
                return self.init['schedule']
              
        elif name == 'mu':
            if self.initialized:
                return self.cyInstance.mu
            else:
                return self.init['mu']
            
        elif name == 'mu_list':
            if self.initialized:
                return list(self.cyInstance.get_mu_list())
            else:
                return self.init['mu_list']
            
        elif name == 'sigma':
            if self.initialized:
                return self.cyInstance.sigma
            else:
                return self.init['sigma']
            
        elif name == 'sigma_list':
            if self.initialized:
                return list(self.cyInstance.get_sigma_list())
            else:
                return self.init['sigma_list']
                      
        elif name == 'period':
            if self.initialized:
                return self.cyInstance.get_period() * ConfigManager().get('dt', self.net_id)
            else:
                return self.init['period']
            
        else:
            return Population.__getattribute__(self, name)
