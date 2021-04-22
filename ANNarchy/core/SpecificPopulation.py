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
from ANNarchy.core.Neuron import Neuron
import ANNarchy.core.Global as Global

import numpy as np

class SpecificPopulation(Population):
    """
    Interface class for user-defined definition of Population objects. An inheriting
    class need to override the implementor functions _generate_[paradigm], otherwise
    a NotImplementedError exception will be thrown.
    """
    def __init__(self, geometry, neuron, name=None, copied=False):
        """
        Initialization, receive default arguments of Population objects.
        """
        Population.__init__(self, geometry=geometry, neuron=neuron, name=name, stop_condition=None, storage_order='post_to_pre', copied=copied)

    def _generate(self):
        """
        Overridden method of Population, called during the code generation process.
        This function selects dependent on the chosen paradigm the correct implementor
        functions defined by the user.
        """
        if Global.config['paradigm'] == "openmp":
            self._generate_omp()
        elif Global.config['paradigm'] == "cuda":
            self._generate_cuda()
        else:
            raise NotImplementedError

    def _generate_omp(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for single thread and openMP paradigm.
        """
        raise NotImplementedError

    def _generate_cuda(self):
        """
        Intended to be overridden by child class. Implememt code adjustments intended for single thread and openMP paradigm.
        """
        raise NotImplementedError

class PoissonPopulation(SpecificPopulation):
    """
    Population of spiking neurons following a Poisson distribution.

    **Case 1:** Input population

    Each neuron of the population will randomly emit spikes, with a mean firing rate defined by the *rates* argument.

    The mean firing rate in Hz can be a fixed value for all neurons:

    ```python
    pop = PoissonPopulation(geometry=100, rates=100.0)
    ```

    but it can be modified later as a normal parameter:

    ```python
    pop.rates = np.linspace(10, 150, 100)
    ```

    It is also possible to define a temporal equation for the rates, by passing a string to the argument:

    ```python
    pop = PoissonPopulation(
        geometry=100, 
        rates="100.0 * (1.0 + sin(2*pi*t/1000.0) )/2.0"
    )
    ```

    The syntax of this equation follows the same structure as neural variables.

    It is also possible to add parameters to the population which can be used in the equation of `rates`:

    ```python
    pop = PoissonPopulation(
        geometry=100,
        parameters = '''
            amp = 100.0
            frequency = 1.0
        ''',
        rates="amp * (1.0 + sin(2*pi*frequency*t/1000.0) )/2.0"
    )
    ```

    **Note:** The preceding definition is fully equivalent to the definition of this neuron:

    ```python
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
    ```

    The refractory period can also be set, so that a neuron can not emit two spikes too close from each other.

    **Case 2:** Hybrid population

    If the ``rates`` argument is not set, the population can be used as an interface from a rate-coded population.

    The ``target`` argument specifies which incoming projections will be summed to determine the instantaneous firing rate of each neuron.

    See the example in ``examples/hybrid/Hybrid.py`` for a usage.

    """

    def __init__(self, geometry, name=None, rates=None, target=None, parameters=None, refractory=None, copied=False):
        """
        :param geometry: population geometry as tuple.
        :param name: unique name of the population (optional).
        :param rates: mean firing rate of each neuron. It can be a single value (e.g. 10.0) or an equation (as string).
        :param target: the mean firing rate will be the weighted sum of inputs having this target name (e.g. "exc").
        :param parameters: additional parameters which can be used in the *rates* equation.
        :param refractory: refractory period in ms.
        """
        if rates is None and target is None:
            Global._error('A PoissonPopulation must define either rates or target.')

        self.target = target
        self.parameters = parameters
        self.refractory_init = refractory
        self.rates_init = rates

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
        SpecificPopulation.__init__(self, geometry=geometry, neuron=poisson_neuron, name=name, copied=copied)

        if isinstance(rates, np.ndarray):
            self.rates = rates

    def _copy(self):
        "Returns a copy of the population when creating networks."
        return PoissonPopulation(self.geometry, name=self.name, rates=self.rates_init, target=self.target, parameters=self.parameters, refractory=self.refractory_init, copied=True)

    def _generate_omp(self):
        """
        Generate openMP code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """
        pass

    def _generate_cuda(self):
        """
        Generate CUDA code.

        We don't need any separate code snippets. All is done during the
        normal code generation path.
        """
        pass

class TimedArray(SpecificPopulation):
    """
    Data structure holding sequential inputs for a rate-coded network.

    The input values are stored in the (recordable) attribute `r`, without any further processing. 
    You will need to connect this population to another one using the ``connect_one_to_one()`` method.

    By default, the firing rate of this population will iterate over the different values step by step:

    ```python
    inputs = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ]
    )

    inp = TimedArray(rates=inputs)

    pop = Population(10, ...)

    proj = Projection(inp, pop, 'exc')
    proj.connect_one_to_one(1.0)

    compile()

    simulate(10.)
    ```

    This creates a population of 10 neurons whose activity will change during the first 10*dt milliseconds of the simulation. After that delay, the last input will be kept (i.e. 1 for the last neuron).

    If you want the TimedArray to "loop" over the different input vectors, you can specify a period for the inputs:

    ```python
    inp = TimedArray(rates=inputs, period=10.)
    ```

    If the period is smaller than the length of the rates, the last inputs will not be set.

    If you do not want the inputs to be set at every step, but every 10 ms for example, youcan use the ``schedule`` argument:

    ```python
    inp = TimedArray(rates=inputs, schedule=10.)
    ```

    The input [1, 0, 0,...] will stay for 10 ms, then[0, 1, 0, ...] for the next 10 ms, etc...

    If you need a less regular schedule, you can specify it as a list of times:

    ```python
    inp = TimedArray(rates=inputs, schedule=[10., 20., 50., 60., 100., 110.])
    ```

    The first input is set at t = 10 ms (r = 0.0 in the first 10 ms), the second at t = 20 ms, the third at t = 50 ms, etc.

    If you specify less times than in the array of rates, the last ones will be ignored.

    Scheduling can be combined with periodic cycling. Note that you can use the ``reset()`` method to manually reinitialize the TimedArray, times becoming relative to that call:

    ```python
    simulate(100.) # ten inputs are shown with a schedule of 10 ms
    inp.reset()
    simulate(100.) # the same ten inputs are presented again.
    ```

    """
    def __init__(self, rates, schedule=0., period= -1., name=None, copied=False):
        """
        :param rates: array of firing rates. The first axis corresponds to time, the others to the desired dimensions of the population.
        :param schedule: either a single value or a list of time points where inputs should be set. Default: every timestep.
        :param period: time when the timed array will be reset and start again, allowing cycling over the inputs. Default: no cycling (-1.).
        """
        neuron = Neuron(
            parameters="",
            equations=" r = 0.0",
            name="Timed Array",
            description="Timed array source."
        )
        # Geometry of the population
        geometry = rates.shape[1:]

        # Check the schedule
        if isinstance(schedule, (int, float)):
            if float(schedule) <= 0.0:
                schedule = Global.config['dt']
            schedule = [ float(schedule*i) for i in range(rates.shape[0])]

        if len(schedule) > rates.shape[0]:
            Global._error('TimedArray: the length of the schedule parameter cannot exceed the first dimension of the rates parameter.')

        if len(schedule) < rates.shape[0]:
            Global._warning('TimedArray: the length of the schedule parameter is smaller than the first dimension of the rates parameter (more data than time points). Make sure it is what you expect.')

        SpecificPopulation.__init__(self, geometry=geometry, neuron=neuron, name=name, copied=copied)

        self.init['schedule'] = schedule
        self.init['rates'] = rates
        self.init['period'] = period

    def _copy(self):
        "Returns a copy of the population when creating networks."
        return TimedArray(self.init['rates'] , self.init['schedule'], self.init['period'], self.name, copied=True)

    def _generate_omp(self):
        """
        adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a TimedArray
    std::vector< int > _schedule; // List of times where new inputs should be set
    std::vector< std::vector< %(float_prec)s > > _buffer; // buffer holding the data
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': Global.config['precision']}
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedArray
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[0]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': Global.config['precision']}
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters of a TimedArray
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': Global.config['precision']}

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': Global.config['precision']}

        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters of a TimedArray
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_period(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id }

        self._specific_template['update_variables'] = """
        if(_active){
            //std::cout << _t << " " << _block<< " " << _schedule[_block] << std::endl;

            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                r = _buffer[_block];
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
        
        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': Global.config['precision']}

    def _generate_cuda(self):
        """
        adjust code templates for the specific population for single thread and CUDA.
        """
        # HD (18. Nov 2016)
        # I suppress the code generation for allocating the variable r on gpu, as
        # well as memory transfer codes. This is only possible as no other variables
        # allowed in TimedArray.
        self._specific_template['init_parameters_variables'] = ""
        self._specific_template['host_device_transfer'] = ""
        self._specific_template['device_host_transfer'] = ""

        #
        # Code for handling the buffer and schedule parameters
        self._specific_template['declare_additional'] = """
    // Custom local parameter timed array
    std::vector< int > _schedule;
    std::vector< %(float_prec)s* > gpu_buffer;
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': Global.config['precision']}
        self._specific_template['access_additional'] = """
    // Custom local parameter timed array
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) {
        if ( gpu_buffer.empty() ) {
            gpu_buffer = std::vector< %(float_prec)s* >(buffer.size(), nullptr);
            // allocate gpu arrays
            for(int i = 0; i < buffer.size(); i++) {
                cudaMalloc((void**)&gpu_buffer[i], buffer[i].size()*sizeof(%(float_prec)s));
            }
        }

        auto host_it = buffer.begin();
        auto dev_it = gpu_buffer.begin();
        for(host_it, dev_it; host_it < buffer.end(); host_it++, dev_it++) {
            cudaMemcpy( *dev_it, host_it->data(), host_it->size()*sizeof(%(float_prec)s), cudaMemcpyHostToDevice);
        }

        gpu_r = gpu_buffer[0];
    }
    std::vector< std::vector< %(float_prec)s > > get_buffer() {
        std::vector< std::vector< %(float_prec)s > > buffer = std::vector< std::vector< %(float_prec)s > >( gpu_buffer.size(), std::vector<%(float_prec)s>(size,0.0) );

        auto host_it = buffer.begin();
        auto dev_it = gpu_buffer.begin();
        for( host_it, dev_it; host_it < buffer.end(); host_it++, dev_it++ ) {
            cudaMemcpy( host_it->data(), *dev_it, size*sizeof(%(float_prec)s), cudaMemcpyDeviceToHost );
        }

        return buffer;
    }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': Global.config['precision']}
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
        gpu_r = gpu_buffer[0];
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters timed array
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': Global.config['precision']}
        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters timed array
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_periodic(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id, 'float_prec': Global.config['precision'] }

        self._specific_template['update_variables'] = """
        if(_active) {
            // std::cout << _t << " " << _block<< " " << _schedule[_block] << std::endl;
            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                gpu_r = gpu_buffer[_block];
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

        self._specific_template['update_variable_body'] = ""
        self._specific_template['update_variable_header'] = ""
        self._specific_template['update_variable_call'] = """
    // host side update of neurons
    pop%(id)s.update();
""" % {'id': self.id}

        self._specific_template['size_in_bytes'] = "//TODO: "

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)

    def __setattr__(self, name, value):
        if name == 'schedule':
            if self.initialized:
                self.cyInstance.set_schedule( np.array(value) / Global.config['dt'] )
            else:
                self.init['schedule'] = value
        elif name == 'rates':
            if self.initialized:
                if len(value.shape) > 2:
                    # we need to flatten the provided data
                    flat_values = value.reshape( (value.shape[0], self.size) )
                    self.cyInstance.set_rates( flat_values )
                else:
                    self.cyInstance.set_rates( value )
            else:
                self.init['rates'] = value
        elif name == "period":
            if self.initialized:
                self.cyInstance.set_period(int(value /Global.config['dt']))
            else:
                self.init['period'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                return Global.config['dt'] * self.cyInstance.get_schedule()
            else:
                return self.init['schedule']
        elif name == 'rates':
            if self.initialized:
                if len(self.geometry) > 1:
                    # unflatten the data
                    flat_values = self.cyInstance.get_rates()
                    values = np.zeros( tuple( [len(self.schedule)] + list(self.geometry) ) )
                    for x in range(len(self.schedule)):
                        values[x] = np.reshape( flat_values[x], self.geometry)
                    return values
                else:
                    return self.cyInstance.get_rates()
            else:
                return self.init['rates']
        elif name == 'period':
            if self.initialized:
                return self.cyInstance.get_period() * Global.config['dt']
            else:
                return self.init['period']
        else:
            return Population.__getattribute__(self, name)

class SpikeSourceArray(SpecificPopulation):
    """
    Spike source generating spikes at the times given in the spike_times array.

    Depending on the initial array provided, the population will have one or several neurons, but the geometry can only be one-dimensional.

    You can later modify the spike_times attribute of the population, but it must have the same number of neurons as the initial one.

    The spike times are by default relative to the start of a simulation (``ANNarchy.get_time()`` is 0.0).
    If you call the ``reset()`` method of a ``SpikeSourceArray``, this will set the spike times relative to the current time.
    You can then repeat a stimulation many times.

    ```python
    # 2 neurons firing at 100Hz with a 1 ms delay
    times = [
        [ 10, 20, 30, 40],
        [ 11, 21, 31, 41]
    ]
    inp = SpikeSourceArray(spike_times=times)

    compile()

    # Spikes at 10/11, 20/21, etc
    simulate(50)

    # Reset the internal time of the SpikeSourceArray
    inp.reset()

    # Spikes at 60/61, 70/71, etc
    simulate(50)
    ```
    """
    def __init__(self, spike_times, name=None, copied=False):
        """
        :param spike_times: a list of times at which a spike should be emitted if the population should have only 1 neuron, a list of lists otherwise. Times are defined in milliseconds, and will be rounded to the closest multiple of the discretization time step dt.
        :param name: optional name for the population.
        """

        if not isinstance(spike_times, list):
            Global._error('In a SpikeSourceArray, spike_times must be a Python list.')

        if isinstance(spike_times[0], list): # several neurons
            nb_neurons = len(spike_times)
        else: # a single Neuron
            nb_neurons = 1
            spike_times = [ spike_times ]

        # Create a fake neuron just to be sure the description has the correct parameters
        neuron = Neuron(
            parameters="",
            equations="",
            spike=" t == 0",
            reset="",
            name="Spike source",
            description="Spike source array."
        )

        SpecificPopulation.__init__(self, geometry=nb_neurons, neuron=neuron, name=name, copied=copied)

        self.init['spike_times'] = spike_times


    def _copy(self):
        "Returns a copy of the population when creating networks."
        return SpikeSourceArray(self.init['spike_times'], self.name, copied=True)

    def _sort_spikes(self, spike_times):
        "Sort, unify the spikes and transform them into steps."
        return [sorted(list(set([round(t/Global.config['dt']) for t in neur_times]))) for neur_times in spike_times]

    def _generate_omp(self):
        """
        Code generation for the single-thread and openMP paradigm.
        """
        # Add possible targets
        for target in self.targets:
            tpl = {
                'name': 'g_%(target)s' % {'target': target},
                'locality': 'local',
                'eq': '',
                'bounds': {},
                'flags': [],
                'ctype': Global.config['precision'],
                'init': 0.0,
                'transformed_eq': '',
                'pre_loop': {},
                'cpp': '',
                'switch': '',
                'untouched': {},
                'method': 'exponential',
                'dependencies': []
            }
            self.neuron_type.description['variables'].append(tpl)
            self.neuron_type.description['local'].append('g_'+target)

        self._specific_template['declare_additional'] = """
    // Custom local parameter spike_times
    // std::vector< %(float_prec)s > r ;
    std::vector< std::vector< long int > > spike_times ;
    std::vector< long int >  next_spike ;
    std::vector< int > idx_next_spike;
    long int _t;

    // Recompute the spike times
    void recompute_spike_times(){
        std::fill(next_spike.begin(), next_spike.end(), -10000);
        std::fill(idx_next_spike.begin(), idx_next_spike.end(), 0);
        for(int i=0; i< size; i++){
            if(!spike_times[i].empty()){
                int idx = 0;
                // Find the first spike time which is not in the past
                while(spike_times[i][idx] < _t){
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
"""% { 'float_prec': Global.config['precision'] }

        #self._specific_template['access_parameters_variables'] = ""

        self._specific_template['init_additional'] = """
        _t = 0;
        next_spike = std::vector<long int>(size, -10000);
        idx_next_spike = std::vector<int>(size, 0);
        this->recompute_spike_times();
""" 

        self._specific_template['reset_additional'] = """
        _t = 0;
        this->recompute_spike_times();
"""

        self._specific_template['update_variables'] = """
        if(_active){
            spiked.clear();
            for(int i = 0; i < size; i++){
                // Emit spike
                if( _t == next_spike[i] ){
                    last_spike[i] = _t;
                    /*
                    while(++idx_next_spike[i]< spike_times[i].size()){
                        if(spike_times[i][idx_next_spike[i]] > _t)
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
            _t++;
        }
"""
        self._specific_template['test_spike_cond'] = ""

        self._specific_template['export_additional'] ="""
        vector[vector[long]] spike_times
        void recompute_spike_times()
"""

        self._specific_template['wrapper_args'] = "size, times, delay"
        self._specific_template['wrapper_init'] = """
        pop%(id)s.spike_times = times
        pop%(id)s.set_size(size)
        pop%(id)s.set_max_delay(delay)""" % {'id': self.id}

        self._specific_template['wrapper_access_additional'] = """
    # Local parameter spike_times
    cpdef get_spike_times(self):
        return pop%(id)s.spike_times
    cpdef set_spike_times(self, value):
        pop%(id)s.spike_times = value
        pop%(id)s.recompute_spike_times()
""" % {'id': self.id}

    def _generate_cuda(self):
        """
        Code generation for the CUDA paradigm.

        As the spike time generation is not a very compute intensive step but
        requires dynamic data structures, we don't implement it on the CUDA
        devices for now. Consequently, we use the CPU side implementation and
        transfer after computation the results to the GPU.
        """
        self._generate_omp()

        # attach transfer of spiked array to gpu
        # IMPORTANT: the outside transfer is necessary.
        # Otherwise, previous spike counts will be not reseted.
        self._specific_template['update_variables'] += """
        if ( _active ) {
            // Update Spike Count on GPU
            spike_count = spiked.size();
            cudaMemcpy( gpu_spike_count, &spike_count, sizeof(unsigned int), cudaMemcpyHostToDevice);

            // Transfer generated spikes to GPU
            if( spike_count > 0 ) {
                cudaMemcpy( gpu_spiked, spiked.data(), spike_count * sizeof(int), cudaMemcpyHostToDevice);
            }
        }
        """
        self._specific_template['update_variable_body'] = ""
        self._specific_template['update_variable_header'] = ""
        self._specific_template['update_variable_call'] = """
    // host side update of neurons
    pop%(id)s.update();
""" % {'id': self.id}

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.init['spike_times'], self.max_delay)

    def __setattr__(self, name, value):
        if name == 'spike_times':
            if not isinstance(value[0], list): # several neurons
                value = [ value ]
            if not len(value) == self.size:
                Global._error('SpikeSourceArray: the size of the spike_times attribute must match the number of neurons in the population.')

            self.init['spike_times'] = value # when reset is called
            if self.initialized:
                self.cyInstance.set_spike_times(self._sort_spikes(value))
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


class TimedPoissonPopulation(SpecificPopulation):
    """
    Poisson population whose rate vary with the provided schedule.

    Example:

    ```python
    inp = TimedPoissonPopulation(
        geometry = 100,
        rates = [10., 20., 100., 20., 5.],
        schedule = [0., 100., 200., 500., 600.],
    )
    ```


    This creates a population of 100 Poisson neurons whose rate will be:
    
    * 10 Hz during the first 100 ms.
    * 20 HZ during the next 100 ms.
    * 100 Hz during the next 300 ms.
    * 20 Hz during the next 100 ms.
    * 5 Hz until the end of the simulation.
    
    
    If you want the TimedPoissonPopulation to "loop" over the schedule, you can specify a period:

    ```python
    inp = TimedPoissonPopulation(
        geometry = 100,
        rates = [10., 20., 100., 20., 5.],
        schedule = [0., 100., 200., 500., 600.],
        period = 1000.,
    )
    ```

    Here the rate will become 10Hz again every 1 second of simulation. If the period is smaller than the schedule, the remaining rates will not be set.

    Note that you can use the ``reset()`` method to manually reinitialize the schedule, times becoming relative to that call:

    ```python
    simulate(1200.) # Should switch to 100 Hz due to the period of 1000.
    inp.reset()
    simulate(1000.) # Starts at 10 Hz again.
    ```

    The rates were here global to the population. If you want each neuron to have a different rate, ``rates`` must have additional dimensions corresponding to the geometry of the population.

    ```python
    inp = TimedPoissonPopulation(
        geometry = 100,
        rates = [ 
            [10. + 0.05*i for i in range(100)], 
            [20. + 0.05*i for i in range(100)],
        ],
        schedule = [0., 100.],
        period = 1000.,
    )
    ```

    """
    def __init__(self, geometry, rates, schedule, period= -1., name=None, copied=False):
        """    
        :param rates: array of firing rates. The first axis corresponds to the times where the firing rate should change. 
            If a different rate should be used by the different neurons, the other dimensions must match with the geometr of the population.
        :param schedule: list of times (in ms) where the firing rate should change.
        :param period: time when the timed array will be reset and start again, allowing cycling over the schedule. Default: no cycling (-1.).
        """
        
        neuron = Neuron(
            parameters = """
            proba = 1.0
            """,
            equations = """
            p = Uniform(0.0, 1.0) * 1000.0 / dt
            """,
            spike = """
            p < proba
            """,
            name="TimedPoisson",
            description="Spiking neuron following a Poisson distribution."
        )

        SpecificPopulation.__init__(self, geometry=geometry, neuron=neuron, name=name, copied=copied)

        # Check arguments
        try:
            rates = np.array(rates)
        except:
            Global._error("TimedPoissonPopulation: the rates argument must be a numpy array.")

        schedule = np.array(schedule)

        nb_schedules = rates.shape[0]
        if nb_schedules != schedule.size:
            Global._error("TimedPoissonPopulation: the first axis of the rates argument must be the same length as schedule.")


        if rates.ndim == 1 : # One rate for the whole population
            rates = np.array([np.full(self.size, rates[i]) for i in range(nb_schedules)]) 

        # Initial values
        self.init['schedule'] = schedule
        self.init['rates'] = rates
        self.init['period'] = period

    def _copy(self):
        "Returns a copy of the population when creating networks."
        return TimedPoissonPopulation(self.geometry, self.init['rates'] , self.init['schedule'], self.init['period'], self.name, copied=True)

    def _generate_omp(self):
        """
        adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    std::vector< int > _schedule; // List of times where new inputs should be set
    std::vector< std::vector< %(float_prec)s > > _buffer; // buffer holding the data
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': Global.config['precision']}
        self._specific_template['access_additional'] = """
    // Custom local parameters of a TimedPoissonPopulation
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) { _buffer = buffer; r = _buffer[0]; }
    std::vector< std::vector< %(float_prec)s > > get_buffer() { return _buffer; }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': Global.config['precision']}
        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters of a TimedPoissonPopulation
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': Global.config['precision']}

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': Global.config['precision']}

        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters of a TimedArray
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_period(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id }

        self._specific_template['update_variables'] = """
        if(_active){
            //std::cout << _t << " " << _block<< " " << _schedule[_block] << std::endl;

            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                proba = _buffer[_block];
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

        if( _active ) {
            spiked.clear();

            // Updating local variables
            %(float_prec)s step = 1000.0/dt;

            #pragma omp simd
            for(int i = 0; i < size; i++){

                // p = Uniform(0.0, 1.0) * 1000.0 / dt
                p[i] = step*rand_0[i];


            }
        } // active
""" % {'float_prec': Global.config['precision']}
        
        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);

        // buffer
        size_in_bytes += _buffer.capacity() * sizeof(std::vector<%(float_prec)s>);
        for( auto it = _buffer.begin(); it != _buffer.end(); it++ )
            size_in_bytes += it->capacity() * sizeof(%(float_prec)s);
""" % {'float_prec': Global.config['precision']}

    def _generate_cuda(self):
        """
        Code generation if the CUDA paradigm is set.
        """
        # I suppress the code generation for allocating the variable r on gpu, as
        # well as memory transfer codes. This is only possible as no other variables
        # allowed in TimedArray.
        self._specific_template['init_parameters_variables'] = """
        // Random numbers
        cudaMalloc((void**)&gpu_rand_0, size * sizeof(curandState));
        init_curand_states( size, gpu_rand_0, seed );
"""
        self._specific_template['host_device_transfer'] = ""
        self._specific_template['device_host_transfer'] = ""

        #
        # Code for handling the buffer and schedule parameters
        self._specific_template['declare_additional'] = """
    // Custom local parameter timed array
    std::vector< int > _schedule;
    std::vector< %(float_prec)s* > gpu_buffer;
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': Global.config['precision']}
        self._specific_template['access_additional'] = """
    // Custom local parameter timed array
    void set_schedule(std::vector<int> schedule) { _schedule = schedule; }
    std::vector<int> get_schedule() { return _schedule; }
    void set_buffer(std::vector< std::vector< %(float_prec)s > > buffer) {
        if ( gpu_buffer.empty() ) {
            gpu_buffer = std::vector< %(float_prec)s* >(buffer.size(), nullptr);
            // allocate gpu arrays
            for(int i = 0; i < buffer.size(); i++) {
                cudaMalloc((void**)&gpu_buffer[i], buffer[i].size()*sizeof(%(float_prec)s));
            }
        }

        auto host_it = buffer.begin();
        auto dev_it = gpu_buffer.begin();
        for(host_it, dev_it; host_it < buffer.end(); host_it++, dev_it++) {
            cudaMemcpy( *dev_it, host_it->data(), host_it->size()*sizeof(%(float_prec)s), cudaMemcpyHostToDevice);
        }

        gpu_proba = gpu_buffer[0];
    }
    std::vector< std::vector< %(float_prec)s > > get_buffer() {
        std::vector< std::vector< %(float_prec)s > > buffer = std::vector< std::vector< %(float_prec)s > >( gpu_buffer.size(), std::vector<%(float_prec)s>(size,0.0) );

        auto host_it = buffer.begin();
        auto dev_it = gpu_buffer.begin();
        for( host_it, dev_it; host_it < buffer.end(); host_it++, dev_it++ ) {
            cudaMemcpy( host_it->data(), *dev_it, size*sizeof(%(float_prec)s), cudaMemcpyDeviceToHost );
        }

        return buffer;
    }
    void set_period(int period) { _period = period; }
    int get_period() { return _period; }
""" % {'float_prec': Global.config['precision']}
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
        gpu_proba = gpu_buffer[0];
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters timed array
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_buffer(vector[vector[%(float_prec)s]])
        vector[vector[%(float_prec)s]] get_buffer()
        void set_period(int)
        int get_period()
""" % {'float_prec': Global.config['precision']}
        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters timed array
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_rates( self, buffer ):
        pop%(id)s.set_buffer( buffer )
    cpdef np.ndarray get_rates( self ):
        return np.array(pop%(id)s.get_buffer( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_periodic(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id, 'float_prec': Global.config['precision'] }

        self._specific_template['update_variables'] = """
        if(_active) {
            // std::cout << _t << " " << _block<< " " << _schedule[_block] << std::endl;
            // Check if it is time to set the input
            if(_t == _schedule[_block]){
                // Set the data
                gpu_proba = gpu_buffer[_block];
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
__global__ void cuPop%(id)s_local_step( double dt, curandState* rand_0, double* proba, unsigned int* num_events, int* spiked, long int* last_spike )
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    %(float_prec)s step = 1000.0/dt;

    while ( i < %(size)s )
    {
        // p = Uniform(0.0, 1.0) * 1000.0 / dt
        %(float_prec)s p = curand_uniform_double( &rand_0[i] ) * step;

        if (p < proba[i]) {
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
    'float_prec': Global.config['precision']
}

        self._specific_template['update_variable_header'] = "__global__ void cuPop%(id)s_local_step( double dt, curandState* rand_0, double* proba, unsigned int* num_events, int* spiked, long int* last_spike );" % {'id': self.id}
        # Please notice, that the GPU kernels can be launched only with one block. Otherwise, the
        # atomicAdd which is called inside the kernel is not working correct (HD: April 1st, 2021)
        self._specific_template['update_variable_call'] = """
    // host side update of neurons
    pop%(id)s.update();

    // Reset old events
    clear_num_events<<< 1, 1, 0, pop%(id)s.stream >>>(pop%(id)s.gpu_spike_count);
#ifdef _DEBUG
    cudaError_t err_clear_num_events_%(id)s = cudaGetLastError();
    if(err_clear_num_events_%(id)s != cudaSuccess)
        std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_clear_num_events_%(id)s) << std::endl;
#endif

    // Compute current events
    cuPop%(id)s_local_step<<< 1, __pop%(id)s_tpb__, 0, pop%(id)s.stream >>>(
        dt,
        pop%(id)s.gpu_rand_0,
        pop%(id)s.gpu_proba,
        pop%(id)s.gpu_spike_count,
        pop%(id)s.gpu_spiked,
        pop%(id)s.gpu_last_spike
    );
#ifdef _DEBUG
    cudaError_t err_pop_spike_gather_%(id)s = cudaGetLastError();
    if(err_pop_spike_gather_%(id)s != cudaSuccess)
        std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_pop_spike_gather_%(id)s) << std::endl;
#endif

    // transfer back the spike counter (needed by record)
    cudaMemcpyAsync( &pop%(id)s.spike_count, pop%(id)s.gpu_spike_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, pop%(id)s.stream );
#ifdef _DEBUG
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "record_spike_count: " << cudaGetErrorString(err) << std::endl;
#endif

    // transfer back the spiked array (needed by record)
    cudaMemcpyAsync( pop%(id)s.spiked.data(), pop%(id)s.gpu_spiked, pop%(id)s.spike_count*sizeof(int), cudaMemcpyDeviceToHost, pop%(id)s.stream );
#ifdef _DEBUG
    err = cudaGetLastError();
    if ( err != cudaSuccess )
        std::cout << "record_spike: " << cudaGetErrorString(err) << std::endl;
#endif

""" % {'id': self.id}

        self._specific_template['size_in_bytes'] = "//TODO: "

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)

    def __setattr__(self, name, value):
        if name == 'schedule':
            if self.initialized:
                self.cyInstance.set_schedule( np.array(value) / Global.config['dt'] )
            else:
                self.init['schedule'] = value
        elif name == 'rates':
            if self.initialized:
                if len(value.shape) > 2:
                    # we need to flatten the provided data
                    flat_values = value.reshape( (value.shape[0], self.size) )
                    self.cyInstance.set_rates( flat_values )
                else:
                    self.cyInstance.set_rates( value )
            else:
                self.init['rates'] = value
        elif name == "period":
            if self.initialized:
                self.cyInstance.set_period(int(value /Global.config['dt']))
            else:
                self.init['period'] = value
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                return Global.config['dt'] * self.cyInstance.get_schedule()
            else:
                return self.init['schedule']
        elif name == 'rates':
            if self.initialized:
                if len(self.geometry) > 1:
                    # unflatten the data
                    flat_values = self.cyInstance.get_rates()
                    values = np.zeros( tuple( [len(self.schedule)] + list(self.geometry) ) )
                    for x in range(len(self.schedule)):
                        values[x] = np.reshape( flat_values[x], self.geometry)
                    return values
                else:
                    return self.cyInstance.get_rates()
            else:
                return self.init['rates']
        elif name == 'period':
            if self.initialized:
                return self.cyInstance.get_period() * Global.config['dt']
            else:
                return self.init['period']
        else:
            return Population.__getattribute__(self, name)


class HomogeneousCorrelatedSpikeTrains(SpecificPopulation):
    """
    Population of spiking neurons following a homogeneous distribution with correlated spike trains.

    The method describing the generation of homogeneous correlated spike trains is described in:

    > Brette, R. (2009). Generation of correlated spike trains. Neural Computation 21(1). <http://romainbrette.fr/WordPress3/wp-content/uploads/2014/06/Brette2008NC.pdf>

    The implementation is based on the one provided by Brian <http://briansimulator.org>.

    To generate correlated spike trains, the population rate of the group of Poisson-like spiking neurons varies following a stochastic differential equation:

    $$\\frac{dx}{dt} = \\frac{(\\mu - x)}{\\tau} + \\sigma \\, \\frac{\\xi}{\\sqrt{\\tau}}$$

    where $\\xi$ is a random variable. In short, $x$ will randomly vary around mu over time, with an amplitude determined by sigma and a speed determined by tau.

    This doubly stochastic process is called a Cox process or Ornstein-Uhlenbeck process.

    To avoid that x becomes negative, the values of mu and sigma are computed from a rectified Gaussian distribution, parameterized by the desired population rate **rates**, the desired correlation strength **corr** and the time constant **tau**. See Brette's paper for details.

    In short, you should only define the parameters ``rates``, ``corr`` and ``tau``, and let the class compute mu and sigma for you. Changing ``rates``, ``corr`` or ``tau`` after initialization automatically recomputes mu and sigma.

    Example:

    ```python
    from ANNarchy import *
    setup(dt=0.1)

    pop_corr = HomogeneousCorrelatedSpikeTrains(200, rates=10., corr=0.3, tau=10.)

    compile()

    simulate(1000.)

    pop_corr.rates=30.

    simulate(1000.)
    ```

    Alternatively, a schedule can be provided to change automatically the value of `rates` and ``corr``(but not ``tau``) at the required times (as in TimedArray or TimedPoissonPopulation):

    ```python
    from ANNarchy import *
    setup(dt=0.1)

    pop_corr = HomogeneousCorrelatedSpikeTrains(
        geometry=200, 
        rates= [10., 30.], 
        corr=[0.3, 0.5], 
        tau=10.,
        schedule=[0., 1000.]
    )

    compile()

    simulate(2000.)
    ```

    Even when using a schedule, ``corr`` accepts a single constant value. The first value of ``schedule`` must be 0. ``period``specifies when the schedule "loops" back to its initial value. 

    """
    def __init__(self, 
        geometry, 
        rates, 
        corr, 
        tau, 
        schedule=None, 
        period=-1., 
        name=None, 
        refractory=None, 
        copied=False):
        """    
        :param geometry: population geometry as tuple.
        :param rates: rate in Hz of the population (must be a positive float or a list)
        :param corr: total correlation strength (float in [0, 1], or a list)
        :param tau: correlation time constant in ms.
        :param schedule: list of times where new values of ``rates``and ``corr``will be used to computre mu and sigma.
        :param period: time when the array will be reset and start again, allowing cycling over the schedule. Default: no cycling (-1.)
        :param name: unique name of the population (optional).
        :param refractory: refractory period in ms (careful: may break the correlation)
        """
        if schedule is not None:
            self._has_schedule = True
            # Rates
            if not isinstance(rates, (list, np.ndarray)):
                Global._error("TimedHomogeneousCorrelatedSpikeTrains: the rates argument must be a list or a numpy array.")
            rates = np.array(rates)

            # Schedule
            schedule = np.array(schedule)

            nb_schedules = rates.shape[0]
            if nb_schedules != schedule.size:
                Global._error("TimedHomogeneousCorrelatedSpikeTrains: the length of rates must be the same length as for schedule.")

            # corr
            corr = np.array(corr)
            if corr.size == 1:
                corr = np.full(nb_schedules, corr)
        else:
            self._has_schedule = False
            rates = np.array([float(rates)])
            schedule = np.array([0.0])
            corr = np.array([corr])

        
        # Store refractory
        self.refractory_init = refractory

        # Correction of mu and sigma
        mu_list, sigma_list = self._correction(rates, corr, tau)

        self.rates = rates
        self.corr = corr
        self.tau = tau

        # Create the neuron
        corr_neuron = Neuron(
            parameters = """
                tau = %(tau)s : population
                mu = %(mu)s : population
                sigma = %(sigma)s : population
            """ % {'tau': tau, 'mu': mu_list[0], 'sigma': sigma_list[0]},
            equations = """
                x += dt*(mu - x)/tau + sqrt(dt/tau) * sigma * Normal(0., 1.) : population, init=%(mu)s
                p = Uniform(0.0, 1.0) * 1000.0 / dt
            """ % {'mu': mu_list[0]},
            spike = "p < x",
            refractory=refractory,
            name="HomogeneousCorrelated",
            description="Homogeneous correlated spike trains."
        )

        SpecificPopulation.__init__(self, geometry=geometry, neuron=corr_neuron, name=name, copied=copied)

        # Initial values
        self.init['schedule'] = schedule
        self.init['rates'] = rates
        self.init['corr'] = corr
        self.init['tau'] = tau
        self.init['period'] = period


        if self._has_schedule:
            self.init['mu'] = mu_list
            self.init['sigma'] = sigma_list
        else:
            self.init['mu'] = mu_list[0]
            self.init['sigma'] = sigma_list[0]

    def _copy(self):
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
            copied=True)

    def _correction(self, rates, corr, tau):

        # Correction of mu and sigma
        mu_list = []
        sigma_list = []
        for i in range(len(rates)):
            mu, sigma = _rectify(rates[i], corr[i], tau)
            mu_list.append(mu)
            sigma_list.append(sigma)

        return mu_list, sigma_list


    def _generate_omp(self):
        """
        adjust code templates for the specific population for single thread and openMP.
        """
        self._specific_template['declare_additional'] = """
    // Custom local parameters of a HomogeneousCorrelatedSpikeTrains
    std::vector< int > _schedule; // List of times where new inputs should be set
    
    std::vector< %(float_prec)s > _mu; // buffer holding the data
    std::vector< %(float_prec)s > _sigma; // buffer holding the data
    
    int _period; // Period of cycling
    long int _t; // Internal time
    int _block; // Internal block when inputs are set not at each step
""" % {'float_prec': Global.config['precision']}

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

""" % {'float_prec': Global.config['precision']}

        self._specific_template['init_additional'] = """
        // Initialize counters
        _t = 0;
        _block = 0;
        _period = -1;
"""
        self._specific_template['export_additional'] = """
        # Custom local parameters of a HomogeneousCorrelatedSpikeTrains
        void set_schedule(vector[int])
        vector[int] get_schedule()

        void set_mu_list(vector[%(float_prec)s])
        vector[%(float_prec)s] get_mu_list()

        void set_sigma_list(vector[%(float_prec)s])
        vector[%(float_prec)s] get_sigma_list()

        void set_period(int)
        int get_period()
""" % {'float_prec': Global.config['precision']}

        self._specific_template['reset_additional'] ="""
        _t = 0;
        _block = 0;

        r.clear();
        r = std::vector<%(float_prec)s>(size, 0.0);
""" % {'float_prec': Global.config['precision']}

        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters of a HomogeneousCorrelatedSpikeTrains
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))

    cpdef set_mu_list( self, buffer ):
        pop%(id)s.set_mu_list( buffer )
    cpdef np.ndarray get_mu_list( self ):
        return np.array(pop%(id)s.get_mu_list( ))

    cpdef set_sigma_list( self, buffer ):
        pop%(id)s.set_sigma_list( buffer )
    cpdef np.ndarray get_sigma_list( self ):
        return np.array(pop%(id)s.get_sigma_list( ))

    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_period(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id }

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
""" % {'float_prec': Global.config['precision']}

        if self._has_schedule:
            self._specific_template['update_variables'] = scheduling_block + update_block
        else:
            self._specific_template['update_variables'] = update_block
        
        self._specific_template['size_in_bytes'] = """
        // schedule
        size_in_bytes += _schedule.capacity() * sizeof(int);
""" % {'float_prec': Global.config['precision']}

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
""" % {'float_prec': Global.config['precision']}
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
""" % {'float_prec': Global.config['precision'], 'id': self.id}
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
        self._specific_template['export_additional'] = """
        # Custom local parameters timed array
        void set_schedule(vector[int])
        vector[int] get_schedule()
        void set_mu_list(vector[%(float_prec)s])
        vector[%(float_prec)s] get_mu_list()
        void set_sigma_list(vector[%(float_prec)s])
        vector[%(float_prec)s] get_sigma_list()
        void set_period(int)
        int get_period()
""" % {'float_prec': Global.config['precision']}
        self._specific_template['wrapper_access_additional'] = """
    # Custom local parameters timed array
    cpdef set_schedule( self, schedule ):
        pop%(id)s.set_schedule( schedule )
    cpdef np.ndarray get_schedule( self ):
        return np.array(pop%(id)s.get_schedule( ))
    cpdef set_mu_list( self, buffer ):
        pop%(id)s.set_mu_list( buffer )
    cpdef np.ndarray get_mu_list( self ):
        return np.array(pop%(id)s.get_mu_list( ))
    cpdef set_sigma_list( self, buffer ):
        pop%(id)s.set_sigma_list( buffer )
    cpdef np.ndarray get_sigma_list( self ):
        return np.array(pop%(id)s.get_sigma_list( ))
    cpdef set_period( self, period ):
        pop%(id)s.set_period(period)
    cpdef int get_periodic(self):
        return pop%(id)s.get_period()
""" % { 'id': self.id, 'float_prec': Global.config['precision'] }

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
__global__ void cuPop%(id)s_global_step( double dt, double* tau, double mu, double* x, curandState* rand_0, double sigma )
{
    // x += dt*(mu - x)/tau + sqrt(dt/tau) * sigma * Normal(0., 1.)
    x[0] += dt*(mu - x[0])/tau[0] + curand_normal_double( &rand_0[0] )*sigma*sqrt(dt/tau[0]);
}

// Updating local variables of population %(id)s
__global__ void cuPop%(id)s_local_step( double dt, curandState* rand_1, double* x, unsigned int* num_events, int* spiked, long int* last_spike )
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
    'float_prec': Global.config['precision']
}

            self._specific_template['update_variable_header'] = """__global__ void cuPop%(id)s_global_step( double dt, double* tau, double mu, double* x, curandState* rand_0, double sigma );
__global__ void cuPop%(id)s_local_step( double dt, curandState* rand_1, double* x, unsigned int* num_events, int* spiked, long int* last_spike );
""" % {'id': self.id}

            # Please notice, that the GPU kernels can be launched only with one block. Otherwise, the
            # atomicAdd which is called inside the kernel is not working correct (HD: April 1st, 2021)
            self._specific_template['update_variable_call'] = """
    if (pop%(id)s._active) {
        // Update the scheduling
        pop%(id)s.update();

        // Reset old events
        clear_num_events<<< 1, 1, 0, pop%(id)s.stream >>>(pop%(id)s.gpu_spike_count);
    #ifdef _DEBUG
        cudaError_t err_clear_num_events_%(id)s = cudaGetLastError();
        if (err_clear_num_events_%(id)s != cudaSuccess)
            std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_clear_num_events_%(id)s) << std::endl;
    #endif

        // compute the value of x based on mu/sigma
        cuPop%(id)s_global_step<<< 1, 1, 0, pop%(id)s.stream >>>(
            dt,
            pop%(id)s.gpu_tau,
            pop%(id)s.mu,
            pop%(id)s.gpu_x,
            pop%(id)s.gpu_rand_0,
            pop%(id)s.sigma 
        );
        #ifdef _DEBUG
            cudaError_t err_pop%(id)s_global_step = cudaGetLastError();
            if( err_pop%(id)s_global_step != cudaSuccess) {
                std::cout << "pop%(id)s_step: " << cudaGetErrorString(err_pop%(id)s_global_step) << std::endl;
                exit(0);
            }
        #endif

        // Generate new spike events
        cuPop%(id)s_local_step<<< 1, __pop%(id)s_tpb__, 0, pop%(id)s.stream >>>(
            dt,
            pop%(id)s.gpu_rand_1,
            pop%(id)s.gpu_x,
            pop%(id)s.gpu_spike_count,
            pop%(id)s.gpu_spiked,
            pop%(id)s.gpu_last_spike
        );
    #ifdef _DEBUG
        cudaError_t err_pop_spike_gather_%(id)s = cudaGetLastError();
        if(err_pop_spike_gather_%(id)s != cudaSuccess) {
            std::cout << "pop%(id)s_spike_gather: " << cudaGetErrorString(err_pop_spike_gather_%(id)s) << std::endl;
            exit(0);
        }
    #endif

        // transfer back the spike counter (needed by record)
        cudaMemcpy( &pop%(id)s.spike_count, pop%(id)s.gpu_spike_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    #ifdef _DEBUG
        cudaError_t err_pop%(id)s_async_copy = cudaGetLastError();
        if ( err_pop%(id)s_async_copy != cudaSuccess ) {
            std::cout << "record_spike_count: " << cudaGetErrorString(err_pop%(id)s_async_copy) << std::endl;
            exit(0);
        }
    #endif

        // transfer back the spiked array (needed by record)
        cudaMemcpy( pop%(id)s.spiked.data(), pop%(id)s.gpu_spiked, pop%(id)s.spike_count*sizeof(int), cudaMemcpyDeviceToHost);
    #ifdef _DEBUG
        cudaError_t err_pop%(id)s_async_copy2 = cudaGetLastError();
        if ( err_pop%(id)s_async_copy2 != cudaSuccess ) {
            std::cout << "record_spike: " << cudaGetErrorString(err_pop%(id)s_async_copy2) << std::endl;
            exit(0);
        }
    #endif
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
                self.cyInstance.set_schedule( np.array(value) / Global.config['dt'] )
            else:
                self.init['schedule'] = value
        elif name == 'mu':
            if self.initialized:
                if self._has_schedule:
                    self.cyInstance.set_mu_list( value )
                else:
                    self.cyInstance.set_mu( value )
            else:
                self.init['mu'] = value
        elif name == 'sigma':
            if self.initialized:
                if self._has_schedule:
                    self.cyInstance.set_sigma_list( value )
                else:
                    self.cyInstance.set_sigma( value )
            else:
                self.init['sigma'] = value
        elif name == "period":
            if self.initialized:
                self.cyInstance.set_period(int(value /Global.config['dt']))
            else:
                self.init['period'] = value
        elif name == 'rates': 
            if self._has_schedule:
                value = np.array(value)
                if not value.size == self.schedule.size:
                    Global._error("HomogeneousCorrelatedSpikeTrains: rates must have the same length as schedule.")
            else:
                value = np.array([float(value)])
            if self.initialized:
                Population.__setattr__(self, name, value)
                # Correction of mu and sigma everytime r, c or tau is changed
                try:
                    mu, sigma = self._correction(self.rates, self.corr, self.tau)
                    if self._has_schedule:
                        self.mu = mu
                        self.sigma = sigma
                    else:
                        self.mu = mu[0]
                        self.sigma = sigma[0]
                except:
                    pass
            else:
                self.init[name] = value
                Population.__setattr__(self, name, value)
        elif name == 'corr': 
            if self._has_schedule:
                if not isinstance(value, (list, np.ndarray)):
                    value = np.full((self.schedule.size, ), value)
                else:
                    value = np.array(value)
                    if not value.size == self.schedule.size:
                        Global._error("HomogeneousCorrelatedSpikeTrains: corr must have the same length as schedule.")
            else:
                value = np.array([float(value)])
            if self.initialized:
                Population.__setattr__(self, name, value)
                try:
                    # Correction of mu and sigma everytime r, c or tau is changed
                    mu, sigma = self._correction(self.rates, self.corr, self.tau)
                    if self._has_schedule:
                        self.mu = mu
                        self.sigma = sigma
                    else:
                        self.mu = mu[0]
                        self.sigma = sigma[0]
                except:
                    pass
            else:
                self.init[name] = value
                Population.__setattr__(self, name, value)
        elif name == 'tau': 
            if self.initialized:
                Population.__setattr__(self, name, value)
                # Correction of mu and sigma everytime r, c or tau is changed
                mu, sigma = self._correction(self.rates, self.corr, self.tau)
                if self._has_schedule:
                    self.mu = mu
                    self.sigma = sigma
                else:
                    self.mu = mu[0]
                    self.sigma = sigma[0]
            else:
                self.init[name] = value
                Population.__setattr__(self, name, value)
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'schedule':
            if self.initialized:
                if self._has_schedule:
                    return Global.config['dt'] * self.cyInstance.get_schedule()
                else:
                    return np.array([0.0])
            else:
                return self.init['schedule']
        elif name == 'mu':
            if self.initialized:
                if self._has_schedule:
                    return self.cyInstance.get_mu_list()
                else:
                    return self.cyInstance.get_mu()
            else:
                return self.init['mu']
        elif name == 'sigma':
            if self.initialized:
                if self._has_schedule:
                    return self.cyInstance.get_sigma_list()
                else:
                    return self.cyInstance.get_sigma()
            else:
                return self.init['sigma']
        elif name == 'tau':
            if self.initialized:
                return self.cyInstance.get_tau()
            else:
                return self.init['tau']
        elif name == 'period':
            if self.initialized:
                return self.cyInstance.get_period() * Global.config['dt']
            else:
                return self.init['period']
        else:
            return Population.__getattribute__(self, name)

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