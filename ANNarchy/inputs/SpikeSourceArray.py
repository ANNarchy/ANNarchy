"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

from ANNarchy.intern.SpecificPopulation import SpecificPopulation
from ANNarchy.intern.ConfigManagement import ConfigManager
from ANNarchy.intern import Messages

from ANNarchy.core.Population import Population
from ANNarchy.core.Neuron import Neuron

class SpikeSourceArray(SpecificPopulation):
    """
    Spike source generating spikes at the times given in the `spike_times` array.

    Depending on the initial array provided, the population will have one or several neurons, but the geometry can only be one-dimensional.

    You can later modify the spike_times attribute of the population, but it must have the same number of neurons as the initial one.

    The spike times are by default relative to the start of a simulation.
    If you call the ``reset()`` method of a ``SpikeSourceArray``, this will set the spike times relative to the current time.
    You can then repeat a stimulation many times.

    ```python
    # 2 neurons firing at 100Hz with a 1 ms delay
    times = [
        [ 10, 20, 30, 40],
        [ 11, 21, 31, 41]
    ]
    inp = net.create(ann.SpikeSourceArray(spike_times=times))

    net.compile()

    # Spikes at 10/11, 20/21, etc
    net.simulate(50)

    # Reset the internal time of the SpikeSourceArray
    inp.reset()

    # Spikes at 60/61, 70/71, etc
    net.simulate(50)
    ```

    :param spike_times: a list of times at which a spike should be emitted if the population should have only 1 neuron, a list of lists otherwise. Times are defined in milliseconds, and will be rounded to the closest multiple of the discretization time step dt.
    :param name: optional name for the population.
    """
    def __init__(self, spike_times:list[float], name:str=None, copied=False, net_id=0):

        if not isinstance(spike_times, list):
            Messages._error('In a SpikeSourceArray, spike_times must be a Python list.')

        if isinstance(spike_times[0], list): # several neurons
            nb_neurons = len(spike_times)
        else: # a single Neuron
            nb_neurons = 1
            spike_times = [ spike_times ]

        # Create a fake neuron just to be sure the description has the correct parameters
        neuron = Neuron(
            parameters="",
            equations="",
            spike="t==0",
            reset="",
            name="Spike source",
            description="Spike source array."
        )

        SpecificPopulation.__init__(self, geometry=nb_neurons, neuron=neuron, name=name, copied=copied, net_id=net_id)

        self.init['spike_times'] = spike_times


    def _copy(self, net_id=None):
        "Returns a copy of the population when creating networks."
        return SpikeSourceArray(
            self.init['spike_times'], self.name, 
            copied=True, net_id=self.net_id if net_id is None else net_id)

    def _sort_spikes(self, spike_times):
        "Sort, unify the spikes and transform them into steps."
        return [
            sorted(
                list(
                    set(
                        [round(t/ConfigManager().get('dt', self.net_id)) for t in neur_times]
                    )
                )
            ) for neur_times in spike_times
        ]

    def _generate_st(self):
        """
        Code generation for single-thread.
        """
        self._generate_omp()

    def _generate_omp(self):
        """
        Code generation for openMP paradigm.
        """
        # Add possible targets
        for target in self.targets:
            tpl = {
                'name': 'g_%(target)s' % {'target': target},
                'locality': 'local',
                'eq': '',
                'bounds': {},
                'flags': [],
                'ctype': ConfigManager().get('precision', self.net_id),
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
"""% { 'float_prec': ConfigManager().get('precision', self.net_id) }

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

        if ConfigManager().get('num_threads', self.net_id) == 1:
            self._specific_template['update_variables'] = """
        if(_active){
            spiked.clear();
            for(int i = 0; i < size; i++){
                // Emit spike
                if( _t == next_spike[i] ){
                    last_spike[i] = _t;
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
        else:
            self._specific_template['update_variables'] = """
        if(_active){
            #pragma omp single
            {
                spiked.clear();
            }

            #pragma omp for
            for(int i = 0; i < size; i++){
                // Emit spike
                if( _t == next_spike[i] ){
                    last_spike[i] = _t;
                    idx_next_spike[i]++ ;
                    if(idx_next_spike[i] < spike_times[i].size()){
                        next_spike[i] = spike_times[i][idx_next_spike[i]];
                    }

                    #pragma omp critical
                    spiked.push_back(i);
                }
            }

            #pragma omp single
            {
                _t++;
            }
        }
"""
        self._specific_template['test_spike_cond'] = ""

        self._specific_template['wrapper'] = f"""
    // SpikeSourceArray pop{self.id}
    nanobind::class_<PopStruct{self.id}>(m, "pop{self.id}_wrapper")
        // Constructor
        .def(nanobind::init<int, int>())

        // Common attributes
        .def_rw("size", &PopStruct{self.id}::size)
        .def_rw("spike_times", &PopStruct{self.id}::spike_times)
        .def_rw("max_delay", &PopStruct{self.id}::max_delay)
        .def_rw("r", &PopStruct{self.id}::r)

        // Other methods
		.def("compute_firing_rate", &PopStruct{self.id}::compute_firing_rate)

        .def("activate", &PopStruct{self.id}::set_active)
        .def("reset", &PopStruct{self.id}::reset)
        .def("clear", &PopStruct{self.id}::clear);
        """

    def _generate_cuda(self):
        """
        Code generation for the CUDA paradigm.

        As the spike time generation is not a very compute intensive step but
        requires dynamic data structures, we don't implement it on the CUDA
        devices for now. Consequently, we use the CPU side implementation and
        transfer after computation the results to the GPU.
        """
        self._generate_st()

        # Maybe not the most elegant solution, but it works (HD: 14th Feb. 2025)
        part1 = self._specific_template['wrapper'].split("// Other methods")[0]
        part2 = self._specific_template['wrapper'].split("// Other methods")[1]
        self._specific_template['wrapper'] = part1
        self._specific_template['wrapper'] += f"""
        // Trigger host-to-device transfer
        .def_rw("r_host_to_device", &PopStruct{self.id}::r_host_to_device)

        // Other methods"""
        self._specific_template['wrapper'] += part2

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

        # overwrite default code generation for neural update
        self._specific_template['update_variable_body'] = ""
        self._specific_template['update_variable_invoke'] = ""
        self._specific_template['update_variable_header'] = ""
        self._specific_template['update_variable_call'] = """
    // host side update of neurons
    pop%(id)s->update();
""" % {'id': self.id}

        # overwrite default code generation for spike gather
        self._specific_template['spike_gather_body'] = ""
        self._specific_template['spike_gather_invoke'] = ""
        self._specific_template['spike_gather_header'] = ""
        self._specific_template['spike_gather_call'] = ""

    def _instantiate(self, module):
        # Create the Cython instance
        self.cyInstance = getattr(module, self.class_name+'_wrapper')(self.size, self.max_delay)
        self.cyInstance.spike_times = self._sort_spikes(self.init['spike_times'])

    def __setattr__(self, name, value):
        if name == 'spike_times':

            if not isinstance(value[0], list): # several neurons
                value = [ value ]
            
            if not len(value) == self.size:
                Messages._error('SpikeSourceArray: the size of the spike_times attribute must match the number of neurons in the population. Pad with `[]` if necessary.')

            self.init['spike_times'] = value # when reset is called
            if self.initialized:
                self.cyInstance.spike_times = self._sort_spikes(value)
        else:
            Population.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == 'spike_times':
            if self.initialized:
                return [ [ConfigManager().get('dt', self.net_id)*time for time in neur] for neur in self.cyInstance.spike_times]
            else:
                return self.init['spike_times']
        else:
            return Population.__getattribute__(self, name)