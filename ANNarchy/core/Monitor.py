#===============================================================================
#
#     Monitor.py
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
from . import Global
from .Population import Population
from .PopulationView import PopulationView
from .Projection import Projection
from .Dendrite import Dendrite

import numpy as np
import re
import sys

class Monitor(object):
    """
    Monitoring class allowing to record easily parameters or variables from Population, PopulationView and Dendrite objects.
    """

    def __init__(self, obj, variables=[], period=None, period_offset=None, start=True, net_id=0):
        """
        *Parameters*:

        * **obj**: object to monitor. Must be a Population, PopulationView, Dendrite or Projection object.

        * **variables**: single variable name or list of variable names to record (default: []).

        * **period**: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).

        * **period_offset**: determine the moment in ms of recording within the period (default 0). Must be smaller than **period**.

        * **start**: defines if the recording should start immediately (default: True). If not, you should later start the recordings with the ``start()`` method.

        Example::

            m = Monitor(pop, ['g_exc', 'v', 'spike'], period=10.0)

        It is also possible to record the sum of inputs to each neuron in a rate-coded population::

            m = Monitor(pop, ['sum(exc)', 'r'])

        """
        # Object to record (Population, PopulationView, Dendrite)
        self.object = obj
        self.cyInstance = None
        self.net_id = net_id
        self.name = 'Monitor'

        # Check type of the object
        if not isinstance(self.object, (Population, PopulationView, Dendrite, Projection)):
            Global._error('Monitor: the object must be a Population, PopulationView, Dendrite or Projection object')

        # Variables to record
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        # Check variables
        for var in self.variables:
            if not var in self.object.attributes and not var in ['spike'] and not var.startswith('sum('):
                Global._error('Monitor: the object does not have an attribute named', var)

        # Period
        if not period:
            self._period = Global.config['dt']
        else:
            self._period = float(period)

        # Period Offset
        if not period_offset:
            self._period_offset = 0
        else:
            # check validity
            if period_offset >= period:
                Global._error("Monitor(): value of period_offset must be smaller than period.")
            else:
                self._period_offset = period_offset

        # Warn users when recording projections
        if isinstance(self.object, Projection) and self._period == Global.config['dt']:
            Global._warning('Monitor(): it is a bad idea to record synaptic variables of a projection at each time step!')

        # Start
        self._start = start
        self._recorded_variables = {}

        # Add the monitor to the global variable
        self.id = len(Global._network[self.net_id]['monitors'])

        Global._network[self.net_id]['monitors'].append(self)

        if Global._network[self.net_id]['compiled']: # Already compiled
            self._init_monitoring()

    # Extend the period attribute
    @property
    def period(self):
        "Period of recording in ms"
        if not self.cyInstance:
            return self._period
        else:
            return self.cyInstance.period * Global.config['dt']
    @period.setter
    def period(self, val):
        if not self.cyInstance:
            self._period = val
        else:
            self.cyInstance.period = int(val/Global.config['dt'])

    # Extend the period_offset attribute
    @property
    def period_offset(self):
        "Shift of moment of time of recording in ms within a period"
        if not self.cyInstance:
            return self._period
        else:
            return self.cyInstance.period_offset * Global.config['dt']

    @period.setter
    def period_offset(self, val):
        if not self.cyInstance:
            self._period = val
        else:
            self.cyInstance.period_offset = int(val/Global.config['dt'])

    def size_in_bytes(self):
        """
        Get the size of allocated memory on C++ side. Please note, this is only valid if compile() was invoked.

        :return: size in bytes of all allocated C++ data.
        """
        if hasattr(self.cyInstance, 'size_in_bytes'):
            return self.cyInstance.size_in_bytes()

    def _clear(self):
        """
        Deallocates the container within the C++ instance. The population object is not usable anymore after calling this function.

        Warning: should be only called by the net deconstructor (in the context of parallel_run).
        """
        if hasattr(self.cyInstance, 'clear'):
            self.cyInstance.clear()


    def _add_variable(self, var):
        if not var in self.variables:
            self.variables.append(var)
        self._recorded_variables[var] = {'start': [Global.get_current_step(self.net_id)], 'stop': [Global.get_current_step(self.net_id)]}

    def _init_monitoring(self):
        "To be called after compile() as it accesses cython objects"
        # Start recording dependent on the recorded object
        if isinstance(self, BoldMonitor):
            self._start_bold_monitor()
        elif isinstance(self.object, (Population, PopulationView)):
            self._start_population()
        elif isinstance(self.object, (Dendrite, Projection)):
            self._start_dendrite()

    def _start_population(self):
        "Creates the C++ object and starts the recording for a population."

        if isinstance(self.object, PopulationView):
            self.ranks = self.object.ranks
        else:
            self.ranks = [-1]

        # Create the wrapper
        period = int(self._period/Global.config['dt'])
        period_offset = int(self._period_offset/Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'PopRecorder'+str(self.object.id)+'_wrapper')(self.ranks, period, period_offset, offset)
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

        for var in self.variables:
            self._add_variable(var)

        # Start recordings if enabled
        if self._start:
            self.start()

    def _start_dendrite(self):
        "Creates the C++ object and starts the recording for a dendrite."

        if isinstance(self.object, Dendrite):
            self.ranks = self.object.post_rank
            self.idx = [self.object.idx]
            proj_id = self.object.proj.id
        else: # Projection
            self.ranks = [-1]
            self.idx = self.object.post_ranks
            proj_id = self.object.id

        # Compute the period and offset
        period = int(self._period/Global.config['dt'])
        period_offset = int(self._period_offset / Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period

        # Create the wrapper
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'ProjRecorder'+str(proj_id)+'_wrapper')(self.idx, period, period_offset, offset)

        # Add the monitor to the network
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

        # Add the variables
        for var in self.variables:
            self._add_variable(var)

        # Start recordings if enabled
        if self._start:
            self.start()

    def start(self, variables=None, period=None):
        """Starts recording the variables. It is called automatically after ``compile()`` if the flag ``start`` was not passed to the constructor.

        *Parameters*:

        * **variables**: single variable name or list of variable names to start recording (default: the ``variables`` argument passed to the constructor).

        * **period**: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).
        """
        if variables:
            if not isinstance(variables, list):
                self._add_variable(variables)
                variables = [variables]
            else:
                for var in variables:
                    self._add_variable(var)
        else:
            variables = self.variables

        if period:
            self._period = period
            self.cyInstance.period = int(self._period/Global.config['dt'])
            self.cyInstance.offset = Global.get_current_step(self.net_id)

        for var in variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, True)
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population ' + self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                    if var in self.object.proj.parameters:
                        Global._print('\t', var, 'is a parameter, its value is constant')
                Global._warning('Monitor: ' + var + ' can not be recorded ('+obj_desc+')')

    def resume(self):
        "Resumes the recordings."
        # Start recording the variables
        for var in self.variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, True)
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population '+self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                Global._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')
            self._recorded_variables[var]['start'].append(Global.get_current_step(self.net_id))

    def pause(self):
        "Resumes the recordings."
        # Start recording the variables
        for var in self.variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, False)
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population '+self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                Global._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')
            self._recorded_variables[var]['stop'].append(Global.get_current_step(self.net_id))

    def stop(self):
        "Stops the recordings."
        # Stop and clear the variables
        for var in self.variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target
            try:
                setattr(self.cyInstance, 'record_'+name, False)
                getattr(self.cyInstance, 'clear_'+name)()
            except:
                obj_desc = ''
                if isinstance(self.object, (Population, PopulationView)):
                    obj_desc = 'population '+self.object.name
                elif isinstance(self.object, Projection):
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                else:
                    obj_desc = 'dendrite between '+self.object.proj.pre.name+' and '+self.object.proj.post.name
                Global._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')

        self.variables = []
        self._recorded_variables = {}
        Global._network[0]['instance'].remove_recorder(self.cyInstance)
        self.cyInstance = None


    def get(self, variables=None, keep=False, reshape=False, force_dict=False):
        """
        Returns the recorded variables as a Numpy array (first dimension is time, second is neuron index).

        If a single variable name is provided, the recorded values for this variable are directly returned.
        If a list is provided or the argument left empty, a dictionary with all recorded variables is returned.

        The ``spike`` variable of a population will be returned as a dictionary of lists, where the spike times (in steps) for each recorded neurons are returned.

        *Parameters*:

        * **variables**: (list of) variables. By default, a dictionary with all variables is returned.

        * **keep**: defines if the content in memory for each variable should be kept (default: False).

        * **reshape**: transforms the second axis of the array to match the population's geometry (default: False).
        """

        def reshape_recording(self, data):
            if not reshape:
                return data
            else:
                return data.reshape((data.shape[0],) + self.object.geometry)

        def return_variable(self, name, keep):
            if isinstance(self.object, (Population, PopulationView)):
                return reshape_recording(self, self._get_population(self.object, name, keep))
            elif isinstance(self.object, (Dendrite, Projection)):
                data = self._get_dendrite(self.object, name, keep)
                # Dendrites have one empty dimension
                if isinstance(self.object, Dendrite):
                    data = data.squeeze()
                return data
            else:
                return None


        if variables:
            if not isinstance(variables, list):
                variables = [variables]
        else:
            variables = self.variables
            force_dict = True

        data = {}
        for var in variables:
            name = var
            # Sums of inputs for rate-coded populations
            if var.startswith('sum('):
                target = re.findall(r"\(([\w]+)\)", var)[0]
                name = '_sum_' + target

            # Retrieve the data
            data[var] = return_variable(self, name, keep)

            # Eventually reshape the array
            try:
                if not keep:
                    if self._recorded_variables[var]['stop'][-1] != Global.get_current_step(self.net_id):
                        self._recorded_variables[var]['start'][-1] = self._recorded_variables[var]['stop'][-1]
                        self._recorded_variables[var]['stop'][-1] = Global.get_current_step(self.net_id)
                else:
                    if self._recorded_variables[var]['stop'][-1] != Global.get_current_step(self.net_id):
                        self._recorded_variables[var]['stop'][-1] = Global.get_current_step(self.net_id)
            except:
                Global._warning('Monitor.get(): you try to get recordings which do not exist:', var)

        if not force_dict and len(variables)==1:
            return data[variables[0]]
        else:
            return data


    def _get_population(self, pop, name, keep):
        try:
            data = getattr(self.cyInstance, name)
            if not keep:
                getattr(self.cyInstance, 'clear_' + name)()
        except:
            data = []

        if name is not 'spike':
            return np.array(data)
        else:
            return data

    def _get_dendrite(self, proj, name, keep):
        try:
            data = getattr(self.cyInstance, name)
            if not keep:
                getattr(self.cyInstance, 'clear_' + name)()
        except:
            data = []
        return np.array(data)

    def times(self, variables=None):
        """ Returns the start and stop times of the recorded variables.

        *Parameters*:

        * **variables**: (list of) variables. By default, the times for all variables is returned.
        """
        import copy
        t = {}
        if variables:
            if not isinstance(variables, list):
                variables = [variables]
        else:
            variables = self.variables
        for var in variables:
            if not var in self.variables:
                continue
            t[var] = copy.deepcopy(self._recorded_variables[var])
        return t

    ###############################
    ### Spike visualisation stuff
    ###############################
    def raster_plot(self, spikes=None):
        """
        Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

        *Parameters*:

        * **spikes**: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

        Example::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            spike_times, spike_ranks = m.raster_plot()
            plot(spike_times, spike_ranks, '.')

        or::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            spikes = m.get('spike')
            spike_times, spike_ranks = m.raster_plot(spikes)
            plot(spike_times, spike_ranks, '.')

        """
        times = []; ranks=[]
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        # Compute raster
        for n in data.keys():
            for t in data[n]:
                times.append(t)
                ranks.append(n)

        return Global.dt()* np.array(times), np.array(ranks)

    def histogram(self, spikes=None, bins=None):
        """
        Returns a histogram for the recorded spikes in the population.

        *Parameters*:

        * **spikes**: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        * **bins**: the bin size in ms (default: dt).

        Example::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            histo = m.histogram()
            plot(histo)

        or::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            spikes = m.get('spike')
            histo = m.histogram(spikes)
            plot(histo)

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        if not bins:
            bins =  Global.config['dt']

        # Compute the duration of the recordings
        t_start = self._recorded_variables['spike']['start'][-1]
        duration = self._recorded_variables['spike']['stop'][-1] - self._recorded_variables['spike']['start'][-1]

        # Number of bins
        nb_bins = int(duration*Global.config['dt']/bins)

        # Initialize histogram
        histo = [0 for t in range(nb_bins)]

        # Compute histogram
        neurons = self.object.ranks if isinstance(self.object, PopulationView) else range(self.object.size)
        for neuron in neurons:
            for t in data[neuron]:
                histo[int((t-t_start)/float(bins/Global.config['dt']))] += 1

        return np.array(histo)

    def mean_fr(self, spikes=None):
        """
        Computes the mean firing rate in the population during the recordings.

        *Parameters*:

        * **spikes**: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

        Example::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            fr = m.mean_fr()

        or::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            spikes = m.get('spike')
            fr = m.mean_fr(spikes)

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes


        # Compute the duration of the recordings
        duration = self._recorded_variables['spike']['stop'][-1] - self._recorded_variables['spike']['start'][-1]

        # Number of neurons
        neurons = self.object.ranks if isinstance(self.object, PopulationView) else range(self.object.size)

        # Compute fr
        fr = 0
        for neuron in neurons:
            fr += len(data[neuron])

        return fr/float(len(neurons))/duration/Global.dt()*1000.0



    def smoothed_rate(self, spikes=None, smooth=0.):
        """
        Computes the smoothed firing rate of the recorded spiking neurons.

        The first axis is the neuron index, the second is time.

        *Parameters*:

        * **spikes**: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
        * **smooth**: smoothing time constant. Default: 0.0 (no smoothing).

        Example::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            r = m.smoothed_rate(smooth=100.)

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        import ANNarchy.core.cython_ext.Transformations as Transformations
        return Transformations.smoothed_rate(
            {
                'data': data,
                'start': self._recorded_variables['spike']['start'][-1],
                'stop': self._recorded_variables['spike']['stop'][-1]
            },
            smooth
        )

    def population_rate(self, spikes=None, smooth=0.):
        """
        Takes the recorded spikes of a population and returns a smoothed firing rate for the population of recorded neurons.

        This method is faster than calling ``smoothed_rate`` and then averaging.

        The first axis is the neuron index, the second is time.

        *Parameters*:

        * **spikes**: the dictionary of spikes returned by ``get('spike')``.
        * **smooth**: smoothing time constant. Default: 0.0 (no smoothing).

        If `spikes` is left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

        Example::

            m = Monitor(P[:1000], 'spike')
            simulate(1000.0)
            r = m.population_rate(smooth=100.)

        """
        if not 'spike' in self.variables:
            Global._error('Monitor: spike was not recorded')

        # Get data
        if not spikes:
            data = self.get('spike')
        else:
            if 'spike' in spikes.keys():
                data = spikes['spike']
            else:
                data = spikes

        import ANNarchy.core.cython_ext.Transformations as Transformations
        return Transformations.population_rate(
            {
                'data': data,
                'start': self._recorded_variables['spike']['start'][-1],
                'stop': self._recorded_variables['spike']['stop'][-1]
            },
            smooth
        )

class BoldMonitor(Monitor):
    """
    Specialized monitor for populations. Transforms the signal *variables* into a BOLD signal.

    Using the hemodynamic model as described in:

    * Friston et al. 2000: Nonlinear Responses in fMRI: The Balloon Model, Volterra Kernels, and Other Hemodynamics
    * Friston et al. 2003: Dynamic causal modelling

    TODO: more explanations
    """
    def __init__(self, obj, variables=[], epsilon=1.0, alpha=0.3215, kappa=0.665, gamma=0.412, E_0=0.3424, V_0=0.02, tau_s=0.8, tau_f=0.4, tau_0=1.0368, period=None, period_offset=None, start=True, net_id=0):
        """
        *Parameters*:

        * **obj**: object to monitor. Must be a Population or PopulationView.

        * **variables**: single variable name or list of variable names to record (default: []).

        * **epsilon**: TODO (default: 1.0)

        * **alpha**: TODO (default: 0.3215)

        * **kappa**: TODO (default: 0.665)

        * **gamma**: TODO (default: 0.412)

        * **E_0**: TODO (default: 0.3424)

        * **V_0**: TODO (default: 0.02)

        * **tau_s**: TODO (default: 0.8)

        * **tau_f**: TODO (default: 0.4)

        * **tau_0**: TODO (default: 1.0368)

        * **period**: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).

        * **period_offset**: determines the moment in ms of recording within the period (default 0). Must be smaller than **period**.

        * **start**: defines if the recording should start immediately (default: True). If not, you should later start the recordings with the ``start()`` method.
        """

        if not isinstance(obj, Population):
            Global._error("BoldMonitors can only record Populations.")

        super(BoldMonitor, self).__init__(obj, variables, period, period_offset, start, net_id)

        # Store the parameters
        self._epsilon = epsilon
        self._alpha = alpha
        self._kappa = kappa
        self._gamma = gamma
        self._E_0 = E_0
        self._V_0 = V_0
        self._tau_s = tau_s
        self._tau_f = tau_f
        self._tau_0 = tau_0

        # TODO: for now, we use the population id as unique identifier. This would be wrong,
        #       if multiple BoldMonitors could be attached to one population ...
        self._specific_template = {
            'cpp': """
// BoldMonitor pop%(pop_id)s (%(pop_name)s)
class BoldMonitor%(pop_id)s : public Monitor{
public:
    BoldMonitor%(pop_id)s(std::vector<int> ranks, int period, int period_offset, long int offset)
        : Monitor(ranks, period, period_offset, offset) {

        E = std::vector<%(float_prec)s>( ranks.size(), 0 );
        v = std::vector<%(float_prec)s>( ranks.size(), 0.02 );
        q = std::vector<%(float_prec)s>( ranks.size(), 0.0 );
        s = std::vector<%(float_prec)s>( ranks.size(), 0.0 );
        f_in = std::vector<%(float_prec)s>( ranks.size(), 1.0 );
        f_out = std::vector<%(float_prec)s>( ranks.size(), 0 );
        std::cout << "BoldMonitor initialized ... " << std::endl;
    }

    void record() {
        %(float_prec)s k1 = 7 * E_0;
        %(float_prec)s k2 = 2;
        %(float_prec)s k3 = 2*E_0 - 0.2;

        std::vector<%(float_prec)s> res = std::vector<%(float_prec)s>(ranks.size());
        int i = 0;
        for(auto it = ranks.begin(); it != ranks.end(); it++, i++) {
            %(float_prec)s u = pop%(pop_id)s.%(var_name)s[*it];

            E[i] = -pow(-E_0 + 1.0, 1.0/f_in[i]) + 1;
            f_out[i] = pow(v[i], 1.0/alpha);

            %(float_prec)s _v = (f_in[i] - f_out[i])/tau_0;
            %(float_prec)s _q = (E[i]*f_in[i]/E_0 - f_out[i]*q[i]/v[i])/tau_0;
            %(float_prec)s _s = epsilon*u - kappa*s[i] - gamma*(f_in[i] - 1);
            %(float_prec)s _f_in = s[i];

            v[i] += dt*_v;
            q[i] += dt*_q;
            s[i] += dt*_s;
            f_in[i] += dt*_f_in;

            res[i] = V_0*(k1*(-q[i] + 1) + k2*(-q[i]/v[i] + 1) + k3*(-v[i] + 1));
        }

        // store the result
        out_signal.push_back(res);
    }

    long int size_in_bytes() {
        long int size_in_bytes = 0;
        return size_in_bytes;
    }

    void record_targets() {} // nothing to do here ...

    std::vector< std::vector<%(float_prec)s> > out_signal;

    %(float_prec)s epsilon;
    %(float_prec)s alpha;
    %(float_prec)s kappa;
    %(float_prec)s gamma;
    %(float_prec)s E_0;
    %(float_prec)s V_0;
    %(float_prec)s tau_s;
    %(float_prec)s tau_f;
    %(float_prec)s tau_0;

private:
    %(float_prec)s k1_;
    %(float_prec)s k2_;
    %(float_prec)s k3_;

    std::vector<%(float_prec)s> E;
    std::vector<%(float_prec)s> v;
    std::vector<%(float_prec)s> q;
    std::vector<%(float_prec)s> s;
    std::vector<%(float_prec)s> f_in;
    std::vector<%(float_prec)s> f_out;
};
""",
            'pyx_struct': """

    # Population %(pop_id)s (%(pop_name)s) : Monitor
    cdef cppclass BoldMonitor%(pop_id)s (Monitor):
        BoldMonitor%(pop_id)s(vector[int], int, int, long) except +
        long int size_in_bytes()

        vector[vector[%(float_prec)s]] out_signal
        %(float_prec)s epsilon
        %(float_prec)s alpha
        %(float_prec)s kappa
        %(float_prec)s gamma
        %(float_prec)s E_0
        %(float_prec)s V_0
        %(float_prec)s tau_s
        %(float_prec)s tau_f
        %(float_prec)s tau_0

""",
            'pyx_wrapper': """

# Population Monitor wrapper
cdef class BoldMonitor%(pop_id)s_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, period_offset, long offset):
        self.thisptr = new BoldMonitor%(pop_id)s(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (<BoldMonitor%(pop_id)s *>self.thisptr).size_in_bytes()

    # Output
    property out_signal:
        def __get__(self): return (<BoldMonitor%(pop_id)s *>self.thisptr).out_signal

    # Parameters
    property epsilon:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).epsilon = val
    property alpha:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).alpha = val
    property kappa:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).kappa = val
    property gamma:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).gamma = val
    property E_0:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).E_0 = val
    property V_0:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).V_0 = val
    property tau_s:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).tau_s = val
    property tau_f:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).tau_f = val
    property tau_0:
        def __set__(self, val): (<BoldMonitor%(pop_id)s *>self.thisptr).tau_0 = val

"""
        }

    #######################################
    ### Attributes
    #######################################
    # epsilon
    @property
    def epsilon(self):
        "TODO"
        if not self.cyInstance:
            return self._epsilon
        else:
            return self.cyInstance.epsilon
    @epsilon.setter
    def epsilon(self, val):
        if not self.cyInstance:
            self._epsilon = val
        else:
            self.cyInstance.epsilon = val
    # alpha
    @property
    def alpha(self):
        "TODO"
        if not self.cyInstance:
            return self._alpha
        else:
            return self.cyInstance.alpha
    @alpha.setter
    def alpha(self, val):
        if not self.cyInstance:
            self._alpha = val
        else:
            self.cyInstance.alpha = val
    # kappa
    @property
    def kappa(self):
        "TODO"
        if not self.cyInstance:
            return self._kappa
        else:
            return self.cyInstance.kappa
    @kappa.setter
    def kappa(self, val):
        if not self.cyInstance:
            self._kappa = val
        else:
            self.cyInstance.kappa = val
    # gamma
    @property
    def gamma(self):
        "TODO"
        if not self.cyInstance:
            return self._gamma
        else:
            return self.cyInstance.gamma
    @gamma.setter
    def gamma(self, val):
        if not self.cyInstance:
            self._gamma = val
        else:
            self.cyInstance.gamma = val
    # E_0
    @property
    def E_0(self):
        "TODO"
        if not self.cyInstance:
            return self._E_0
        else:
            return self.cyInstance.E_0
    @E_0.setter
    def E_0(self, val):
        if not self.cyInstance:
            self._E_0 = val
        else:
            self.cyInstance.E_0 = val
    # V_0
    @property
    def V_0(self):
        "TODO"
        if not self.cyInstance:
            return self._V_0
        else:
            return self.cyInstance.V_0
    @V_0.setter
    def V_0(self, val):
        if not self.cyInstance:
            self._V_0 = val
        else:
            self.cyInstance.V_0 = val
    # tau_s
    @property
    def tau_s(self):
        "TODO"
        if not self.cyInstance:
            return self._tau_s
        else:
            return self.cyInstance.tau_s
    @tau_s.setter
    def tau_s(self, val):
        if not self.cyInstance:
            self._tau_s = val
        else:
            self.cyInstance.tau_s = val
    # tau_f
    @property
    def tau_f(self):
        "TODO"
        if not self.cyInstance:
            return self._tau_f
        else:
            return self.cyInstance.tau_f
    @tau_f.setter
    def tau_f(self, val):
        if not self.cyInstance:
            self._tau_f = val
        else:
            self.cyInstance.tau_f = val
    # tau_0
    @property
    def tau_0(self):
        "TODO"
        if not self.cyInstance:
            return self._tau_0
        else:
            return self.cyInstance.tau_0
    @tau_0.setter
    def tau_0(self, val):
        if not self.cyInstance:
            self._tau_0 = val
        else:
            self.cyInstance.tau_0 = val

    #######################################
    ### Data access
    #######################################
    def _start_bold_monitor(self):
        """
        Automatically called from Compiler._instantiate()
        """
        # Create the wrapper
        period = int(self._period/Global.config['dt'])
        period_offset = int(self._period_offset/Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'BoldMonitor'+str(self.object.id)+'_wrapper')(self.object.ranks, period, period_offset, offset)
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

        # Set the parameter
        self.cyInstance.epsilon = self._epsilon
        self.cyInstance.alpha = self._alpha
        self.cyInstance.kappa = self._kappa
        self.cyInstance.gamma = self._gamma
        self.cyInstance.E_0 = self._E_0
        self.cyInstance.V_0 = self._V_0
        self.cyInstance.tau_s = self._tau_s
        self.cyInstance.tau_f = self._tau_f
        self.cyInstance.tau_0 = self._tau_0

    def get(self):
        """
        Get the recorded BOLD signal.
        """
        return self._get_population(self.object, "out_signal", True)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

class MemoryStats(object):
    """
    Create memory statistics for the main objects in ANNarchy. The current implementation
    focusses on the C++ simulation core. But this module could be further extended to measure
    also the Python objects.
    """
    def __init__(self):
        pass

    def print_cpp(self, net_id=0):
        """
        Print memory consumption of CPP objects. The method calls
        the size_in_bytes() methods implemented by the C++ modules.
        """
        for pop in Global._network[net_id]['populations']:
            if hasattr(pop, 'size_in_bytes'):
                print(pop.name, ":", self._human_readable_bytes(pop.size_in_bytes()))
            else:
                Global._warning("MemoryStats.print_cpp(): the object", pop, "does not have a size_in_bytes() function.")

        for proj in Global._network[net_id]['projections']:
            if hasattr(proj, 'size_in_bytes'):
                print(proj.pre.name, "->", proj.post.name, "(", proj.target, "):", self._human_readable_bytes(proj.size_in_bytes()))
            else:
                Global._warning("MemoryStats.print_cpp(): the object", proj, "does not have a size_in_bytes() function.")

        for mon in Global._network[net_id]['monitors']:
            if hasattr(proj, 'size_in_bytes'):
                print("Monitor on", mon.object.name, ":", self._human_readable_bytes(mon.size_in_bytes()))
            else:
                Global._warning("MemoryStats.print_cpp(): the object", mon, "does not have a size_in_bytes() function.")

    def _human_readable_bytes(self, num):
        """
        All cpp functions return there size in bytes *num* as long int. This function
        divides this by 1024 until the result is lower than the next unit.
        """
        for x in ['bytes','KB','MB','GB']:
            if num < 1024.0:
                return "%3.2f %s" % (num, x)
            num /= 1024.0
        return "%3.1f%s" % (num, 'TB')

######################
# Static methods to plot spike patterns without a Monitor (e.g. offline)
######################
def raster_plot(spikes):
    """
    Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

    *Parameters*:

    * **spikes**: the dictionary of spikes returned by ``get('spike')``.

    Example::

        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        spike_times, spike_ranks = raster_plot(spikes)
        plot(spike_times, spike_ranks, '.')

    """
    times = []; ranks=[]

    # Compute raster
    for n in spikes.keys():
        for t in spikes[n]:
            times.append(t)
            ranks.append(n)

    return Global.dt()* np.array(times), np.array(ranks)


def histogram(spikes, bins=None):
    """
    Returns a histogram for the recorded spikes in the population.

    *Parameters*:

    * **spikes**: the dictionary of spikes returned by ``get('spike')``.
    * **bins**: the bin size in ms (default: dt).

    Example::

        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        histo = histogram(spikes)
        plot(histo)

    """
    if bins is None:
        bins =  Global.config['dt']

    bin_step = int(bins/Global.config['dt'])

    # Compute the duration of the recordings
    t_maxes = []
    t_mines = []
    for neuron in spikes.keys():
        if len(spikes[neuron]) == 0 : continue
        t_maxes.append(np.max(spikes[neuron]))
        t_mines.append(np.min(spikes[neuron]))

    t_max = np.max(t_maxes)
    t_min = np.min(t_mines)
    duration = t_max - t_min

    # Number of bins
    nb_bins = int(duration/bin_step)
    print(t_min, t_max, duration, nb_bins)

    # Initialize histogram
    histo = [0 for t in range(nb_bins+1)]

    # Compute per step histogram
    for neuron in spikes.keys():
        for t in spikes[neuron]:
            histo[int((t-t_min)/float(bin_step))] += 1

    return np.array(histo)

def population_rate(spikes, smooth=0.0):
    """
    Takes the recorded spikes of a population and returns a smoothed firing rate for the population of recorded neurons.

    This method is faster than calling ``smoothed_rate`` and then averaging.

    The first axis is the neuron index, the second is time.

    *Parameters*:

    * **spikes**: the dictionary of spikes returned by ``get('spike')``.
    * **smooth**: smoothing time constant. Default: 0.0 (no smoothing).

    Example::

        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        r = population_rate(smooth=100.)

    """
    # Compute the duration of the recordings
    t_maxes = []
    t_mines = []
    for neuron in spikes.keys():
        if len(spikes[neuron]) == 0 : continue
        t_maxes.append(np.max(spikes[neuron]))
        t_mines.append(np.min(spikes[neuron]))

    t_max = np.max(t_maxes)
    t_min = np.min(t_mines)

    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.population_rate(
        {
            'data': spikes,
            'start':t_min,
            'stop': t_max
        },
        smooth
    )

def smoothed_rate(spikes, smooth=0.):
    """
    Computes the smoothed firing rate of the recorded spiking neurons.

    The first axis is the neuron index, the second is time.

    *Parameters*:

    * **spikes**: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.
    * **smooth**: smoothing time constant. Default: 0.0 (no smoothing).

    Example::

        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        r = smoothed_rate(smooth=100.)

    """
    # Compute the duration of the recordings
    t_maxes = []
    t_mines = []
    for neuron in spikes.keys():
        if len(spikes[neuron]) == 0 : continue
        t_maxes.append(np.max(spikes[neuron]))
        t_mines.append(np.min(spikes[neuron]))

    t_max = np.max(t_maxes)
    t_min = np.min(t_mines)

    import ANNarchy.core.cython_ext.Transformations as Transformations
    return Transformations.smoothed_rate(
        {
            'data': spikes,
            'start': t_min,
            'stop': t_max
        },
        smooth
    )

def mean_fr(spikes, duration=None):
    """
    Computes the mean firing rate in the population during the recordings.

    *Parameters*:

    * **spikes**: the dictionary of spikes returned by ``get('spike')``.
    * **duration**: duration of the recordings. By default, the mean firing rate is computed between the first and last spikes of the recordings.

    Example::

        m = Monitor(P[:1000], 'spike')
        simulate(1000.0)
        spikes = m.get('spike')
        fr = mean_fr(spikes)

    """
    if duration is None:

        # Compute the duration of the recordings
        t_maxes = []
        t_mines = []
        for neuron in spikes.keys():
            if len(spikes[neuron]) == 0 : continue
            t_maxes.append(np.max(spikes[neuron]))
            t_mines.append(np.min(spikes[neuron]))

        t_max = np.max(t_maxes)
        t_min = np.min(t_mines)
        duration = t_max - t_min

    nb_neurons = len(spikes.keys())

    # Compute fr
    fr = 0
    for neuron in spikes:
        fr += len(spikes[neuron])

    return fr/float(nb_neurons)/duration/Global.dt()*1000.0
