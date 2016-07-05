"""

    Record.py

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
from . import Global
from .Population import Population
from .PopulationView import PopulationView
from .Dendrite import Dendrite

import numpy as np
import re

class Monitor(object):
    """
    Monitoring class allowing to record easily variables from Population, PopulationView and Dendrite objects.
    """

    def __init__(self, obj, variables=[], period=None, start=True, net_id=0):
        """
        *Parameters*:

        * **obj**: object to monitor. Must be a Population, PopulationView or Dendrite object.

        * **variables**: single variable name or list of variable names to record (default: []).

        * **period**: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).

        * **start**: defines if the recording should start immediately (default: True). If not, you should later start the recordings with the ``start()`` method.

        Example::

            m = Monitor(pop, ['g_exc', 'v', 'spike'], period=10.0, ranks=range(:100))

        It is also possible to record the sum of inputs to each neuron in a rate-coded population::

            m = Monitor(pop, ['sum(exc)', 'r'])

        """
        # Object to record (Population, PopulationView, Dendrite)
        self.object = obj
        self.cyInstance = None
        self.net_id = net_id

        # Variables to record
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        # Period
        if not period:
            self._period = Global.config['dt']
        else:
            self._period = float(period)

        # Start
        self._start = start
        self._recorded_variables = {}

        # Add the population to the global variable
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

    def _add_variable(self, var):
        if not var in self.variables:
            self.variables.append(var)
        self._recorded_variables[var] = {'start': [Global.get_current_step(self.net_id)], 'stop': [Global.get_current_step(self.net_id)]}

    def _init_monitoring(self):
        "To be called after compile() as it accesses cython objects"
        # Start recording
        if isinstance(self.object, (Population, PopulationView)):
            self._start_population()
        elif isinstance(self.object, Dendrite):
            self._start_dendrite()

    def _start_population(self):
        "Creates the C++ object and starts the recording for a population."

        if isinstance(self.object, PopulationView):
            self.ranks = self.object.ranks
        else:
            self.ranks = [-1]

        # Create the wrapper
        period = int(self._period/Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'PopRecorder'+str(self.object.id)+'_wrapper')(self.ranks, period, offset)
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

        for var in self.variables:
            self._add_variable(var)

        # Start recordings if enabled
        if self._start:
            self.start()

    def _start_dendrite(self):
        "Creates the C++ object and starts the recording for a dendrite."

        self.ranks = self.object.post_rank
        self.idx = self.object.idx

        # Create the wrapper
        period = int(self._period/Global.config['dt'])
        offset = Global.get_current_step(self.net_id) % period
        self.cyInstance = getattr(Global._network[self.net_id]['instance'], 'ProjRecorder'+str(self.object.proj.id)+'_wrapper')([self.idx], period, offset)
        Global._network[self.net_id]['instance'].add_recorder(self.cyInstance)

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
                else:
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
                Global._warning('Monitor:' + var + ' can not be recorded ('+obj_desc+')')

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
                else:
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
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
                else:
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
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
                else:
                    obj_desc = 'projection between '+self.object.pre.name+' and '+self.object.post.name
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
            elif isinstance(self.object, Dendrite):
                return self._get_dendrite(self.object, name, keep)


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
        """ Returns two vectors representing for each recorded spike 1) the spike times and 2) the ranks of the neurons.

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
        """ Returns a histogram for the recorded spikes in the population.

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

        # Number of neurons
        nb_neurons = self.object.size

        # Number of bins
        nb_bins = int(duration*Global.config['dt']/bins)

        # Initialize histogram
        histo = [0 for t in range(nb_bins)]

        # Compute histogram
        for neuron in range(nb_neurons):
            for t in data[neuron]:
                histo[int((t-t_start)/float(bins/Global.config['dt']))] += 1

        return np.array(histo)

    def mean_fr(self, spikes=None):
        """ Computes the mean firing rate in the population during the recordings.

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
        nb_neurons = self.object.size

        # Compute fr
        fr = 0
        for neuron in range(nb_neurons):
            fr += len(data[neuron])

        return fr/float(nb_neurons)/duration/Global.dt()*1000.0



    def smoothed_rate(self, spikes=None, smooth=0.):
        """ Computes the smoothed firing rate of the recorded spiking neurons.

        The first axis is the neuron index, the second is time.

        *Parameters*:

        * **spikes**: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

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
        """ Takes the recorded spikes of a population and returns a smoothed firing rate for the population of recorded neurons.

        This method is faster than calling ``smoothed_rate`` and averaging.

        *Parameters*:

        * **spikes**: the dictionary of spikes returned by ``get('spike')``. If left empty, ``get('spike')`` will be called. Beware: this erases the data from memory.

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
