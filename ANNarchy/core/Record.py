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

import numpy as np

class Monitor(object):
    """
    Monitoring class allowing to record easily variables from Population, PopulationView and Dendrite objects.
    """
    
    def __init__(self, obj, variables=[], period=None, start=True):
        """        
        *Parameters*:
        
        * **obj**: object to monitor. Must be a Population, PopulationView or Dendrite object.
        
        * **variables**: single variable name or list of variable names to record (default: []).  

        * **period**: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable of a Population(View).

        * **start**: defines if the recording should start immediately (default: True). If not, you should later start the recordings with the ``start()`` method.

        Example::

            m = Monitor(pop1, ['v', 'spike'], period=10.0, ranks=range(:100))

        """

        # Object to record (Population, PopulationView, Dendrite)
        self.object = obj

        # Variables to record
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        # Period
        if not period:
            self.period = Global.config['dt']
        else:
            self.period = float(period)

        # Start
        self._start = start
        self._recorded_variables = {}

        # Add the population to the global variable
        Global._monitors.append(self)
        if Global._compiled: # Already compiled
            self._init_monitoring()

    def _add_variable(self, var):
        if not var in self.variables:
            self.variables.append(var)
        self._recorded_variables[var] = {'start': [Global.get_current_step()], 'stop': [Global.get_current_step()]}

    def _init_monitoring(self):
        "To be called after compile() as it accesses cython objects"
        # Start recording
        if isinstance(self.object, (Population, PopulationView)):
            self._start_population()
        elif isinstance(self.object, Dendrite):
            _error('Recording projections is not implemented yet')

    def _start_population(self):
        "Creates the C++ object and starts the recording for a population."

        if isinstance(self.object, PopulationView):
            self.ranks = self.object.ranks
        else:
            self.ranks = [-1]

        # Create the wrapper
        self.cyInstance = getattr(Global._network, 'PopRecorder'+str(self.object.id)+'_wrapper')(self.ranks, int(self.period/Global.config['dt']), Global.get_current_step())
        Global._network.add_recorder(self.cyInstance)

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
            self.period = period
            self.cyInstance.period = int(self.period/Global.config['dt'])
            self.cyInstance.offset = Global.get_current_step()

        for var in variables:
            try:
                setattr(self.cyInstance, 'record_'+var, True)
            except:
                Global._warning('Monitor: ' + var + ' can not be recorded.')

    def resume(self):
        "Resumes the recordings."
        # Start recording the variables
        for var in self.variables:
            try:
                setattr(self.cyInstance, 'record_'+var, True)
            except:
                Global._warning('Monitor:' + var + 'can not be recorded.')
            self._recorded_variables[var]['start'].append(Global.get_current_step())

    def pause(self):
        "Resumes the recordings."
        # Start recording the variables
        for var in self.variables:
            try:
                setattr(self.cyInstance, 'record_'+var, False)
            except:
                Global._warning('Monitor:' + var + 'can not be recorded.')
            self._recorded_variables[var]['stop'].append(Global.get_current_step())

    def stop(self):
        "Stops the recordings."
        # Start recording the variables
        for var in self.variables:
            try:
                setattr(self.cyInstance, 'record_'+var, False)
            except:
                Global._warning('Monitor:' + var + 'can not be recorded.')
        self.variables = []
        self._recorded_variables = {}


    def get(self, variables=None, keep=False, force_dict=False):
        """
        Returns the recorded variables as a Numpy array (first dimension is time, second is neuron index).

        If a single variable name is provided, the recorded values for this variable are directly returned. If a list is provided or the argument left empty, a dictionary with all recorded variables is returned. 

        The ``spike`` variable of a population will be returned as a list of lists, where the ranks of the neurons which fired at each step are returned.

        *Parameters*:
        
        * **variables**: (list of) variables. By default, a dictionary with all variables is returned.

        * **keep**: defines if the content in memory for each variable should be kept (default: False).
        """

        def return_variable(self, name, keep):
            if isinstance(self.object, (Population, PopulationView)):
                return self._get_population(self.object, name, keep)
            elif isinstance(self.object, Dendrite):
                _error('Recording projections is not implemented yet')
                exit(0)

        if variables:
            if not isinstance(variables, list):
                variables = [variables]
        else:
            variables = self.variables

        data = {}
        for var in variables:
            data[var] = return_variable(self, var, keep)
            if not keep:
                if self._recorded_variables[var]['stop'][-1] != Global.get_current_step():
                    self._recorded_variables[var]['start'][-1] = self._recorded_variables[var]['stop'][-1]
                    self._recorded_variables[var]['stop'][-1] = Global.get_current_step()
            else:
                if self._recorded_variables[var]['stop'][-1] != Global.get_current_step():
                    self._recorded_variables[var]['stop'][-1] = Global.get_current_step()

        if not force_dict and len(variables)==1:
            return data[variables[0]]
        else:
            return data


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