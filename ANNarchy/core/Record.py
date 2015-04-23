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
    
    def __init__(self, obj, variables, period=None):
        """        
        *Parameters*:
        
        * **obj**: object to monitor. Must be a Population, PopulationView or Dendrite object.
        
        * **variables**: single variable name or list of variable names to record.  

        * **period**: delay in ms between two recording (default: dt). Not valid for the ``spike`` variable.

        Example::

            m = Monitor(pop1, ['v', 'spike'], period=10.0, ranks=range(:100))

        """
        # Store arguments
        self.object = obj

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        if not period:
            self.period = Global.config['dt']
        else:
            self.period = float(period)

        # Structure to store the recordings
        self.recorded_variables = {}

        # Add the population to the global variable
        Global._monitors.append(self)
        if Global._compiled: # Already compiled
            self._init_monitoring()

    def _init_monitoring(self):
        "To be called after compile() as it accesses cython objects"
        # Start recording
        if isinstance(self.object, Population):
            self.cyInstance = self.object.cyInstance
            self._start_population()
        elif isinstance(self.object, PopulationView):
            self.cyInstance = self.object.population.cyInstance
            self._start_population()
        elif isinstance(self.object, Dendrite):
            _error('Recording projections is not implemented yet')



    def _start_population(self):
        "Starts the recording for a population."
        # set period and offset
        self.cyInstance.set_record_period( int(self.period/Global.config['dt']), Global.get_current_step() )

        # set ranks
        self.cyInstance.set_record_ranks( [-1] if isinstance(self.object, Population) else self.object.ranks )

        # start recording of variables
        for var in self.variables:
            if not var in self.object.variables + ['spike']:
                Global._error('Monitor: ' + var, 'is not a recordable variable of the population' + self.object.name)
                exit(0)

            self.recorded_variables[var] = {'start': [Global.get_current_step()], 'stop': [-1]}

            try:
                getattr(self.cyInstance, 'start_record_'+var)()
            except:
                Global._error('Monitor:' + var + 'can not be recorded.')
                exit(0)

    def get(self, name):
        """
        Return the recorded values of the provided variable.
        """
        if not name in self.variables:
            Global._error(name + ' is not a recorded variable for this monitor.')
            exit(0)

        if isinstance(self.object, (Population, PopulationView)):
            return self._get_population(self.object, name)
        elif isinstance(self.object, Dendrite):
            _error('Recording projections is not implemented yet')
            exit(0)


    def _get_population(self, pop, name):    
        try:
            data = getattr(self.cyInstance, 'get_record_'+name)()
            getattr(self.cyInstance, 'clear_record_'+name)()
        except:
            Global._error('Monitor: ' + name + ' is not a recordable variable.')
            exit(0)

        return np.array(data)