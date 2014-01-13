"""

    Network.py
    
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
from Population import Population
from Projection import Projection

from ANNarchy4 import generator
from ANNarchy4.core import Global

class Network(object):
    """
    Represents a network in ANNarchy.
    """
    
    def __init__(self, *args):
        """
        Initialised with any collection of objects that should be added to the Network. Objects that need to passed to the Network object are either Population or Projection objects.
        """
        self._populations = []
        self._projections = []
        
        for object in args:
            self.add(object)
        
    def compile(self):
        """
        Compile all classes and setup the network
        """
        generator.compile(populations = self._populations, projections = self._projections)
    
    def add(self, object):
        """
        Add additional object to network.
        """
        if isinstance(object, Population):
            self._populations.append(object)
        elif isinstance(object, Projection):
            self._projections.append(object)        
        else:
            print 'wrong argument provided to Network.add()'

    
    def remove(self, object):
        """
        Remove the object from the Network.
        """
        if isinstance(object, Population):
            self._populations = self._populations.remove(object)
        elif isinstance(object, Projection):
            self._projections = self._projections.remove(object)         
        else:
            print 'wrong argument provided to Network.add()'

    
    def reset(self, states=False, connections=False):
        """
        Reinitialises the network, runs each object's reset() method (resetting them to 0). If states=False then it will not reinitialise the NeuronGroup state variables. If connections=False then it will not reinitialise the NeuronGroup state variables.
        """
        Global.reset(states, connections)
    
    def simulate(self, duration):
        """
        Runs the network for the given duration.
        """
        Global.simulate(duration)