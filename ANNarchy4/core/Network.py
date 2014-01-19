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
from .Population import Population
from .Projection import Projection

from ANNarchy4 import generator
from . import Global

from math import ceil

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
        self.cython_module = None
        
        for object in args:
            self.add(object)
        
    def compile(self, clean=False, debug_build=False, cpp_stand_alone=False, profile_enabled=False):
        """
        Compile all classes and setup the network
    
        Parameters:
        
        * *clean*: boolean to specifying if the library should be recompiled entirely or only the changes since last compilation (default: False).
        * *debug_build*: creates a debug version of ANNarchy, which logs the creation of objects and some other data (default: False).
    
        .. hint: these parameters are also available but should only used if performance issues exists
    
            * *cpp_stand_alone*: creates a cpp library solely. It's possible to run the simulation, but no interaction possibilities exist. These argument should be always False.
            * *profile_enabled*: creates a profilable version of ANNarchy, which logs several computation timings (default: False).
        
        """
        generator.compile(clean, self._populations, self._projections, cpp_standalone, debug_build, profile_enabled)
        
        # copy back the data
        self.cython_module = __import__('ANNarchyCython')
        self.cy_instance = self.cython_module.pyNetwork()
    
    def add(self, object):
        """
        Add additional object to network.
        """
        if isinstance(object, Population):
            self._populations.append(object)
        elif isinstance(object, Projection):
            self._projections.append(object)        
        else:
            Global._error('wrong argument provided to Network.add()')

    
    def remove(self, object):
        """
        Remove the object from the Network.
        """
        if isinstance(object, Population):
            self._populations = self._populations.remove(object)
        elif isinstance(object, Projection):
            self._projections = self._projections.remove(object)         
        else:
            Global._error('wrong argument provided to Network.add()')

    
    def reset(self, states=False, connections=False):
        """
        Reinitialises the network, runs each object's reset() method (resetting them to 0).
    
        Parameter:
    
        * *states*: if set to True then it will reinitialise the neuron state variables.
        * *connections*: if set to True then it will reinitialise the connection variables.
        """
        if states:
            for pop in self._populations:
                pop.reset()
                
        if connections:
            Global._error('reset of projections is currently not implemented')
            
    def get_population(self, name):
        """
        Returns population corresponding to *name*.
        
        Parameter:
        
        * *name*: population name
    
        Returns:
        
        the requested population if existing otherwise None is returned.
        """
        for pop in self._populations:
            if pop.name == name:
                return pop
            
        Global._error('no population with the name',name,'found.')
        return None
    
    def get_projection(self, pre, post, target):
        """
        Returns projection corresponding to the arguments.
        
        Parameters:
        
        * *pre*: presynaptic population
        * *post*: postsynaptic population
        * *target*: connection type
        
        Returns:
        
        the requested projection if existing otherwise None is returned.
        """
        for proj in self._projections:
            
            if proj.post == post:
                if proj.pre == pre:
                    if proj.target == target:
                        return proj
        
        Global._error("no projection '", pre.name, "'->'", post.name, "' with target '", target,"' found.")
        return None
    
    def simulate(self, duration):
        """
        Runs the network for the given duration.
        """
        nb_steps = ceil(duration / Global.config['dt'])
        self.cy_instance.Run(nb_steps)
        
    def current_time(self):
        """
        Returns current simulation time in ms.
        
        **Note**: computed as number of simulation steps times dt
        """
        self.cy_instance.get_time() * config['dt']
    
    def current_step(self):
        """
        Returns current simulation step.
        """
        self.cy_instance.get_time()    
    
    def set_current_time(self, time):
        """
        Set current simulation time in ms.
        
        **Note**: computed as number of simulation steps times dt
        """
        self.cy_instance.set_time(int( time / config['dt']))
    
    def set_current_step(self, time):
        """
        set current simulation step.
        """
        self.cy_instance.set_time( time )    
        
class MagicNetwork(Network):
    """
    Creates a Network object from any suitable objects
    """
    def __init__(self):
        Network.__init__(self)
        
        self._populations = Global._populations
        self._projections = Global._projections
        