""" 

    Profile.py
    
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
import exceptions
import datetime
import pyqtgraph as pg
from math import ceil, floor, sqrt

from ANNarchy.core import Global
from ANNarchy import *

from DataLog import DataLog
from Custom import IntAxis

class Profile:
    """
    Contains all the functionality to retrieve, analyze and visualize the profiling of **one** network instance. 
    """
    def __init__(self, num_threads, num_trials, name=None, folder='.'):
        """
        Constructor, setup the overall configuration of profiling session.
        
        Parameter:
        
            * *num_threads*: thread configuration, array consisting of thread amounts of the different runs.
            * *num_trials*: amount of measurement points for each thread configuration
        """
        try:
            import ANNarchyCython
        except exceptions.ImportError:
            print 'Error on Profile'
        else:
            print 'Inited profiler.'
            self._profile_instance = ANNarchyCython.pyProfile()
            self._network = ANNarchyCython.pyNetwork()
        
        self._folder = folder
        self._name = name
        self._threads = {}
        for i in range(len(num_threads)):
            self._threads.update({ num_threads[i]: i })

        self._num_trials = num_trials
            
        self._net_data = None
        self._pop_data = {}

    def add_to_profile(self, object):
        """
        Which network objects should be tracked.
        
        Parameter:
        
            * *object*: either 'network' or a population name
        """
        if object == "network":
            self._net_data = DataLog(self._threads, self._num_trials,'overall')
        elif isinstance(object, Population):
            if object.description['type'] == 'rate':
                self._pop_data[object.name] = { 'sum' : DataLog(self._threads, self._num_trials, 'sum'),
                                                'step' : DataLog(self._threads, self._num_trials, 'step'), 
                                                'local' : DataLog(self._threads, self._num_trials, 'local'),
                                                'global' : DataLog(self._threads, self._num_trials, 'global')
                                              }
            else:
                print 'TODO: ...'
        else:
            self._net_data = DataLog(self._threads, self._num_trials,'overall')
            
            for pop in object._populations:
                if pop.description['type'] == 'rate':
                    self._pop_data[pop.name] = { 'sum' : DataLog(self._threads, self._num_trials, 'sum'),
                                                 'step' : DataLog(self._threads, self._num_trials, 'step'), 
                                                 'local' : DataLog(self._threads, self._num_trials, 'local'),
                                                 'global' : DataLog(self._threads, self._num_trials, 'global')
                                               }
                else:
                    self._pop_data[pop.name] = { 'cond' : DataLog(self._threads, self._num_trials, 'cond'),
                                                 'del' : DataLog(self._threads, self._num_trials, 'del'),
                                                 'pre' : DataLog(self._threads, self._num_trials, 'pre'),
                                                 'post' : DataLog(self._threads, self._num_trials, 'post')                                                 
                                               }
                    