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
from ANNarchy4 import *

class Profile:
    def __init__(self):
        try:
            import ANNarchyCython
        except exceptions.ImportError:
            print 'Error on Profile'
        else:
            print 'Inited profiler.'
            self._profile_instance = ANNarchyCython.pyProfile()
            self._network = ANNarchyCython.pyNetwork()
            
    def reset_timer(self):
        self._profile_instance.resetTimer()

    def last_step_sum(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeSum(pop.class_name)
        
    def average_sum(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeSum(pop.class_name, begin, end)

    def last_step_step(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeStep(pop.class_name)
        
    def average_step(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeStep(pop.class_name, begin, end)

    def last_step_local(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeLocal(pop.class_name)
        
    def average_local(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeLocal(pop.class_name, begin, end)

    def last_step_global(self, name):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.lastTimeGlobal(pop.class_name)
        
    def average_global(self, name, begin, end):
        if isinstance(name, str):
            pop = get_population(name)
            return self._profile_instance.avgTimeGlobal(pop.class_name, begin, end)

    def set_num_threads(self, threads):
        self._network.set_num_threads(threads)
        
