"""

    test_NumericalMethod.py

    This file is part of ANNarchy.

    Copyright (C) 2017-2019 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
import unittest
from ANNarchy import *

class test_ExponentialMethod(unittest.TestCase):
    """
    Test the code generation for equations evaluated with exponential scheme
    
    TODO: until now, only the successful compilation is tested, some value tests need to be added ...
    """
    @classmethod
    def setUpClass(self):
        neuron = Neuron(equations="r=sum(exc); dd/dt = -d: population, exponential")
        
        synapse = Synapse(parameters="tau = 10.0 :projection", equations="dd/dt = -d/tau : projection, exponential")
        
        pop=Population(10, neuron)
        proj=Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(Uniform(1.0, 2.0))
        
        compile()
        
    def setUp(self):
        self.test_net.reset() # network reset
