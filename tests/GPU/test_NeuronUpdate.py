"""

    test_NeuronUpdate.py

    This file is part of ANNarchy.

    Copyright (C) 2016-2018 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
setup(paradigm="cuda", seed=1)

# neuron defintions common used for test cases
LocalEquation = Neuron(
    equations = """
        noise = Uniform(0,1)
	    r = t
    """
)

GlobalEquation = Neuron(
    parameters = "",
    equations = """
        noise = Uniform(0,1) : population
        glob_r = t : population
        r = t
    """
)

MixedEquation = Neuron(
    parameters = "",
    equations = """
        glob_r = t : population
        r = glob_r
    """
)

tc_loc_up_pop = Population(3, LocalEquation)
tc_glob_up_pop = Population(3, GlobalEquation)

class test_LocalUpdate(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([tc_loc_up_pop])
        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(tc_loc_up_pop)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset() # network reset

    def testSingleUpdate(self):
        self.test_net.simulate(5)

        # after 5ms simulation, r should be at 4
        self.assertTrue(numpy.allclose(self.net_pop.r, [4.0, 4.0, 4.0]))

    def testRandomVariable(self):
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.net_pop.noise, [0.72449183, 0.72449183, 0.72449183]))

class test_GlobalUpdate(unittest.TestCase):
    """
    Test the correct evaluation of global equation updates.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([tc_glob_up_pop])
        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(tc_glob_up_pop)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset() # network reset

    def testSingleUpdate(self):
        self.test_net.simulate(5)

        # after 5ms simulation, glob_r should be at 4
        self.assertTrue(numpy.allclose(self.net_pop.glob_r, [4.0]))

    def testRandomVariable(self):
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.net_pop.noise, [0.724491826019]))
