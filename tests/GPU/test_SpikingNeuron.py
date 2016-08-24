"""

    test_SpikingNeuron.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
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
import unittest
import numpy

from ANNarchy import *
setup(paradigm="cuda")

neuron1 = Neuron(
    equations="""
        v = v + 1.0
    """,
    spike = "v == 3.0",
    reset = "v = 1.0"
)


neuron2 = Neuron(
    equations="""
        v = v + 1.0
    """,
    spike = "v == 3.0",
    reset = "v = 1.0 ",
    refractory = 3.0
)

pop1 = Population(3, neuron1)
pop2 = Population(3, neuron2)

m = Monitor(pop1, 'v')
n = Monitor(pop2, 'v')

class test_SpikingNeuron(unittest.TestCase):
    """
    This class tests the functionality of neurons with a defined *spike* condition.
    The functionality of the optional *refractory* period is also tested.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([pop1, pop2, m, n])
        self.test_net.compile(silent=True)

    def setUp(self):
        """
        In our *setUp()* method we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def test_v(self):
        """
        After every time step we check if the evolution of the variable *v* fits the defined conditions of the neuron.
        """
        self.assertTrue(numpy.allclose(self.test_net.get(pop1).neuron(0).v, 0.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop1).neuron(0).v, 1.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop1).neuron(0).v, 2.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop1).neuron(0).v, 1.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop1).neuron(0).v, 2.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop1).neuron(0).v, 1.0))

    def test_v_ref(self):
        """
        After every time step we check if the evolution of the variable *v* fits the defined conditions of the neuron, which also contain the optional *refractory* period.
        """
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 0.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 1.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 2.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 1.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 1.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 1.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 1.0))
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.test_net.get(pop2).neuron(0).v, 2.0))

