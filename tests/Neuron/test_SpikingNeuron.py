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

from ANNarchy import Network, Neuron

class test_SpikingCondition(unittest.TestCase):
    """
    This class tests the functionality of neurons with a defined *spike*
    condition.  The functionality of the optional *refractory* period is also
    tested.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron1 = Neuron(
            equations="""
                v = v + 1.0
            """,
            spike = "v == 3.0",
            reset = "v = 1.0"
        )

        neuron2 = Neuron(
            parameters="Vt = 3.0",
            equations="""
                v = v + 1.0
            """,
            spike = "v == Vt",
            reset = "v = 1.0 ",
            refractory = 3.0
        )

        neuron3 = Neuron(
            parameters="Vt = 3.0 : population",
            equations="""
                v = v + 1.0
            """,
            spike = "v == Vt",
            reset = "v = 1.0 ",
        )

        cls._network = Network()
        cls._population_1 = cls._network.population(geometry=3, neuron=neuron1)
        cls._population_2 = cls._network.population(geometry=3, neuron=neuron2)
        cls._population_3 = cls._network.population(geometry=3, neuron=neuron3)
        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* method we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_v(self):
        """
        After every time step we check if the evolution of the variable *v*
        fits the defined conditions of the neuron.
        """
        numpy.testing.assert_allclose(self._population_1.neuron(0).v, 0.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_1.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_1.neuron(0).v, 2.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_1.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_1.neuron(0).v, 2.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_1.neuron(0).v, 1.0)

    def test_v_ref(self):
        """
        After every time step we check if the evolution of the variable *v*
        fits the defined conditions of the neuron, which also contain the
        optional *refractory* period.
        """
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 0.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 2.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_2.neuron(0).v, 2.0)

    def test_v_conditioned(self):
        """
        After every time step we check if the evolution of the variable *v*
        fits the defined conditions of the neuron, threshold is conditioned
        with a global neuron threshold
        """
        numpy.testing.assert_allclose(self._population_3.neuron(0).v, 0.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_3.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_3.neuron(0).v, 2.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_3.neuron(0).v, 1.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_3.neuron(0).v, 2.0)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._population_3.neuron(0).v, 1.0)

