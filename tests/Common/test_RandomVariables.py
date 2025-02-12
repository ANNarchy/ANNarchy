"""

    test_RandomVariables.py

    This file is part of ANNarchy.

    Copyright (C) 2016-2022 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
    2022 Alex Schwarz <alex.schwarz@informatik.tu-chemnitz.de>

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

from ANNarchy import Network, Neuron, Synapse, LeakyIntegrator

class test_NeuronRandomVariables(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        LocalEquation = Neuron(
            equations = """
                r = Uniform(0,1)
            """
        )

        GlobalEquation = Neuron(
            parameters = "",
            equations = """
                r = Uniform(0,1) : population
            """
        )

        cls._network = Network(seed=1)

        cls._tc_loc_pop = cls._network.create(3, LocalEquation)
        cls._tc_glob_pop = cls._network.create(3, GlobalEquation)

        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()

    def test_LocalNeuronRandomVariable(self):
        """
        Test if the local Uniform distribution yields non-homogeneous values in
        the range 0 < x <= 1.
        """
        self._network.simulate(1)

        r = self._tc_loc_pop.r
        numpy.testing.assert_allclose(r, 0.5, atol=0.5)  # 0<=x<=1
        self.assertFalse(numpy.allclose(r, 0.0))  # x != 0
        self.assertFalse(numpy.max(r) == numpy.min(r))  # different values

    def test_GlobalNeuronRandomVariable(self):
        """
        Test if the global Uniform distribution yields values between 0 and 1.
        """
        self._network.simulate(1)

        r = self._tc_glob_pop.r
        numpy.testing.assert_allclose(r, 0.5, atol=0.5)  # 0<=x<=1
        self.assertNotAlmostEqual(r, 0.0)  # x != 0


class test_SynapseRandomVariables(unittest.TestCase):
    """
    Test the usage of random distributions for synapse variables
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        my_synapse = Synapse(
            equations="""
                w = Normal(1.0, 0.001)
            """
        )

        cls._network = Network(seed=1)

        v = cls._network.create(geometry=5, neuron=LeakyIntegrator())
        cls._test_proj = cls._network.connect(pre=v, post=v, target="exc", synapse=my_synapse)
        cls._test_proj.connect_fixed_number_pre(number = 1, weights=0.0)

        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()

    def test_LocalSynapseRandomVariable(self):
        """
        Test if the Normal distribution in the Synapse definition yields
        reasonable (within 10 sigma, non-homogeneous) values.
        """
        self._network.simulate(1)

        w = self._test_proj.w
        numpy.testing.assert_allclose(w, 1.0, atol=0.01)  # -0.99<=x<=1.01
        self.assertFalse(numpy.max(w) == numpy.min(w))
