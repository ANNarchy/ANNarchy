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

from ANNarchy import *
from ANNarchy.intern.ConfigManagement import _check_paradigm

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

        tc_loc_pop = Population(3, LocalEquation)
        tc_glob_pop = Population(3, GlobalEquation)

        cls.test_net = Network()
        cls.test_net.add([tc_loc_pop, tc_glob_pop])
        cls.test_net.compile(silent=True)
        cls.test_net.set_seed(seed=1)

        cls.net_local_pop = cls.test_net.get(tc_loc_pop)
        cls.net_global_pop = cls.test_net.get(tc_glob_pop)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net
        clear()

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self.test_net.reset()

    def test_LocalNeuronRandomVariable(self):
        """
        Test if the local Uniform distribution yields non-homogeneous values in
        the range 0 < x <= 1.
        """
        self.test_net.simulate(1)

        r = self.net_local_pop.r
        np.testing.assert_allclose(r, 0.5, atol=0.5)  # 0<=x<=1
        self.assertFalse(np.allclose(r, 0.0))  # x != 0
        self.assertFalse(np.max(r) == np.min(r))  # different values

    def test_GlobalNeuronRandomVariable(self):
        """
        Test if the global Uniform distribution yields values between 0 and 1.
        """
        self.test_net.simulate(1)

        r = self.net_global_pop.r
        np.testing.assert_allclose(r, 0.5, atol=0.5)  # 0<=x<=1
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

        v = Population(geometry=5, neuron=LeakyIntegrator())

        proj = Projection(pre=v, post=v, target="exc", synapse=my_synapse)
        proj.connect_fixed_number_pre(number = 1, weights=0.0)

        cls.test_net = Network()
        cls.test_net.add([v, proj])
        cls.test_net.compile(silent=True)
        cls.test_net.set_seed(seed=1)

        cls.test_proj = cls.test_net.get(proj)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net
        clear()

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self.test_net.reset()

    def test_LocalSynapseRandomVariable(self):
        """
        Test if the Normal distribution in the Synapse definition yields
        reasonable (within 10 sigma, non-homogeneous) values.
        """
        self.test_net.simulate(1)

        w = self.test_proj.w
        np.testing.assert_allclose(w, 1.0, atol=0.01)  # -0.99<=x<=1.01
        self.assertFalse(np.max(w) == np.min(w))
