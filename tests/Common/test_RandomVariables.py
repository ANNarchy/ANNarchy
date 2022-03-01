"""

    test_RandomVariables.py

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
from ANNarchy.core.Global import _check_paradigm

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

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset() # network reset

    def test_LocalNeuronRandomVariable(self):
        self.test_net.simulate(1)

        if _check_paradigm("openmp"):
            np.testing.assert_allclose(self.net_local_pop.r, [0.99718481, 0.93255736, 0.12812445])
        elif _check_paradigm("cuda"):
            np.testing.assert_allclose(self.net_local_pop.r, [0.72449183, 0.43824338, 0.50516922])
        else:
            raise NotImplementedError

    def test_GlobalNeuronRandomVariable(self):
        self.test_net.simulate(1)

        if _check_paradigm("openmp"):
            np.testing.assert_allclose(self.net_global_pop.r, [0.669746040447])
        elif _check_paradigm("cuda"):
            np.testing.assert_allclose(self.net_global_pop.r, [0.106874590279])
        else:
            raise NotImplementedError

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
                w = Normal(0.0, 0.001)
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

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset() # network reset

    def test_LocalSynapseRandomVariable(self):
        self.test_net.simulate(1)

        if _check_paradigm("openmp"):
            np.testing.assert_allclose(self.test_proj.w, [[-0.0005497461789554497], [-0.001402872709127921], [0.0015827522919751402], [-0.0010451468104420224], [0.0002575935412914901]])
        elif _check_paradigm("cuda"):
            np.testing.assert_allclose(self.test_proj.w, [[0.00042327516097052], [-0.0012390467863954901], [0.000405209302949961], [0.00023072272200176617], [0.0005326660317661457]])
        else:
            raise NotImplementedError
