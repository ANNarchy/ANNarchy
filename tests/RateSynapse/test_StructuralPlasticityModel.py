"""

    test_StructuralPlasticityModel.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2023 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
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

from ANNarchy import LeakyIntegrator, Network, setup, Synapse


class test_StructuralPlasticityModel(unittest.TestCase):
    """
    This class tests the *Structural Plasticity* feature, which can optinally
    be enabled.

    In the synapse description an user can define prune and create conditions.
    This unittest was inspired by an submitted issue #41 (on bitbucket.org).
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test, this was an example defintion
        provided by the reporter.
        """
        setup(structural_plasticity=True)

        value_synapse = Synapse(
            parameters="""
                tau_utility = 1000
            """,
            equations="""
                tau_utility * dutility/dt = pre.r * post.r : init=1.0
            """,
            creating="pre.r * post.r > 0.9 : proba = 1.0, w = 1.0",
            pruning="utility < 0.0 : proba = 0.5",
            operation="max"
        )

        cls.test_net = Network()

        v = cls.test_net.create(geometry=5, neuron=LeakyIntegrator())

        value_proj = cls.test_net.connect(pre=v, post=v, target="exc", synapse=value_synapse)
        value_proj.connect_fixed_number_pre(number=1, weights=1.0)

        # build the network
        cls.test_net.compile(silent=True)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset()

    @classmethod
    def tearDownClass(cls):
        """
        Remove the structural_plasticity global flag to not interfere with
        further tests.
        """
        setup(structural_plasticity=False)

    def test_invoke_compile(self):
        """
        This test case just invoke the compilation process for the above
        defined network definition.
        """
        pass


class test_StructuralPlasticityModelDelay(unittest.TestCase):
    """
    This class tests the *Structural Plasticity* feature, which can optionally
    be enabled.

    In the synapse description an user can define prune and create conditions.
    This class behaves like test_StructuralPlasticityModel apart from using
    synaptic delays.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test, this was an example defintion
        provided by the reporter.
        """
        setup(structural_plasticity=True)

        value_synapse = Synapse(
            parameters="""
                tau_utility = 1000
            """,
            equations="""
                tau_utility * dutility/dt = pre.r * post.r : init=1.0
            """,
            creating="pre.r * post.r > 0.9 : proba = 1.0, w = 1.0",
            pruning="utility < 0.0 : proba = 0.5",
            operation="max"
        )

        cls.test_net = Network()

        v = cls.test_net.create(geometry=5, neuron=LeakyIntegrator())

        value_proj = cls.test_net.connect(pre=v, post=v, target="exc", synapse=value_synapse)
        value_proj.connect_fixed_number_pre(number=1, weights=1.0, delays=2.0)

        # build the network
        cls.test_net.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        Remove the structural_plasticity global flag to not interfere with
        further tests.
        """
        setup(structural_plasticity=False)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        setup(structural_plasticity=True)
        self.test_net.reset()

    def test_invoke_compile(self):
        """
        This test case just invoke the compilation process for the above
        defined network definition.
        """
        pass
