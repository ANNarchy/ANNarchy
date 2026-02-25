"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest

from conftest import TARGET_FOLDER
from ANNarchy import LeakyIntegrator, Network, Synapse


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
        value_synapse = Synapse(
            parameters="""
                tau_utility = 1000
            """,
            equations="""
                tau_utility * dutility/dt = pre.r * post.r : init=1.0
            """,
            creating="pre.r * post.r > 0.9 : proba = 1.0, w = 1.0",
            pruning="utility < 0.0 : proba = 0.5",
            operation="max",
        )

        cls.test_net = Network()
        cls.test_net.config(structural_plasticity=True)

        v = cls.test_net.create(geometry=5, neuron=LeakyIntegrator())

        value_proj = cls.test_net.connect(
            pre=v, post=v, target="exc", synapse=value_synapse
        )
        value_proj.fixed_number_pre(number=1, weights=1.0)

        # build the network
        cls.test_net.compile(silent=True, directory=TARGET_FOLDER)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset()

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
        value_synapse = Synapse(
            parameters="""
                tau_utility = 1000
            """,
            equations="""
                tau_utility * dutility/dt = pre.r * post.r : init=1.0
            """,
            creating="pre.r * post.r > 0.9 : proba = 1.0, w = 1.0",
            pruning="utility < 0.0 : proba = 0.5",
            operation="max",
        )

        cls.test_net = Network()
        cls.test_net.config(structural_plasticity=True)

        v = cls.test_net.create(geometry=5, neuron=LeakyIntegrator())

        value_proj = cls.test_net.connect(
            pre=v, post=v, target="exc", synapse=value_synapse
        )
        value_proj.fixed_number_pre(number=1, weights=1.0, delays=2.0)

        # build the network
        cls.test_net.compile(silent=True, directory=TARGET_FOLDER)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def test_invoke_compile(self):
        """
        This test case just invoke the compilation process for the above
        defined network definition.
        """
        pass
