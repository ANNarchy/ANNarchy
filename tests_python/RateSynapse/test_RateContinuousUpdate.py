"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest

from conftest import TARGET_FOLDER
from ANNarchy import Neuron, Synapse, Network


class test_RateCodedContinuousUpdate(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates in synapses.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        simple_neuron = Neuron(parameters="r=1.0")

        eq_set = Synapse(
            equations="""
                glob_var = 0.1 : projection
                semi_glob_var = 0.2 : postsynaptic
                w = t + glob_var + semi_glob_var
            """
        )

        cls._network = Network()

        pop0 = cls._network.create(geometry=3, neuron=simple_neuron)
        pop1 = cls._network.create(geometry=1, neuron=simple_neuron)

        proj = cls._network.connect(pop0, pop1, "exc", eq_set)
        proj.all_to_all(
            weights=0.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()  # network reset

    def test_invoke_compile(self):
        self._network.simulate(1)
