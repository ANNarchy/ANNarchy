"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Population, Neuron, Network, Projection


class test_SpikingPopulationView(unittest.TestCase):
    """
    Test Projections for differently sliced spiking PopulationViews.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        LIF = Neuron(
            parameters="""
                tau = 10.0  : population
                Er = -60.0  : population
                Ee = 0.0    : population
                T = -45.0   : population
            """,
            equations="""
                tau * dv/dt = (Er - v) + (g_STP + g_STDP) * (Ee - v) : init = 0.0
            """,
            spike="v > T",
            reset="v = Er",
            refractory=5.0,
        )

        inp = Population((5, 5), neuron=LIF)
        out = Population((5, 5), neuron=LIF)

        cls.spikeModels = ["STP", "STDP"]

        STP = Projection(inp, out, target="STP")
        STP.all_to_all(0.1)

        STDP = Projection(inp, out, target="STDP")
        STDP.all_to_all(0.1)

        cls.test_net = Network()
        cls.test_net.add([inp, out, STP, STDP])
        cls.test_net.compile(silent=True, directory=TARGET_FOLDER)

        cls.inp = cls.test_net.get(inp)
        cls.out = cls.test_net.get(out)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset(populations=True, projections=True)
        self.inp.r = 1.0

    def test_compile(self):
        """
        Test Compile.
        """
        pass

    def test_get_exc_STP(self):
        """
        Test the value of the STP projection.
        """
        self.test_net.simulate(2)
        numpy.testing.assert_allclose(self.out.sum("STP"), 0.05)

    def test_get_exc_STDP(self):
        """
        Test the value of the STDP projection.
        """
        self.test_net.simulate(2)
        numpy.testing.assert_allclose(self.out.sum("STDP"), 0.1)
