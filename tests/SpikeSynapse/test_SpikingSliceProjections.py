#==============================================================================
#
#     test_SliceProjections.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2022  Alex Schwarz <alex.schwarz@informatik.tu-chemnitz.de>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#==============================================================================
import unittest
import numpy

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
            refractory=5.0
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
        cls.test_net.compile(silent=True)

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
