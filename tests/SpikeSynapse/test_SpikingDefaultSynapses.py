#==============================================================================
#
#     test_DefaultSynapses.py
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
#     but WITH_OUT ANY WARRANTY; with_out even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#==============================================================================
import unittest
import numpy

from ANNarchy import Neuron, Network, models


class test_SpikingDefaultSynapses(unittest.TestCase):
    """
    Test the predefined default Synapse types. Just by compiling.
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
            spike = "v > T",
            reset = "v = Er",
            refractory = 5.0
        )

        cls._network = Network()

        cls._inp = cls._network.create(geometry=1, neuron=LIF)
        cls._out = cls._network.create(geometry=1, neuron=LIF)

        cls.spikeModels = ["STP", "STDP"]

        STP = cls._network.connect(cls._inp, cls._out, synapse=models.STP, target="STP")
        STP.all_to_all(
            weights=0.1,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        STDP = cls._network.connect(cls._inp, cls._out, synapse=models.STDP, target="STDP")
        STDP.all_to_all(
            weights=0.1,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self._network.reset(populations=True, projections=True)

    def test_compile(self):
        """
        Test Compile.
        """
        pass
