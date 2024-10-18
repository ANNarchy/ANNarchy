#==============================================================================
#
#     test_RateDefaultSynapseModels.py
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

from ANNarchy import Population, Neuron, Network, Projection, models

class test_RateDefaultSynapseModels(unittest.TestCase):
    """
    Test the predefined default Synapse types. Just by compiling.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        rNeuron = Neuron(equations="r=sum(Hebb)+sum(Oja)+sum(IBCM)")

        inp = Population(1, neuron=rNeuron)
        out = Population(1, neuron=rNeuron)

        Hebb = Projection(inp, out, synapse=models.Hebb, target="Hebb")
        Hebb.connect_all_to_all(weights=0.1, storage_format=cls.storage_format, storage_order=cls.storage_order, force_multiple_weights=True)

        Oja = Projection(inp, out, synapse=models.Oja, target="Oja")
        Oja.connect_all_to_all(weights=0.1, storage_format=cls.storage_format, storage_order=cls.storage_order, force_multiple_weights=True)

        IBCM = Projection(inp, out, synapse=models.IBCM, target="IBCM")
        IBCM.connect_all_to_all(weights=0.1, storage_format=cls.storage_format, storage_order=cls.storage_order, force_multiple_weights=True)

        cls.test_net = Network()
        cls.test_net.add([inp, out, Hebb, Oja, IBCM])
        cls.test_net.compile(silent=True)

        cls.inp = cls.test_net.get(inp)
        cls.out = cls.test_net.get(out)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset(populations=True, projections=True)
        self.test_net.disable_learning()
        self.inp.r = 1.0

    def test_compile(self):
        """
        Test Compile.
        """
        pass

    def test_get_exc_Hebb(self):
        """
        Test the value of the hebb projection.
        """
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.out.sum("Hebb"), 0.1)

    def test_get_exc_Oja(self):
        """
        Test the value of the Oja projection.
        """
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.out.sum("Oja"), 0.1)

    def test_get_exc_IBCM(self):
        """
        Test the value of the IBCM projection.
        """
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.out.sum("IBCM"), 0.1)

    def test_list_standard_synapses(self):
        """
        Test if the standard synapses are all listed and tested.
        (just tested once, not in Spiking class)
        """
        synapses = [m.__name__ for m in models.list_standard_synapses()]
        self.assertEqual(synapses, ["STP", "STDP", "Hebb", "Oja", "IBCM"])
