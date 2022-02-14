"""

    test_SpikingSynapse.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2022 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
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
import numpy

from ANNarchy import Neuron, Population, Synapse, Projection, Network

class test_PreSpike():
    """
    This class tests the functionality of neurons with a defined *pre-spike*
    equations.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        # Each time step emit an event
        SpkNeuron = Neuron(
            parameters = "v = 1",
            equations = "mp = g_exc",
            spike = "v > 0"
        )
        pop = Population(3, neuron=SpkNeuron)

        # decrease weight until zero.
        BoundedSynapse = Synapse(
            parameters="g = 1.0",
            pre_spike="""
                w -= 1.0 : min=0.0
            """,
            psp="g"
        )
        proj = Projection( pop, pop, "exc", synapse = BoundedSynapse)
        proj.connect_all_to_all(5.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])
        cls.test_net.compile(silent=True)

        cls.test_proj = cls.test_net.get(proj)

    def setUp(self):
        """
        In our *setUp()* method we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def test_w(self):
        """
        Test if the weight value is decreased and check the boundary.
        """
        # is correctly initialized
        self.assertTrue(numpy.allclose(self.test_proj.dendrite(0).w, [5.0, 5.0]))

        # w should has decreased
        self.test_net.simulate(5)
        self.assertTrue(numpy.allclose(self.test_proj.dendrite(0).w, [1.0, 1.0]))

        # w should not decrease further
        self.test_net.simulate(5)
        self.assertTrue(numpy.allclose(self.test_proj.dendrite(0).w, [0.0, 0.0]))

class test_PostSpike():
    """
    This class tests the functionality of neurons with a defined
    *post_spike* equations.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        # Each time step emit an event
        SpkNeuron = Neuron(
            parameters = "v = 1",
            equations = "mp = g_exc",
            spike = "v > 0"
        )
        pop = Population(3, neuron=SpkNeuron)

        # increase weight towards a limit
        BoundedSynapse = Synapse(
            parameters="""
                constant = 1.0
            """,
            post_spike="""
                w += 1.0 * constant : max=10.0
            """
        )
        pop = Population(3, neuron=SpkNeuron)
        proj = Projection( pop, pop, "exc", synapse = BoundedSynapse)
        proj.connect_all_to_all(5.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])
        cls.test_net.compile(silent=True)

        cls.test_proj = cls.test_net.get(proj)

    def setUp(self):
        """
        In our *setUp()* method we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def test_w(self):
        """
        Test if the weight value is decreased and check the boundary.
        """
        # is correctly initialized
        self.assertTrue(numpy.allclose(self.test_proj.dendrite(0).w, [5.0, 5.0]))

        # w should has increased
        self.test_net.simulate(5)
        self.assertTrue(numpy.allclose(self.test_proj.dendrite(0).w, [10.0, 10.0]))

        # w should not increase further
        self.test_net.simulate(5)
        self.assertTrue(numpy.allclose(self.test_proj.dendrite(0).w, [10.0, 10.0]))
