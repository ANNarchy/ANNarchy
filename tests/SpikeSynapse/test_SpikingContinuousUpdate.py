"""

    test_ContinuousUpdate.py

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
from ANNarchy import Neuron, Synapse, Population, Projection, Network
from ANNarchy.intern.Messages import InvalidConfiguration

class test_SpikingContinuousUpdate(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates in synapses.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        simple_pre_neuron = Neuron(
            equations="mp=t",
            spike="(modulo(t,3))"
        )
        simple_post_neuron = Neuron(
            equations="mp=t",
            spike="(modulo(t,5))"
        )

        eq_set = Synapse(
            pre_spike="pre_trace=pre.mp : init=0.0",
            post_spike="post_trace=post.mp : init=0.0",
            equations="""
                post_trace = post_trace -1 : min=0.0
                pre_trace = pre_trace -1 : min=0.0

                w = post_trace + pre_trace
            """
        )

        pop0 = Population(1, simple_pre_neuron)
        pop1 = Population(1, simple_post_neuron)

        proj = Projection(pop0, pop1, "exc", eq_set)
        proj.connect_all_to_all(
            weights=0.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        cls.test_net = Network()
        cls.test_net.add([pop0, pop1, proj])
        cls.test_net.compile(silent=True)

        cls.test_proj = cls.test_net.get(proj)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self.test_net.reset() # network reset

    def test_invoke_compile(self):
        self.test_net.simulate(1)

class test_ContinuousTransmission(unittest.TestCase):
    """
    By default, spiking neurons use event-driven updates. However, we allow
    also a continuous exchange between neurons.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        # Each time step emit an event
        SpkNeuron = Neuron(
            equations = "v = Uniform(0,1)",
            spike = "v > 0.9"
        )
        pop = Population(3, neuron=SpkNeuron)

        # decrease weight until zero.
        ContSynapse = Synapse(
            psp="post.v - pre.v"
        )
        proj = Projection( pop, pop, "exc", synapse = ContSynapse)
        proj.connect_all_to_all(5.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])

        try:
            cls.test_net.compile(silent=True)
            cls.test_proj = cls.test_net.get(proj)
        except InvalidConfiguration:
            cls.test_net = None

    def setUp(self):
        """
        In our *setUp()* method we call *reset()* to reset the network.
        """
        if self.test_net is not None:
            self.test_net.reset()

    def test_invoke_compile(self):
        """
        Test if the weight value is decreased and check the boundary.
        """
        pass
