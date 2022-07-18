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

class test_RateCodedContinuousUpdate(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates in synapses.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        simple_neuron = Neuron(
            parameters="r=1.0"
        )

        eq_set = Synapse(
            equations="""
                glob_var = 0.1 : projection
                semi_glob_var = 0.2 : postsynaptic
                w = t + glob_var + semi_glob_var
            """
        )

        pop0 = Population(3, simple_neuron)
        pop1 = Population(1, simple_neuron)

        proj = Projection(pop0, pop1, "exc", eq_set)
        proj.connect_all_to_all(weights=0.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop0, pop1, proj])
        cls.test_net.compile(silent=True)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self.test_net.reset() # network reset

    def test_invoke_compile(self):
        self.test_net.simulate(1)

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
            storage_format="lil", #cls.storage_format,
            storage_order="post_to_pre" #cls.storage_order
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
