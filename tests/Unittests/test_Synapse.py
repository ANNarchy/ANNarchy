"""

    test_SpikingNeuron.py

    This file is part of ANNarchy.

    Copyright (C) 2018-2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
import numpy

from ANNarchy import *

class test_Locality(unittest.TestCase):
    """
    ANNarchy support several three different localities for variables/parameters:
    synaptic, postsynaptic, projection. This test should verify, that these keywords
    does not lead to compiler errors.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r = 0
            """
        )

        syn = Synapse(
            parameters="""
                a = 0.1
                b = 0.1 : postsynaptic
                c = 0.1 : projection
            """
        )

        pre = Population(3, neuron)
        post = Population(1, neuron)
        proj = Projection(pre, post, "exc", synapse = syn).connect_all_to_all(weights=1.0)

        self.test_net = Network(True)

    def test_compile(self):
        """
        Tests if the network description is compilable.
        """
        self.test_net.compile(silent=True)

class test_AccessPSP(unittest.TestCase):
    """
    In this setup we test, if the access to post-synaptic potential, more detailed the statements
    pre.sum(exc) or post.sum(exc), is correctly implemented.

    Other statements like mean(pre.r) are covered by test_GlobalOperation.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations="""
                r = sum(exc)
            """
        )

        syn = Synapse(
            equations="""
                w = pre.sum(exc) + post.sum(exc)
            """
        )

        pre = Population(1, neuron)
        post = Population(1, neuron)

        # to have an "exc" target in pre, we need to create forward and backward connection
        proj = Projection(pre, post, "exc", synapse = syn).connect_one_to_one(weights=1.0)
        proj = Projection(post, pre, "exc", synapse = syn).connect_one_to_one(weights=1.0)

        self.test_net = Network(True)

    def test_compile(self):
        """
        Tests if the network description is compilable.
        """
        self.test_net.compile(silent=True)

class test_ModifiedPSP(unittest.TestCase):
    """
    Test modified psp statements
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations="""
                r = sum(exc)
            """
        )

        ReversedSynapse = Synapse(
            parameters="""
                reversal = 1.0
            """,
            psp="""
                w*pos(reversal-pre.r)
            """
        )

        pre = Population(1, neuron)
        post = Population(1, neuron)

        # to have an "exc" target in pre, we need to create forward and backward connection
        proj = Projection(pre, post, "exc", synapse = ReversedSynapse).connect_one_to_one(weights=1.0)

        self.test_net = Network(True)

    def test_compile(self):
        """
        Tests if the network description is compilable.
        """
        self.test_net.compile(silent=True)
