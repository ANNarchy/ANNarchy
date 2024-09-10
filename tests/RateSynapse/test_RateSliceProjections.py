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


class test_RateSliceProjections(unittest.TestCase):
    """
    Test Projections for differently sliced rate coded PopulationViews.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        preNeuron = Neuron(parameters="r=0.0")
        rNeuron = Neuron(equations="""
            p = sum(prev)
            q = sum(postv)
            r = sum(bothv)
                         """)

        pop1 = Population((7, 7), neuron=preNeuron)
        pop2 = Population((7, 7), neuron=rNeuron)
        pop1_view = pop1[1:3, 1:4] + pop1[5, 6]
        pop2_view = pop1[1:3, 1:4] + pop1[5, 6]

        # Connection with a presynaptic PopulationView
        pre_view = Projection(pop1_view, pop2, target="prev")
        pre_view.connect_all_to_all(0.1)

        # Connection with a postsynaptic PopulationView
        post_view = Projection(pop1, pop2_view, target="postv")
        post_view.connect_all_to_all(0.1)

        # Connection with a pre- and postsynaptic PopulationView
        both_view = Projection(pop1_view, pop2_view, target="bothv")
        both_view.connect_all_to_all(0.1)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, pre_view, post_view, both_view])
        cls.test_net.compile(silent=True)

        cls.pop1 = cls.test_net.get(pop1)
        cls.pop2 = cls.test_net.get(pop2)
        cls.prev = cls.test_net.get(pre_view)
        cls.post = cls.test_net.get(post_view)
        cls.both = cls.test_net.get(both_view)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset(populations=True, projections=True)
        self.test_net.disable_learning()

    def test_compile(self):
        """
        Test Compile.
        """
        pass

    def test_pre_view(self):
        """
        Test the projection with pre-PopulationView.
        """
        self.pop1.r = numpy.full((7, 7), numpy.arange(7))
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.pop2.sum("p"), 0.1)

    def test_post_view(self):
        """
        Test the projection with post-PopulationView.
        """
        self.pop1.r = numpy.full((7, 7), numpy.arange(7))
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.pop2.sum("q"), 0.1)

    def test_both_view(self):
        """
        Test the projection with pre- and post-PopulationView.
        """
        self.pop1.r = numpy.full((7, 7), numpy.arange(7))
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.pop2.sum("r"), 0.1)
