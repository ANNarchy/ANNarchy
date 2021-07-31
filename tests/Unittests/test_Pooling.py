"""
    test_Pooling.py

    This file is part of ANNarchy.

    Copyright (C) 2021 Alex Schwarz <alex.schwarz@informatik.tu-chemnitz.de>
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
import unittest
import numpy

from ANNarchy import Neuron, Population, compile, simulate, setup # , Network
from ANNarchy.extensions.convolution import Pooling


class test_Pooling(unittest.TestCase):
    """
    Tests the functionality of the *Projection* object. We test:

        *access to parameters
        *method to get the ranks of post-synaptic neurons recieving synapses
        *method to get the number of post-synaptic neurons recieving synapses
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        TODO! copy of Pooling is not possible
        """
        # setup(structural_plasticity=False)
        neuron = Neuron(parameters="baseline = 0", equations="r = baseline")
        neuron2 = Neuron(equations="r = sum(exc) : init = 0.0")

        self.pop1 = Population((2, 3, 10), neuron)
        self.pop2 = Population((2, 3), neuron2)
        self.pop3 = Population((1, 1, 10), neuron2)

        self.proj1 = Pooling(pre=self.pop1, post=self.pop2, target="exc", operation='mean')
        self.proj1.connect_pooling(extent=(1, 1, 10))

        self.proj2 = Pooling(pre=self.pop1, post=self.pop3, target="exc", operation='mean')
        self.proj2.connect_pooling(extent=(2, 3, 1))

        compile()

        baseline = numpy.arange(0.0, 6.0, 0.1)
        baseline = numpy.moveaxis(numpy.reshape(baseline, (10, 2, 3)), 0, -1)
        self.pop1.baseline = baseline
        simulate(2)

    def setUp(self):
        """
        In our *setUp()* function we reset the network before every test.
        """
        # self.test_net.reset()
        pass

    def test_get_size(self):
        """
        Tests the *size* method, which returns the number of post-synaptic neurons recieving synapses.
        """
        self.assertEqual(self.proj1.size, 6)

    def test_get_post_ranks(self):
        """
        Tests the *post_ranks* method, which returns the ranks of post-synaptic neurons recieving synapses.
        """
        self.assertEqual(self.proj1.post_ranks, list(range(6)))

    def test_pool(self):
        """
        Tests if after pooling the last dimension, the rates in the post projection are as expected
        """
        comb = numpy.array([[2.7, 2.8, 2.9],
                            [3.0, 3.1, 3.2]])
        numpy.testing.assert_allclose(self.pop2.get('r'), comb)

    def test_pool2(self):
        """
        Tests if after pooling the first two dimensions, the rates in the post
        projection are as expected
        """
        comb = numpy.array([0.25 + 0.6 * i for i in range(10)])
        numpy.testing.assert_allclose(self.pop3.get('r').flatten(), comb)
