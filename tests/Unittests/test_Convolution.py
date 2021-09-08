"""
    test_Convoluvion.py

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

from ANNarchy import Neuron, Population, compile, simulate, setup,Network
from ANNarchy.extensions.convolution import Convolution


conv_filter = numpy.array([[-1., 0.0, 1.],
                           [-1., 0.1, 1.],
                           [-1., 0.5, 1.]])
bo_filters = numpy.array([[[[1, 0, 0],
                            [0, 1, 1],
                            [0, 0, 1]],
                           [[0, 1, 0],
                            [0, 1, 1],
                            [0, 1, 0]]],
                          [[[1, 0, 0],
                            [0, -1, -1],
                            [0, 0, 1]],
                           [[0, 0, 1],
                            [0, 1, 1],
                            [1, 0, 0]]]])
bo_filters = numpy.moveaxis(bo_filters, 1, -1)


neuron = Neuron(
    parameters="baseline = 0",
    equations="r = baseline"
)

neuron2 = Neuron(
    equations="""
        r = sum(exc) : init = 0.0
        mr = sum(mex) : init = 0.0
        pr = sum(pex) : init = 0.0
        br = sum(bex) : init = 0.0
    """
)

pop1 = Population((3, 4), neuron)
pop2 = Population((3, 4), neuron2)
pop3 = Population((3, 4, 2), neuron)
pop4 = Population((3, 4, 2), neuron2)

proj1 = Convolution(pre=pop1, post=pop2, target="exc")
proj1.connect_filter(conv_filter)

proj2 = Convolution(pre=pop1, post=pop2, target="mex",
                        operation="max")
proj2.connect_filter(conv_filter)

proj3 = Convolution(pre=pop1, post=pop2, target="pex",
                        psp="w * pre.r * pre.r")
proj3.connect_filter(conv_filter)

proj4 = Convolution(pre=pop3, post=pop2, target="bex")
proj4.connect_filter(bo_filters[0, :, :, :])

proj5 = Convolution(pre=pop3, post=pop4, target="pex")
proj5.connect_filter(conv_filter, keep_last_dimension=True)

ssList = [[i, j, 0] for i in range(3) for j in range(4)]
proj6 = Convolution(pre=pop3, post=pop4, target="exc")
proj6.connect_filters(bo_filters, padding=0.0, subsampling=ssList)

compile()

class test_Convolution(unittest.TestCase):
    """
    Tests the functionality of the *Projection* object. We test:

        *access to parameters
        *method to get the ranks of post-synaptic neurons recieving synapses
        *method to get the number of post-synaptic neurons recieving synapses
    """
    def setUp(self):
        """
        In our *setUp()* function we reset the network before every test.
        """
        baseline = numpy.reshape(numpy.arange(0.0, 1.2, 0.1), (3, 4))
        baseline2 = numpy.moveaxis(numpy.array([baseline, baseline + 2]), 0, 2)
        pop1.baseline = baseline
        pop3.baseline = baseline2
        simulate(2)
        # self.test_net.reset()

    def test_get_weights(self):
        """
        Tests the access to the parameter *weights* of our *Projection* with the *get()* method.
        """
        self.assertTrue(numpy.allclose(self.proj1.get('weights'), conv_filter))

    def test_get_size(self):
        """
        Tests the *size* method, which returns the number of post-synaptic neurons recieving synapses.
        """
        self.assertEqual(self.proj1.size, 12)

    def test_get_post_ranks(self):
        """
        Tests the *post_ranks* method, which returns the ranks of post-synaptic neurons recieving synapses.
        """
        self.assertEqual(self.proj1.post_ranks, list(range(12)))

    def test_post_simple_convolution(self):
        """
        Tests if the rates after convolution in the post projection are as expected
        """
        comb = numpy.array([[0.8, 0.66, 0.72, -0.42],
                            [1.94, 1.1, 1.16, -1.18],
                            [1.48, 0.49, 0.5, -1.49]])
        self.assertTrue(numpy.allclose(self.pop2.get('r'), comb))

    def test_post_max_convolution(self):
        """
        Tests if the rates after convolution in the post projection are as expected
        """
        comb = numpy.array([[0.5, 0.6, 0.7, 0.35],
                            [0.9, 1.0, 1.1, 0.55],
                            [0.9, 1.0, 1.1, 0.11]])
        self.assertTrue(numpy.allclose(self.pop2.get('mr'), comb))

    def test_non_linear_conv(self):
        """
        Tests the convolution with a different psp
        """
        comb = numpy.array([[0.34, 0.366, 0.504, -0.146],
                            [1.406, 1.03, 1.256, -0.746],
                            [1.124, 0.641, 0.74, -1.239]])
        self.assertTrue(numpy.allclose(self.pop2.get('pr'), comb))

    def test_layer_wise_conv(self):
        """
        Test a layer-wise convolution
        """
        comb = numpy.array([[[0.8, 0.66, 0.72, -0.42],
                             [1.94, 1.1, 1.16, -1.18],
                             [1.48, 0.49, 0.5, -1.49]],
                            [[6.0, 1.86, 1.92, -3.22],
                             [9.14, 2.3, 2.36, -5.98],
                             [5.68, 0.69, 0.7, -5.29]]])
        numpy.testing.assert_allclose(numpy.rollaxis(self.pop4.get('pr'), 2), comb)

    def test_simple_bank_of_filters(self):
        """
        Test the bank of filters.
        """
        comb = numpy.array([[7.1, 7.7, 8.3, 5.3],
                            [11.5, 12.2, 13.0, 9.0],
                            [9.8, 10.7, 11.3, 7.5]])
        numpy.testing.assert_allclose(self.pop2.get('br'), comb)

    def test_bank_of_filters(self):
        """
        Test a bank of filters using a subsampling list
        """
        comb = numpy.array([[[7.1, 7.7, 8.3, 5.3],
                             [11.5, 12.2, 13.0, 9.0],
                             [9.8, 10.7, 11.3, 7.5]],
                            [[4.5, 7.0, 7.2, 4.6],
                             [7.0, 10.0, 10.4, 5.2],
                             [6.5, 7.0, 7.2, 2.6]]])
        numpy.testing.assert_allclose(numpy.rollaxis(self.pop4.get('r'), 2), comb)
