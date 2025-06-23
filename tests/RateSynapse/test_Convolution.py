"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
import numpy

from ANNarchy import Network, Neuron
from ANNarchy.extensions.convolution import Convolution


conv_filter = numpy.array([[-1., 0.0, 1.], [-1., 0.1, 1.], [-1., 0.5, 1.]])
bo_filters = numpy.array([[[[1, 0, 0], [0,  1,  1], [0, 0, 1]],
                           [[0, 1, 0], [0,  1,  1], [0, 1, 0]]],
                          [[[1, 0, 0], [0, -1, -1], [0, 0, 1]],
                           [[0, 0, 1], [0,  1,  1], [1, 0, 0]]]])
bo_filters = numpy.moveaxis(bo_filters, 1, -1)


class test_Convolution(unittest.TestCase):
    """
    Tests the functionality of the *Projection* object. We test:

        *access to parameters
        *method to get the ranks of post-synaptic neurons recieving synapses
        *method to get the number of post-synaptic neurons recieving synapses
    """
    @classmethod
    def setUpClass(cls):
        neuron = Neuron(
            parameters="baseline = 0",
            equations="r = baseline"
        )

        neuron2 = Neuron(
            equations="""
                r = sum(exc) : init = 0.0
                rd = sum(exc_delay) : init = 0.0
                mr = sum(mex) : init = 0.0
                pr = sum(pex) : init = 0.0
                br = sum(bex) : init = 0.0
            """
        )

        cls.test_net = Network()

        cls.pop0 = cls.test_net.create(geometry=(3, 4), neuron=neuron)
        cls.pop1 = cls.test_net.create(geometry=(3, 4), neuron=neuron2)
        cls.pop2 = cls.test_net.create(geometry=(3, 4, 2), neuron=neuron)
        cls.pop3 = cls.test_net.create(geometry=(3, 4, 2), neuron=neuron2)

        cls.proj0 = cls.test_net.connect(Convolution(pre=cls.pop0, post=cls.pop1, target="exc"))
        cls.proj0.connect_filter(conv_filter)

        proj1 = cls.test_net.connect(Convolution(pre=cls.pop0, post=cls.pop1, target="mex", operation="max"))
        proj1.connect_filter(conv_filter)

        proj2 = cls.test_net.connect(Convolution(pre=cls.pop0, post=cls.pop1, target="pex",
                                     psp="w * pre.r * pre.r"))
        proj2.connect_filter(conv_filter)

        proj3 = cls.test_net.connect(Convolution(pre=cls.pop2, post=cls.pop1, target="bex"))
        proj3.connect_filter(bo_filters[0, :, :, :])

        proj4 = cls.test_net.connect(Convolution(pre=cls.pop2, post=cls.pop3, target="pex"))
        proj4.connect_filter(conv_filter, keep_last_dimension=True)

        ssList = [[i, j, 0] for i in range(3) for j in range(4)]
        proj5 = cls.test_net.connect(Convolution(pre=cls.pop2, post=cls.pop3, target="exc"))
        proj5.connect_filters(bo_filters, padding=0.0, subsampling=ssList)

        proj6 = cls.test_net.connect(Convolution(pre=cls.pop0, post=cls.pop1, target="exc_delay"))
        proj6.connect_filter(conv_filter, delays=3.0)

        cls.test_net.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net

    def setUp(self):
        """
        In our *setUp()* function we reset the network before every test.
        """
        self.test_net.reset()
        
        # set the input
        baseline = numpy.reshape(numpy.arange(0.0, 1.2, 0.1), (3, 4))
        baseline2 = numpy.moveaxis(numpy.array([baseline, baseline + 2]), 0, 2)
        self.pop0.baseline = baseline
        self.pop2.baseline = baseline2
        
        # ensure that the psp is updated
        self.test_net.simulate(2)

    def test_get_weights(self):
        """
        Tests the access to the parameter *weights* of our *Projection* with
        the *get()* method.
        """
        numpy.testing.assert_allclose(self.proj0.get('weights'), conv_filter)

    def test_get_size(self):
        """
        Tests the *size* method, which returns the number of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self.proj0.size, 12)

    def test_get_post_ranks(self):
        """
        Tests the *post_ranks* method, which returns the ranks of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self.proj0.post_ranks, list(range(12)))

    def test_post_simple_convolution(self):
        """
        Tests if the rates after convolution in the post projection are as
        expected
        """
        comb = numpy.array([[0.8, 0.66, 0.72, -0.42],
                            [1.94, 1.1, 1.16, -1.18],
                            [1.48, 0.49, 0.5, -1.49]])
        numpy.testing.assert_allclose(self.pop1.get('r'), comb)

    def test_post_simple_convolution(self):
        """
        Tests if the rates after convolution in the post projection are as
        expected when a delay is used.
        """
        # After 2 ms all should be zero (delay is 3 ms)
        comb = numpy.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]])
        
        numpy.testing.assert_allclose(self.pop1.get('rd'), comb)

        # Simulate another 2 ms and verify the output
        self.test_net.simulate(2)
        comb = numpy.array([[0.8, 0.66, 0.72, -0.42],
                            [1.94, 1.1, 1.16, -1.18],
                            [1.48, 0.49, 0.5, -1.49]])
        
        numpy.testing.assert_allclose(self.pop1.get('rd'), comb)

    def test_post_max_convolution(self):
        """
        Tests if the rates after convolution in the post projection are as
        expected
        """
        comb = numpy.array([[0.5, 0.6, 0.7, 0.35],
                            [0.9, 1.0, 1.1, 0.55],
                            [0.9, 1.0, 1.1, 0.11]])
        numpy.testing.assert_allclose(self.pop1.get('mr'), comb)

    def test_non_linear_conv(self):
        """
        Tests the convolution with a different psp
        """
        comb = numpy.array([[0.34, 0.366, 0.504, -0.146],
                            [1.406, 1.03, 1.256, -0.746],
                            [1.124, 0.641, 0.74, -1.239]])
        numpy.testing.assert_allclose(self.pop1.get('pr'), comb)

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
        pr = numpy.rollaxis(self.pop3.get('pr'), 2)
        numpy.testing.assert_allclose(pr, comb)

    def test_simple_bank_of_filters(self):
        """
        Test the bank of filters.
        """
        comb = numpy.array([[7.1, 7.7, 8.3, 5.3],
                            [11.5, 12.2, 13.0, 9.0],
                            [9.8, 10.7, 11.3, 7.5]])
        numpy.testing.assert_allclose(self.pop1.get('br'), comb)

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
        r = numpy.rollaxis(self.pop3.get('r'), 2)
        numpy.testing.assert_allclose(r, comb)


if __name__ == "__main__":
    unittest.main()
