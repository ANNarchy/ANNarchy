"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
import numpy

from ANNarchy import Neuron, Network
from ANNarchy.extensions.convolution import Pooling


class test_Pooling(unittest.TestCase):
    """
    Tests the functionality of the *Projection* object. We test:

        *access to parameters
        *method to get the ranks of post-synaptic neurons recieving synapses
        *method to get the number of post-synaptic neurons recieving synapses
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="baseline = 0",
            equations="r = baseline"
        )

        neuron2 = Neuron(
            equations="""
                r = sum(exc)        : init = 0.0
                rd = sum(exc_delay) : init = 0.0
            """
        )

        cls.test_net = Network()

        cls.pop1 = cls.test_net.create(geometry=(2, 3, 10), neuron=neuron)
        cls.pop2 = cls.test_net.create(geometry=(2, 3), neuron=neuron2)
        cls.pop3 = cls.test_net.create(geometry=(1, 1, 10), neuron=neuron2)

        cls.proj1 = cls.test_net.connect(
            Pooling(pre=cls.pop1, post=cls.pop2, target="exc", operation='mean')
        )
        cls.proj1.pooling(extent=(1, 1, 10))

        cls.proj2 = cls.test_net.connect(
            Pooling(pre=cls.pop1, post=cls.pop3, target="exc", operation='mean')
        )
        cls.proj2.pooling(extent=(2, 3, 1))

        cls.proj3 = cls.test_net.connect(
            Pooling(pre=cls.pop1, post=cls.pop2, target="exc_delay", operation='mean')
        )
        cls.proj3.pooling(extent=(1, 1, 10), delays=3.0)

        cls.test_net.compile(silent=True)

    def setUp(self):
        """
        In our *setUp()* function we reset the network before every test.
        """
        self.test_net.reset()

        baseline = numpy.arange(0.0, 6.0, 0.1)
        baseline = numpy.moveaxis(numpy.reshape(baseline, (10, 2, 3)), 0, -1)

        self.pop1.baseline = baseline
        self.test_net.simulate(2)

    def test_get_size(self):
        """
        Tests the *size* method, which returns the number of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self.proj1.size, 6)

    def test_get_post_ranks(self):
        """
        Tests the *post_ranks* method, which returns the ranks of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self.proj1.post_ranks, list(range(6)))

    def test_pool(self):
        """
        Tests if after pooling the last dimension, the rates in the post
        projection are as expected
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

    def test_pool_delay(self):
        """
        Tests if after pooling the last dimension, the rates in the post
        projection are as expected
        """
        comb = numpy.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]])
        numpy.testing.assert_allclose(self.pop2.get('rd'), comb)

        # Simulate another 2 ms and verify the output
        self.test_net.simulate(2.0)
        comb = numpy.array([[2.7, 2.8, 2.9],
                            [3.0, 3.1, 3.2]])
        numpy.testing.assert_allclose(self.pop2.get('rd'), comb)
