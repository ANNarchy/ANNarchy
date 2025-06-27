"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import add_function, Network, Neuron, Synapse

add_function("glob_pos(x) = pos(x)")

class test_CustomFunc(unittest.TestCase):
    """
    This class tests the definition of custom functions, they
    can defined on three levels:

        * globally
        * within neurons
        * within synapses
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations = "r = transfer_function(sum(exc), 0.0)",
            functions = "transfer_function(x, t) = if x > t: if x > 2*t : (x - 2*t)^2 else: x - t else: 0."
        )

        neuron2 = Neuron(
            equations = "r = glob_pos(sum(exc))"
        )

        synapse = Synapse(
            equations="w += hebb(pre.r, post.r)",
            functions="hebb(x, y) = x * y"
        )

        cls._network = Network()

        cls._pop1 = cls._network.create(geometry=10, neuron=neuron)
        cls._pop2 = cls._network.create(geometry=10, neuron=neuron2)
        cls._proj = cls._network.connect(cls._pop1, cls._pop1, 'exc', synapse)
        cls._proj.all_to_all(1.0, storage_format="lil", storage_order="post_to_pre")

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_neuron(self):
        """
        Custom func defined within a neuron object, providing numpy.array data.
        """
        numpy.testing.assert_allclose(
            self._pop1.transfer_function(numpy.array([0., 1., 2., 3.]),
                                         numpy.array([2., 2., 2., 2.])),
            [0, 0, 0, 1])

    def test_neuron2(self):
        """
        Custom func defined within a neuron object, providing simple lists.
        """
        numpy.testing.assert_allclose(
            self._pop1.transfer_function([0., 1., 2., 3.], [2., 2., 2., 2.]),
            [0, 0, 0, 1])

    def test_synapse(self):
        """
        Custom func defined within a synapse object, providing simple lists.
        """
        numpy.testing.assert_allclose(
            self._proj.hebb(numpy.array([0., 1., 2., 3.]),
                               numpy.array([0., 1., 2., 3.])),
            [0, 1, 4, 9])

    def test_synapse2(self):
        """
        Custom func defined within a synapse object, providing simple lists.
        """
        numpy.testing.assert_allclose(
            self._proj.hebb([0., 1., 2., 3.], [0., 1., 2., 3.]),
            [0, 1, 4, 9])
