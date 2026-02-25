"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron


class test_RateTransmission(unittest.TestCase):
    """
    This class tests the functionality of the some transmission patterns
    between rate-coded *Projections*.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(equations="r = 1")

        neuron2 = Neuron(equations="r = sum(exc)")

        cls._network = Network()
        pop1 = cls._network.create(geometry=(3, 3), neuron=neuron)
        pop2 = cls._network.create(geometry=(3, 3), neuron=neuron2)

        # One-to-one pattern, would raise an exception for dense pattern
        # and therefore we exclude this case
        if cls.storage_format != "dense":
            cls._proj_one_2_one = cls._network.connect(
                pre=pop1, post=pop2, target="exc"
            )
            cls._proj_one_2_one.one_to_one(
                weights=0.1,
                storage_format=cls.storage_format,
                storage_order=cls.storage_order,
            )

        # All-to-all pattern
        cls._proj_all_2_all = cls._network.connect(pre=pop1, post=pop2, target="exc")
        cls._proj_all_2_all.all_to_all(
            weights=0.1,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )

        # Fixed-number-pre pattern
        cls._proj_fnp = cls._network.connect(pre=pop1, post=pop2, target="exc")
        cls._proj_fnp.fixed_number_pre(
            3,
            weights=0.1,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before
        every test.
        """
        self._network.reset()

    def test_one_to_one_ranks(self):
        """
        Tests the *one_to_one* connectivity pattern, in which every
        pre-synaptic neuron is connected to every post-synaptic neuron.

        We test correctness of assigned ranks.
        """
        if self.storage_format == "dense":
            self.assertTrue(True)
        else:
            self.assertTrue(self._proj_one_2_one.dendrite(3).pre_ranks == [3])

    def test_one_to_one_weights(self):
        """
        Tests the *one_to_one* connectivity pattern, in which every
        pre-synaptic neuron is connected to every post-synaptic neuron.

        We test correctness of assigned weight values.
        """
        if self.storage_format == "dense":
            self.assertTrue(True)
        else:
            numpy.testing.assert_allclose(self._proj_one_2_one.dendrite(3).w, [0.1])

    def test_all_to_all_ranks(self):
        """
        Tests the *all_to_all* connectivity pattern, in which every
        pre-synaptic neuron is connected to every post-synaptic neuron.

        We test correctness of assigned ranks.
        """
        self.assertTrue(
            self._proj_all_2_all.dendrite(3).pre_ranks == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )

    def test_all_to_all_weights(self):
        """
        Tests the *all_to_all* connectivity pattern, in which every
        pre-synaptic neuron is connected to every post-synaptic neuron.

        We test correctness of assigned weight values.
        """
        numpy.testing.assert_allclose(
            self._proj_all_2_all.dendrite(3).w, numpy.ones((9)) * 0.1
        )

    def test_fnp_weights(self):
        """
        To verfiy the pattern, we determine the number of synapses in all
        dendrites.
        """
        tmp = [dend.size for dend in self._proj_fnp.dendrites]
        numpy.testing.assert_allclose(tmp, 3)
