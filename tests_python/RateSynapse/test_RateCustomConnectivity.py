"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest

from conftest import TARGET_FOLDER
from ANNarchy import LILConnectivity, DiscreteUniform, Network, Neuron


class test_CustomConnectivityNoDelay(unittest.TestCase):
    """
    This class tests the functionality of user-defined connectivity patterns
    between two populations. The synapses are configured without synaptic delay.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """

        def my_diagonal(pre, post, dt, weight):
            synapses = LILConnectivity(dt=dt)
            for post_rk in post.ranks:
                pre_ranks = []
                if post_rk - 1 in pre.ranks:
                    pre_ranks.append(post_rk - 1)
                if post_rk in pre.ranks:
                    pre_ranks.append(post_rk)
                if post_rk + 1 in pre.ranks:
                    pre_ranks.append(post_rk + 1)

                synapses.add(
                    post_rk, pre_ranks, [weight] * len(pre_ranks), [0] * len(pre_ranks)
                )

            return synapses

        neuron = Neuron(equations="r = 1")

        neuron2 = Neuron(equations="r = sum(exc)")

        cls._network = Network()
        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=5, neuron=neuron2)

        cls._proj1 = cls._network.connect(pre=pop1, post=pop2, target="exc")
        cls._proj1.from_function(
            method=my_diagonal, weight=0.1, storage_format=cls.storage_format
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

    def test_invoke_compile(self):
        """
        Executes compile.
        """
        pass

    def test_no_delay(self):
        """
        If a projection has no delay, dt is returned.
        """
        return self.assertEqual(self._proj1.delay, 1.0)


class test_CustomConnectivityUniformDelay(unittest.TestCase):
    """
    This class tests the functionality of user-defined connectivity patterns
    between two populations. All synapses share the same delay.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """

        def my_diagonal_with_uniform_delay(pre, post, dt, weight, delay):
            synapses = LILConnectivity(dt=dt)
            for post_rk in post.ranks:
                pre_ranks = []
                if post_rk - 1 in pre.ranks:
                    pre_ranks.append(post_rk - 1)
                if post_rk in pre.ranks:
                    pre_ranks.append(post_rk)
                if post_rk + 1 in pre.ranks:
                    pre_ranks.append(post_rk + 1)

                synapses.add(post_rk, pre_ranks, [weight] * len(pre_ranks), [delay])

            return synapses

        neuron = Neuron(equations="r = 1")

        neuron2 = Neuron(equations="r = sum(exc)")

        cls._network = Network()

        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=5, neuron=neuron2)

        cls._proj1 = cls._network.connect(pre=pop1, post=pop2, target="exc2")
        cls._proj1.from_function(
            method=my_diagonal_with_uniform_delay,
            weight=0.1,
            delay=2,
            storage_format=cls.storage_format,
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

    def test_invoke_compile(self):
        """
        Executes compile.
        """
        pass

    def test_uniform_delay(self):
        """
        Tests the projection with a uniform delay.
        """
        return self.assertEqual(self._proj1.delay, 2.0)


class test_CustomConnectivityNonUniformDelay(unittest.TestCase):
    """
    This class tests the functionality of user-defined connectivity patterns
    between two populations. The synapses are implemented with a non-uniform
    delay.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """

        def my_diagonal_with_non_uniform_delay(pre, post, dt, weight, delay):
            synapses = LILConnectivity(dt=dt)
            for post_rk in post.ranks:
                pre_ranks = []
                if post_rk - 1 in pre.ranks:
                    pre_ranks.append(post_rk - 1)
                if post_rk in pre.ranks:
                    pre_ranks.append(post_rk)
                if post_rk + 1 in pre.ranks:
                    pre_ranks.append(post_rk + 1)

                synapses.add(
                    post_rk,
                    pre_ranks,
                    [weight] * len(pre_ranks),
                    delay.get_values(len(pre_ranks)),
                )

            return synapses

        neuron = Neuron(equations="r = 1")

        neuron2 = Neuron(equations="r = sum(exc)")

        cls._network = Network()

        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=5, neuron=neuron2)

        cls._proj1 = cls._network.connect(pre=pop1, post=pop2, target="exc3")
        cls._proj1.from_function(
            method=my_diagonal_with_non_uniform_delay,
            weight=0.1,
            delay=DiscreteUniform(1, 5),
            storage_format=cls.storage_format,
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

    def test_invoke_compile(self):
        """
        Executes compile.
        """
        pass
