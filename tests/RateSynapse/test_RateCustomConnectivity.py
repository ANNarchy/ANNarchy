"""

    test_RateCustomConnectivity.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    Copyright (C) 2016-2023 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import CSR, DiscreteUniform, Network, Neuron

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
        def my_diagonal(pre, post, weight):
            synapses = CSR()
            for post_rk in post.ranks:
                pre_ranks = []
                if post_rk-1 in pre.ranks:
                    pre_ranks.append(post_rk-1)
                if post_rk in pre.ranks:
                    pre_ranks.append(post_rk)
                if post_rk+1 in pre.ranks:
                    pre_ranks.append(post_rk+1)

                synapses.add(post_rk, pre_ranks, [weight]*len(pre_ranks),
                             [0]*len(pre_ranks))

            return synapses

        neuron = Neuron(
            equations="r = 1"
        )

        neuron2 = Neuron(
            equations="r = sum(exc)"
        )

        cls._network = Network()
        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=5, neuron=neuron2)

        cls._proj1 = cls._network.connect(pre=pop1, post=pop2, target="exc")
        cls._proj1.connect_with_func(method=my_diagonal, weight=0.1, storage_format=cls.storage_format)

        cls._network.compile(silent=True)

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
        def my_diagonal_with_uniform_delay(pre, post, weight, delay):
            synapses = CSR()
            for post_rk in post.ranks:
                pre_ranks = []
                if post_rk-1 in pre.ranks:
                    pre_ranks.append(post_rk-1)
                if post_rk in pre.ranks:
                    pre_ranks.append(post_rk)
                if post_rk+1 in pre.ranks:
                    pre_ranks.append(post_rk+1)

                synapses.add(post_rk, pre_ranks, [weight]*len(pre_ranks),
                             [delay])

            return synapses

        neuron = Neuron(
            equations="r = 1"
        )

        neuron2 = Neuron(
            equations="r = sum(exc)"
        )

        cls._network = Network()

        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=5, neuron=neuron2)

        cls._proj1 = cls._network.connect(pre=pop1, post=pop2, target="exc2")
        cls._proj1.connect_with_func(
            method=my_diagonal_with_uniform_delay,
            weight=0.1, delay=2,
            storage_format=cls.storage_format
        )

        cls._network.compile(silent=True)

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
        def my_diagonal_with_non_uniform_delay(pre, post, weight, delay):
            synapses = CSR()
            for post_rk in post.ranks:
                pre_ranks = []
                if post_rk-1 in pre.ranks:
                    pre_ranks.append(post_rk-1)
                if post_rk in pre.ranks:
                    pre_ranks.append(post_rk)
                if post_rk+1 in pre.ranks:
                    pre_ranks.append(post_rk+1)

                synapses.add(post_rk, pre_ranks, [weight]*len(pre_ranks),
                             delay.get_values(len(pre_ranks)))

            return synapses

        neuron = Neuron(
            equations="r = 1"
        )

        neuron2 = Neuron(
            equations="r = sum(exc)"
        )

        cls._network = Network()

        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=5, neuron=neuron2)

        cls._proj1 = cls._network.connect(pre=pop1, post=pop2, target="exc3")
        cls._proj1.connect_with_func(
            method=my_diagonal_with_non_uniform_delay,
            weight=0.1, delay=DiscreteUniform(1,5),
            storage_format=cls.storage_format
        )

        cls._network.compile(silent=True)

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
