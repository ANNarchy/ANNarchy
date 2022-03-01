"""

    test_RateTransmission.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    Copyright (C) 2016-2019 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import Neuron, Population, Projection, Network, CSR, DiscreteUniform


class test_RateTransmission():
    """
    This class tests the functionality of the transmission patterns within
    rate-coded *Projections*.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations="r = 1"
        )

        neuron2 = Neuron(
            equations="r = sum(exc)"
        )

        pop1 = Population((3, 3), neuron)
        pop2 = Population((3, 3), neuron2)

        proj1 = Projection(pre=pop1, post=pop2, target="exc")
        proj2 = Projection(pre=pop1, post=pop2, target="exc")
        proj3 = Projection(pre=pop1, post=pop2, target="exc")

        proj1.connect_one_to_one(weights=0.1, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)
        proj2.connect_all_to_all(weights=0.1, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)
        proj3.connect_fixed_number_pre(3, weights=0.1,
                                       storage_format=cls.storage_format,
                                       storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj1, proj2, proj3])
        cls.test_net.compile(silent=True)

        cls.test_proj1 = cls.test_net.get(proj1)
        cls.test_proj2 = cls.test_net.get(proj2)
        cls.test_proj3 = cls.test_net.get(proj3)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before
        every test.
        """
        self.test_net.reset()

    def test_one_to_one(self):
        """
        Tests the *one_to_one* connectivity pattern, in which every
        pre-synaptic neuron is connected to its ranked equivalent post-synaptic
        neuron.

        We test correctness of ranks and weight values.
        """
        self.assertTrue(self.test_proj1.dendrite(3).pre_ranks == [3])
        numpy.testing.assert_allclose(self.test_proj1.dendrite(3).w, [0.1])

    def test_all_to_all(self):
        """
        Tests the *all_to_all* connectivity pattern, in which every
        pre-synaptic neuron is connected to every post-synaptic neuron.

        We test correctness of ranks and weight values.
        """
        self.assertTrue(self.test_proj2.dendrite(3).pre_ranks == [0, 1, 2, 3, 4, 5, 6, 7, 8])
        numpy.testing.assert_allclose(self.test_proj2.dendrite(3).w,
                                      numpy.ones((8, 1)) * 0.1)

    def test_fixed_number_pre(self):
        """
        To verfiy the pattern, we determine the number of synapses in all
        dendrites.
        """
        tmp = [dend.size for dend in self.test_proj3.dendrites]
        numpy.testing.assert_allclose(tmp, 3)

class test_CustomConnectivity(unittest.TestCase):
    """
    This class tests the functionality of user-defined connectivity patterns
    between two populations.
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
                             [delay]*len(pre_ranks))

            return synapses

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

        pop1 = Population(5, neuron)
        pop2 = Population(5, neuron2)

        proj1 = Projection(pre=pop1, post=pop2, target="exc")
        proj1.connect_with_func(method=my_diagonal, weight=0.1)

        proj2 = Projection(pre=pop1, post=pop2, target="exc2")
        proj2.connect_with_func(method=my_diagonal_with_uniform_delay,
                                weight=0.1, delay=2)

        proj3 = Projection(pre=pop1, post=pop2, target="exc3")
        proj3.connect_with_func(method=my_diagonal_with_non_uniform_delay,
                                weight=0.1, delay=DiscreteUniform(1,5))

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj1, proj2, proj3])
        cls.test_net.compile(silent=True)

        cls.test_proj1 = cls.test_net.get(proj1)
        cls.test_proj2 = cls.test_net.get(proj2)
        cls.test_proj3 = cls.test_net.get(proj3)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before
        every test.
        """
        self.test_net.reset()

    def test_invoke_compile(self):
        """
        Executes compile.
        """
        pass

    def test_no_delay(self):
        """
        If a projection has no delay, dt is returned.
        """
        return self.assertEqual(self.test_proj1.delay, 1.0)

    def test_uniform_delay(self):
        """
        Tests the projection with a uniform delay.
        """
        all_equal = True
        for dendrite_delay in self.test_proj2.delay:
            if not numpy.allclose(dendrite_delay, 2.0):
                all_equal = False
                break

        return self.assertTrue(all_equal)
