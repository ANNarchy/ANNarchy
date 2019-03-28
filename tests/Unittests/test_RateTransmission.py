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
from scipy import sparse

from ANNarchy import *


class test_RateTransmission(unittest.TestCase):
    """
    One major function for rate-coded neurons is the computation of the
    continous transmission between neurons. In this class the continous
    transmission is computed and testd for the following patterns:

        * one2one
        * all2all

    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations="r = 1 + t"
        )

        neuron2 = Neuron(
            equations="""
                sum1 = sum(one2one)
                sum2 = sum(all2all)
                r =  sum1 + sum2
            """
        )

        pop1 = Population((3, 3), neuron)
        pop2 = Population((3, 3), neuron2)

        proj = Projection(pre=pop1, post=pop2, target="one2one")
        proj.connect_one_to_one(weights=1.0)

        proj2 = Projection(pre=pop1, post=pop2, target="all2all")
        proj2.connect_all_to_all(weights=1.0)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj, proj2])
        cls.test_net.compile(silent=True)

        cls.net_pop2 = cls.test_net.get(pop2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_one_to_one(self):
        """
        tests functionality of the one_to_one connectivity pattern
        """
        # sum up r = 1
        self.test_net.simulate(2)
        self.assertTrue(numpy.allclose(self.net_pop2.sum1, 1.0))

    def test_all_to_all(self):
        """
        tests functionality of the all_to_all connectivity pattern
        """
        # sum up r = 1, 9 neurons
        self.test_net.simulate(2)
        self.assertTrue(numpy.allclose(self.net_pop2.sum2, 9.0))

class test_RateTransmissionDelayLocalVariable(unittest.TestCase):
    """
    One major function for rate-coded neurons is the computation of continuous
    transmission between neurons. In this class the continuous transmission is
    computed and tested for the one2one patterns with a special focus on using
    uniform synaptic delays.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.

        The input_neuron will generate a sequence of values:

            r_t = [-1, 0, 2, 5, 9, 14, 20, ...]
        """
        input_neuron = Neuron(
            equations="""
                r = r + t : init = -1
            """
        )

        neuron2 = Neuron(
            equations="""
                sum1 = sum(uni_delay)
                sum2 = sum(non_uni_delay)
                r = sum1 + sum2
            """
        )

        pop1 = Population((3), input_neuron)
        pop2 = Population((3), neuron2)

        # A projection with uniform delay
        proj_uni_d = Projection(pre=pop1, post=pop2, target="uni_delay")
        proj_uni_d.connect_one_to_one(weights=1.0, delays=10.0)

        # Build up network
        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj_uni_d])
        cls.test_net.compile(silent=True)

        # Store references for easier usage in test cases
        cls.net_proj_uni_d = cls.test_net.get(proj_uni_d)
        cls.net_pop1 = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_init_values(self):
        """
        Check if the connectivity pattern was correctly set.
        """
        self.assertTrue(numpy.allclose(self.net_proj_uni_d.w, 1.0))
        self.assertTrue(numpy.allclose(self.net_proj_uni_d.delay, 10.0))

    def test_set_delay(self):
        """
        Check if the connectivity pattern was correctly set.
        """
        # check uniform delay
        new_d = 2.0
        self.net_proj_uni_d.delay = new_d
        self.assertTrue(numpy.allclose(self.net_proj_uni_d.delay, new_d))

    def test_configured_delay(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # The first ten steps, we have
        # initialization value
        self.test_net.simulate(10)
        self.assertTrue(numpy.allclose(self.net_pop2.sum1, -1.0))

        # at 11th step we have the first queue
        # value in our case t = 0
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.net_pop2.sum1, -1.0))

        # at 15th -> t = 4
        self.test_net.simulate(4)
        self.assertTrue(numpy.allclose(self.net_pop2.sum1, 9.0))

    def test_modified_delay(self):
        """
        tests the delay functionality but the delay is changed.
        """
        # redefine synaptic delay
        self.net_proj_uni_d.delay = 5.0

        # The first ten steps, we have
        # initialization value
        self.test_net.simulate(5)
        self.assertTrue(numpy.allclose(self.net_pop2.sum1, -1.0))

        # at 6th step we have the first queue
        # value in our case t = 0
        self.test_net.simulate(1)
        self.assertTrue(numpy.allclose(self.net_pop2.sum1, -1.0))

        # at 10th -> t = 4
        self.test_net.simulate(4)
        self.assertTrue(numpy.allclose(self.net_pop2.sum1, 9.0))

class test_RateTransmissionNonuniformDelayLocalVariable(unittest.TestCase):
    """
    One major function for rate-coded neurons is the computation of continuous
    transmission between neurons. In this class the continuous transmission is
    computed and tested for the one2one patterns with a special focus on using
    uniform synaptic delays.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.

        The input_neuron will generate a sequence of values:

            r_t = [-1, 0, 2, 5, 9, 14, 20, ...]
        """
        input_neuron = Neuron(
            equations="""
                r = r + t : init = -1
            """
        )

        neuron2 = Neuron(
            equations="""
                sum1 = sum(uni_delay)
                sum2 = sum(non_uni_delay)
                r = sum1 + sum2
            """
        )

        pop1 = Population((3), input_neuron)
        pop2 = Population((3), neuron2)

        # A projection with non-uniform delay
        proj_non_uni_d = Projection(pop1, pop2, target="non_uni_delay")
        proj_non_uni_d.connect_one_to_one(weights=1.0, delays=Uniform(1, 5))

        # Build up network
        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj_non_uni_d])
        cls.test_net.compile(silent=True)

        # Store references for easier usage in test cases
        cls.net_proj_non_uni_d = cls.test_net.get(proj_non_uni_d)
        cls.net_pop1 = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

        # for unittest we fix the delay of the non-uniform case
        self.net_proj_non_uni_d.delay = [[3], [5], [2]]

    def test_configured_nonuniform_delay(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # run 5ms
        self.test_net.simulate(5)

        # r = [-1, 0, 2, 5, 9, 14, 20, ...]
        self.assertTrue(numpy.allclose(self.net_pop2.sum2, [0.0, -1.0, 2.0]))

    def test_modified_nonuniform_delay(self):
        """
        tests the delay functionality but the delay is changed.
        """
        # redefine synaptic delay, in this case to uniform
        self.net_proj_non_uni_d.delay = [[3.0], [3.0], [3.0]]

        # run 10 ms
        self.test_net.simulate(10)

        # should access (t-3)th element
        self.assertTrue(numpy.allclose(self.net_pop2.sum2, [20.0, 20.0, 20.0]))

class test_RateTransmissionGlobal(unittest.TestCase):
    """
    One major function for rate-coded neurons is the computation of continuous
    transmission between neurons. In this class the continuous transmission is
    computed and tested with a special focus on using global variables.
    """
    @classmethod
    def setUpClass(cls):
        """
        Define and compile the network for this test.

        The input_neuron will generate a sequence of values:

            r_t = [-1, 0, 2, 5, 9, 14, 20, ...]

        The initial value is set to -1 to be different from default vector
        initialization values.

        The output_neuron will replicate the chain, as psp is default (w*r),
        while w=1 and we have a one2one pattern. This makes it easy, to predict
        the outcome.
        """
        input_neuron = Neuron(
            equations="""
                r = r+t : population, init=-1
            """
        )

        output_neuron = Neuron(
            equations="""
                r =  sum(one2one)
            """
        )

        pop1 = Population((3), input_neuron)
        pop2 = Population((3), output_neuron)

        proj = Projection(pre=pop1, post=pop2, target="one2one")
        proj.connect_one_to_one(weights=1.0, delays=10.0)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj])
        cls.test_net.compile(silent=True)

        cls.net_proj = cls.test_net.get(proj)
        cls.net_pop = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_configured_delay(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # run 10 ms
        self.test_net.simulate(10)

        # should access (t-3)th element
        self.assertTrue(numpy.allclose(self.net_pop2.sum('one2one'), [-1, -1, -1]))

        # run another 5 ms
        self.test_net.simulate(5)

        # after 15 ms we access r[4] (t=15, d=10 t-d-1 leads to 4)
        self.assertTrue(numpy.allclose(self.net_pop2.sum('one2one'), [9, 9, 9]))
