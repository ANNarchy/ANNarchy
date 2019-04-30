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

from ANNarchy import Neuron, Population, Projection, Network, Uniform, Synapse

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
            parameters="r=0.0"
        )

        out1 = Neuron(
            equations="""
                r =  sum(one2one)
            """
        )

        out2 = Neuron(
            equations="""
                r =  sum(all2all)
            """
        )

        pop1 = Population((3, 3), neuron)
        pop2 = Population((3, 3), out1)
        pop3 = Population(4, out2)

        proj = Projection(pre=pop1, post=pop2, target="one2one")
        proj.connect_one_to_one(weights=0.1)    # creates 3x3 matrix

        proj2 = Projection(pre=pop1, post=pop3, target="all2all")
        proj2.connect_all_to_all(weights=0.1)   # creates 4x9 matrix

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, pop3, proj, proj2])
        cls.test_net.compile(silent=True)

        cls.net_pop1 = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)
        cls.net_pop3 = cls.test_net.get(pop3)
        cls.net_proj2 = cls.test_net.get(proj2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_one_to_one(self):
        """
        tests functionality of the one_to_one connectivity pattern
        """
        # sum up r = 2
        self.net_pop1.r = numpy.ones((3, 3))*2
        self.test_net.simulate(1)

        # sum up on entry per neuron with r = 2 * w = 0.1 -> 0.2 as result
        self.assertTrue(numpy.allclose(self.net_pop2.sum("one2one"), 0.2))

    def test_all_to_all(self):
        """
        tests functionality of the all_to_all connectivity pattern
        """
        # sum up r = 2
        self.net_pop1.r = numpy.ones((3, 3))
        self.test_net.simulate(1)

        # sum up on 9 entries per neuron with r = 1 * w = 0.1 -> 0.9 as result
        self.assertTrue(numpy.allclose(self.net_pop3.sum("all2all"), 0.9))

    def test_all_to_all_rand_values(self):
        """
        tests functionality of the all_to_all connectivity pattern but
        with random drawn numbers.
        """
        # generate test values
        pre_r = numpy.random.random((1, 9))
        weights = numpy.random.random((4, 9))
        result = numpy.sum(numpy.multiply(weights, pre_r), axis=1)

        # set values
        self.net_pop1.r = pre_r
        self.net_proj2.w = weights

        # Compute
        self.test_net.simulate(1)

        # Verify with numpy result
        self.assertTrue(numpy.allclose(self.net_pop3.sum("all2all"), result))

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


class test_RateTransmissionGlobalOperations(unittest.TestCase):
    """
    Next to the weighted sum across inputs we allow the application of global
    operations (min, max, mean).
    """
    @classmethod
    def setUpClass(cls):
        """
        Define and compile the network for this test.
        """
        input_neuron = Neuron(
            parameters="r=0.0"
        )

        output_neuron = Neuron(
            equations="""
                r = sum(p1) + sum(p2)
            """
        )

        syn_max = Synapse(
            psp="pre.r * w",
            operation="max"
        )

        syn_min = Synapse(
            psp="pre.r * w",
            operation="min"
        )

        pop1 = Population((3, 3), neuron=input_neuron)
        pop2 = Population(4, neuron=output_neuron)

        proj1 = Projection(pop1, pop2, target="p1", synapse=syn_max)
        proj1.connect_all_to_all(weights=1.0)
        proj2 = Projection(pop1, pop2, target="p2", synapse=syn_min)
        proj2.connect_all_to_all(weights=1.0)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj1, proj2])
        cls.test_net.compile(silent=True)

        cls.net_pop1 = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_max(self):
        """
        tests functionality of the all_to_all connectivity pattern combined
        with maximum across pre-synaptic firing rate.
        """
        pre_r = numpy.random.random((3, 3))
        res_max = numpy.amax(pre_r) # weights=1.0

        # set value
        self.net_pop1.r = pre_r

        # compute
        self.test_net.simulate(1)

        # verify agains numpy
        self.assertTrue(numpy.allclose(self.net_pop2.sum("p1"), res_max))

    def test_min(self):
        """
        tests functionality of the all_to_all connectivity pattern combined
        with minimum across pre-synaptic firing rate.
        """
        pre_r = numpy.random.random((3, 3))
        res_min = numpy.amin(pre_r) # weights=1.0

        # set value
        self.net_pop1.r = pre_r

        # compute
        self.test_net.simulate(1)

        # verify agains numpy
        self.assertTrue(numpy.allclose(self.net_pop2.sum("p2"), res_min))
