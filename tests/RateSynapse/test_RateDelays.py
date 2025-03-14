"""

    test_RateDelays.py

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
import numpy
import unittest
from ANNarchy import Network, Neuron, Synapse, DiscreteUniform, Uniform

class test_NoDelay(unittest.TestCase):
    """
    One major function for rate-coded neurons is the computation of the
    continous transmission between neurons. In this class the continous
    transmission is computed and tested for the following patterns:

        * one_to_one
        * fixed_number_pre
        * all_to_all

    without any synaptic delay.
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
                r =  sum(all2all) + sum(fnp)
            """
        )

        cls._network = Network()

        cls.net_pop1 = cls._network.create(geometry=(17, 17), neuron=neuron)
        cls.net_pop2 = cls._network.create(geometry=(17, 17), neuron=out1)
        cls.net_pop3 = cls._network.create(geometry=4, neuron=out2)

        # One-to-one pattern, would raise an exception for dense pattern
        # and therefore we exclude this case
        if cls.storage_format != "dense":
            cls.net_proj = cls._network.connect(pre=cls.net_pop1, post=cls.net_pop2, target="one2one")
            cls.net_proj.one_to_one(weights=Uniform(0,1),
                                    storage_format=cls.storage_format,
                                    storage_order=cls.storage_order)

        cls.net_proj2 = cls._network.connect(pre=cls.net_pop1, post=cls.net_pop3, target="all2all")
        cls.net_proj2.all_to_all(weights=Uniform(0,1),
                                 storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)

        cls.net_proj3 = cls._network.connect(pre=cls.net_pop1, post=cls.net_pop3, target="fnp")
        cls.net_proj3.fixed_number_pre(5, weights=Uniform(0,1),
                                       storage_format=cls.storage_format,
                                       storage_order=cls.storage_order)

        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self._network.reset()

    def test_one_to_one(self):
        """
        tests functionality of the one_to_one connectivity pattern
        """
        if self.storage_format == "dense":
            self.assertTrue(True)
        else:
            # generate test values
            pre_r = numpy.random.random((1, 289))
            weights = self.net_proj.connectivity_matrix()
            result = numpy.sum(numpy.multiply(weights, pre_r), axis=1)

            # set values
            self.net_pop1.r = pre_r

            # simulate 1 step
            self._network.simulate(1)

            # Verify with numpy result
            numpy.testing.assert_allclose(self.net_pop2.sum("one2one"), result)

    def test_all_to_all(self):
        """
        tests functionality of the all_to_all connectivity pattern
        """
        # generate test values
        pre_r = numpy.random.random((1, 289))
        weights = self.net_proj2.connectivity_matrix()
        result = numpy.sum(numpy.multiply(weights, pre_r), axis=1)

        # set values
        self.net_pop1.r = pre_r

        # simulate 1 step
        self._network.simulate(1)

        # Verify with numpy result
        numpy.testing.assert_allclose(self.net_pop3.sum("all2all"), result)

    def test_fixed_number_pre(self):
        """
        tests functionality of the fixed_number_pre connectivity pattern
        """
        pre_r = numpy.random.random((1, 289))
        weights = self.net_proj3.connectivity_matrix()
        result = numpy.sum(numpy.multiply(weights, pre_r), axis=1)

        # set values
        self.net_pop1.r = pre_r

        # simulate 1 step
        self._network.simulate(1)

        # Verify with numpy result
        numpy.testing.assert_allclose(self.net_pop3.sum("fnp"), result)


class test_UniformDelay(unittest.TestCase):
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

        one time as global (glob_r) and one time as local variable (r).
        """
        input_neuron = Neuron(
            equations="""
                glob_r = glob_r + t : init = -1, population
                r = r + t : init = -1
            """
        )

        neuron2 = Neuron(
            equations="""
                r = sum(ff)
            """
        )

        synapse_glob = Synapse(psp="pre.glob_r * w")

        cls._network = Network()

        cls.net_pop1 = cls._network.create(geometry=(3), neuron=input_neuron)
        cls.net_pop2 = cls._network.create(geometry=(3), neuron=neuron2)

        # HD: previously, we use here an one-to-one pattern. However, dense
        #     matrix formats throw an exception in this case. Therefore, I use
        #     a little trick here limiting the fixed-number-pre pattern to one
        #     synapse.

        # A projection with uniform delay - default psp accesses local attribute r
        cls.net_proj = cls._network.connect(pre=cls.net_pop1, post=cls.net_pop2, target="ff")
        cls.net_proj.fixed_number_pre(
            1, weights=1.0, delays=10.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        # A projection with uniform delay - default psp accesses global attribute r
        cls.net_proj2 = cls._network.connect(pre=cls.net_pop1, post=cls.net_pop2, target="ff_glob", synapse=synapse_glob)
        cls.net_proj2.fixed_number_pre(
            1, weights=1.0, delays=10.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        # Build up network
        cls._network.compile(silent=True)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self._network.reset()

    def test_get_delay(self):
        """
        Check if the connection delay is accessable.
        """
        numpy.testing.assert_allclose(self.net_proj.delay, 10.0)

    def test_set_delay(self):
        """
        Check if the connection delay can be modified.
        """
        self.net_proj.delay = 2.0
        numpy.testing.assert_allclose(self.net_proj.delay, 2.0)

    def test_configured_delay_local(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # The first ten steps, we have
        # initialization value
        self._network.simulate(10)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 11th step we have the first queue
        # value in our case t = 0
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 15th -> t = 4
        self._network.simulate(4)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), 9.0)

    def test_configured_delay_global(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # The first ten steps, we have
        # initialization value
        self._network.simulate(10)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 11th step we have the first queue
        # value in our case t = 0
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 15th -> t = 4
        self._network.simulate(4)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), 9.0)

    def test_modified_delay_local(self):
        """
        tests the delay functionality for local attributes but the delay is
        changed.
        """
        # redefine synaptic delay
        self.net_proj.delay = 5.0

        # The first ten steps, we have
        # initialization value
        self._network.simulate(5)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 6th step we have the first queue
        # value in our case t = 0
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 10th -> t = 4
        self._network.simulate(4)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), 9.0)

    def test_modified_delay_global(self):
        """
        tests the delay with global attributes but the delay is changed.
        """
        # redefine synaptic delay
        self.net_proj2.delay = 5.0

        # The first ten steps, we have
        # initialization value
        self._network.simulate(5)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 6th step we have the first queue
        # value in our case t = 0
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 10th -> t = 4
        self._network.simulate(4)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), 9.0)


class test_NonUniformDelay(unittest.TestCase):
    """
    One major function for rate-coded neurons is the computation of continuous
    transmission between neurons. In this class the continuous transmission is
    computed and tested for the one2one patterns with a special focus on using
    non-uniform synaptic delays.
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
                r = sum(ff)
            """
        )

        cls._network = Network()

        cls.net_pop1 = cls._network.create(geometry=(3), neuron=input_neuron)
        cls.net_pop2 = cls._network.create(geometry=(3), neuron=neuron2)

        # A projection with non-uniform delay
        cls.net_proj = cls._network.connect(cls.net_pop1, cls.net_pop2, target="ff")
        cls.net_proj.fixed_number_pre(
            1, weights=1.0, delays=DiscreteUniform(1, 5),
            storage_format=cls.storage_format,
            storage_order=cls.storage_order)

        # Build up network
        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self._network.reset()

        # for unittest we fix the delay of the non-uniform case
        self.net_proj.delay = [[3], [5], [2]]

    def test_configured_nonuniform_delay(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # run 5ms
        self._network.simulate(5)

        # r = [-1, 0, 2, 5, 9, 14, 20, ...]
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), [0.0, -1.0, 2.0])

    def test_modified_nonuniform_delay(self):
        """
        tests the delay functionality but the delay is changed.
        """
        # redefine synaptic delay, in this case to uniform
        self.net_proj.delay = [[3.0], [3.0], [3.0]]

        # run 10 ms
        self._network.simulate(10)

        # should access (t-3)th element
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), [20.0, 20.0, 20.0])


class test_SynapseOperations(unittest.TestCase):
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

        syn_mean = Synapse(
            psp="pre.r * w",
            operation="mean"
        )

        cls._network = Network()

        cls.net_pop1 = cls._network.create(geometry=(3, 3), neuron=input_neuron)
        cls.net_pop2 = cls._network.create(geometry=4, neuron=output_neuron)

        proj1 = cls._network.connect(cls.net_pop1, cls.net_pop2, target="p1", synapse=syn_max)
        proj1.all_to_all(weights=1.0, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)
        proj2 = cls._network.connect(cls.net_pop1, cls.net_pop2, target="p2", synapse=syn_min)
        proj2.all_to_all(weights=1.0, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)
        proj3 = cls._network.connect(cls.net_pop1, cls.net_pop2, target="p3", synapse=syn_mean)
        proj3.all_to_all(weights=1.0, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)

        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self._network.reset()

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
        self._network.simulate(1)

        # verify agains numpy
        numpy.testing.assert_allclose(self.net_pop2.sum("p1"), res_max)

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
        self._network.simulate(1)

        # verify agains numpy
        numpy.testing.assert_allclose(self.net_pop2.sum("p2"), res_min)

    def test_mean(self):
        """
        tests functionality of the all_to_all connectivity pattern combined
        with mean across pre-synaptic firing rate.
        """
        pre_r = numpy.random.random((3, 3))
        res_mean = numpy.mean( pre_r ) # weights=1.0

        # set value
        self.net_pop1.r = pre_r

        # compute
        self._network.simulate(1)

        # verify agains numpy
        numpy.testing.assert_allclose(self.net_pop2.sum("p3"), res_mean)


class test_SynapticAccess(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects.

    This particular test focuses on the usage of them in synaptic learning
    rules (for instance covariance).
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r=0
            """
        )

        cov = Synapse(
            parameters="""
                tau = 5000.0
            """,
            equations="""
                tau * dw/dt = (pre.r - mean(pre.r) ) * (post.r - mean(post.r) )
            """
        )

        cls._network = Network()

        pre = cls._network.create(geometry=6, neuron=neuron)
        cls.net_pop = cls._network.create(1, neuron)
        proj = cls._network.connect(pre, cls.net_pop, "exc", synapse = cov)
        proj.all_to_all(weights=1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def test_compile(self):
        """
        Tests the result of *norm1(r)* for *pop*.
        """
        pass
