"""

    test_SpikingDelays.py

    This file is part of ANNarchy.

    Copyright (C) 2022 Alex Schwarz, Helge Uelo Dinkelbach
        <helge.dinkelbach@gmail.com>

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

from ANNarchy import DiscreteUniform, Neuron, Population, Projection, Network, Uniform, Synapse

class test_SpikingNoDelay():
    """
    In this class the spiking transmission is computed and tested without any
    synaptic delay for the following patterns:
        * one_to_one
        * fixed_number_pre
        * all_to_all
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(parameters="v=0.0", spike="v>0")
        out1 = Neuron(equations="v += g_one2one", spike="v>30")
        out2 = Neuron(equations="v += g_all2all + g_fnp", spike="v>30")

        pop1 = Population((3, 3), neuron)
        pop2 = Population((3, 3), out1)
        pop3 = Population(4, out2)

        proj = Projection(pre=pop1, post=pop2, target="one2one")
        # weights set in the test
        proj.connect_one_to_one(weights=1.0, force_multiple_weights=True,
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        proj2 = Projection(pre=pop1, post=pop3, target="all2all")
        proj2.connect_all_to_all(weights=Uniform(0,1),
                                 storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)

        proj3 = Projection(pre=pop1, post=pop3, target="fnp")
        proj3.connect_fixed_number_pre(5, weights=Uniform(0,1),
                                       storage_format=cls.storage_format,
                                       storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, pop3, proj, proj2, proj3])
        cls.test_net.compile(silent=True)

        cls.net_pop1 = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)
        cls.net_pop3 = cls.test_net.get(pop3)
        cls.net_proj = cls.test_net.get(proj)
        cls.net_proj2 = cls.test_net.get(proj2)
        cls.net_proj3 = cls.test_net.get(proj3)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_one_to_one(self):
        """
        tests functionality of the one_to_one connectivity pattern
        """
        # generate test values
        pre_v = numpy.random.randint(0, 2, (3, 3))
        result = pre_v

        # set values
        self.net_pop1.v = pre_v

        # simulate 1 step
        self.test_net.simulate(2)


        # Verify with numpy result
        numpy.testing.assert_allclose(self.net_pop2.v, result)

    def test_all_to_all(self):
        """
        tests functionality of the all_to_all connectivity pattern
        """
        # generate test values
        pre_v = numpy.random.random((1, 9))
        weights = self.net_proj2.connectivity_matrix()
        result = numpy.sum(numpy.multiply(weights, pre_v), axis=1)

        # set values
        self.net_pop1.v = pre_v

        # simulate 1 step
        self.test_net.simulate(1)

        # Verify with numpy result
        numpy.testing.assert_allclose(self.net_pop3.sum("all2all"), result)

    def test_fixed_number_pre(self):
        """
        tests functionality of the fixed_number_pre connectivity pattern
        """
        pre_v = numpy.random.random((1, 9))
        weights = self.net_proj3.connectivity_matrix()
        result = numpy.sum(numpy.multiply(weights, pre_v), axis=1)

        # set values
        self.net_pop1.v = pre_v

        # simulate 1 step
        self.test_net.simulate(1)

        # Verify with numpy result
        numpy.testing.assert_allclose(self.net_pop3.sum("fnp"), result)

class test_SpikingUniformDelay():
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

        pop1 = Population((3), input_neuron)
        pop2 = Population((3), neuron2)

        # A projection with uniform delay
        proj = Projection(pre=pop1, post=pop2, target="ff")
        proj.connect_one_to_one(weights=1.0, delays=10.0,
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        # A projection with uniform delay
        proj2 = Projection(pre=pop1, post=pop2, target="ff_glob",
                           synapse=synapse_glob)
        proj2.connect_one_to_one(weights=1.0, delays=10.0,
                                 storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)

        # Build up network
        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj, proj2])
        cls.test_net.compile(silent=True)

        # Store references for easier usage in test cases
        cls.net_proj = cls.test_net.get(proj)
        cls.net_proj2 = cls.test_net.get(proj2)
        cls.net_pop1 = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

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
        self.test_net.simulate(10)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 11th step we have the first queue
        # value in our case t = 0
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 15th -> t = 4
        self.test_net.simulate(4)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), 9.0)

    def test_configured_delay_global(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # The first ten steps, we have
        # initialization value
        self.test_net.simulate(10)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 11th step we have the first queue
        # value in our case t = 0
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 15th -> t = 4
        self.test_net.simulate(4)
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
        self.test_net.simulate(5)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 6th step we have the first queue
        # value in our case t = 0
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), -1.0)

        # at 10th -> t = 4
        self.test_net.simulate(4)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), 9.0)

    def test_modified_delay_global(self):
        """
        tests the delay with global attributes but the delay is changed.
        """
        # redefine synaptic delay
        self.net_proj2.delay = 5.0

        # The first ten steps, we have
        # initialization value
        self.test_net.simulate(5)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 6th step we have the first queue
        # value in our case t = 0
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), -1.0)

        # at 10th -> t = 4
        self.test_net.simulate(4)
        numpy.testing.assert_allclose(self.net_pop2.sum("ff_glob"), 9.0)

class test_SpikingNonuniformDelay():
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

        pop1 = Population((3), input_neuron)
        pop2 = Population((3), neuron2)

        # A projection with non-uniform delay
        proj = Projection(pop1, pop2, target="ff")
        proj.connect_one_to_one(weights=1.0, delays=DiscreteUniform(1, 5),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        # Build up network
        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj])
        cls.test_net.compile(silent=True)

        # Store references for easier usage in test cases
        cls.net_proj = cls.test_net.get(proj)
        cls.net_pop1 = cls.test_net.get(pop1)
        cls.net_pop2 = cls.test_net.get(pop2)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

        # for unittest we fix the delay of the non-uniform case
        self.net_proj.delay = [[3], [5], [2]]

    def test_configured_nonuniform_delay(self):
        """
        tests the delay functionality with the configured 10ms in connect call.
        """
        # run 5ms
        self.test_net.simulate(5)

        # r = [-1, 0, 2, 5, 9, 14, 20, ...]
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), [0.0, -1.0, 2.0])

    def test_modified_nonuniform_delay(self):
        """
        tests the delay functionality but the delay is changed.
        """
        # redefine synaptic delay, in this case to uniform
        self.net_proj.delay = [[3.0], [3.0], [3.0]]

        # run 10 ms
        self.test_net.simulate(10)

        # should access (t-3)th element
        numpy.testing.assert_allclose(self.net_pop2.sum("ff"), [20.0, 20.0, 20.0])

class test_SynapseOperations():
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

        pop1 = Population((3, 3), neuron=input_neuron)
        pop2 = Population(4, neuron=output_neuron)

        proj1 = Projection(pop1, pop2, target="p1", synapse=syn_max)
        proj1.connect_all_to_all(weights=1.0, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)
        proj2 = Projection(pop1, pop2, target="p2", synapse=syn_min)
        proj2.connect_all_to_all(weights=1.0, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)
        proj3 = Projection(pop1, pop2, target="p3", synapse=syn_mean)
        proj3.connect_all_to_all(weights=1.0, storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj1, proj2, proj3])
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
        self.test_net.simulate(1)

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
        self.test_net.simulate(1)

        # verify agains numpy
        numpy.testing.assert_allclose(self.net_pop2.sum("p3"), res_mean)

class test_SynapticAccess():
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

        pre = Population(6, neuron)
        post = Population(1, neuron)
        proj = Projection(pre, post, "exc", synapse = cov)
        proj.connect_all_to_all(weights=1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pre, post, proj])

        cls.test_net.compile(silent=True)

        cls.net_pop = cls.test_net.get(post)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net

    def test_compile(self):
        """
        Tests the result of *norm1(r)* for *pop*.
        """
        pass

if __name__ == "__main__":
    import unittest
    def run_with(c, formats, orders):
        """
        Run the tests with all given storage formats and orders. This is achieved
        by copying the classes for every data format.
        """
        for s_format in formats:
            for s_order in orders:
                if s_order == "pre_to_post" and s_format not in ["lil", "csr"]:
                    continue
                cls_name = c.__name__ + "_" + str(s_format) + "_" + str(s_order)
                glob = {"storage_format":s_format, "storage_order":s_order}
                globals()[cls_name] = type(cls_name, (c, unittest.TestCase), glob)
        # Delete the base class so that it will not be done again
        del globals()[c.__name__]
        del c

    mode = ["lil"]
    storage_orders = ['post_to_pre']
    loc = [l for l in locals() if l.startswith('test_')]

    for c in loc:
        run_with(locals()[c], mode, storage_orders)
    unittest.main()
