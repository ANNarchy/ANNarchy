"""

    test_StructuralPlasticity.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import clear, LeakyIntegrator, Network, Neuron, Population, \
    Projection, setup, Synapse


class test_StructuralPlasticityModel(unittest.TestCase):
    """
    This class tests the *Structural Plasticity* feature, which can optinally
    be enabled.

    In the synapse description an user can define prune and create conditions.
    This unittest was inspired by an submitted issue #41 (on bitbucket.org).
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test, this was an example defintion
        provided by the reporter.
        """
        setup(structural_plasticity=True)

        value_synapse = Synapse(
            parameters="""
                tau_utility = 1000
            """,
            equations="""
                tau_utility * dutility/dt = pre.r * post.r : init=1.0
            """,
            creating="pre.r * post.r > 0.9 : proba = 1.0, w = 1.0",
            pruning="utility < 0.0 : proba = 0.5",
            operation="max"
        )

        v = Population(geometry=5, neuron=LeakyIntegrator())

        value_proj = Projection(pre=v, post=v, target="exc", synapse=value_synapse)
        value_proj.connect_fixed_number_pre(number=1, weights=1.0)

        # build the network
        cls.test_net = Network()
        cls.test_net.add([v, value_proj])
        cls.test_net.compile(silent=True)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset()

    @classmethod
    def tearDownClass(cls):
        """
        Remove the structural_plasticity global flag to not interfere with
        further tests.
        """
        clear()
        setup(structural_plasticity=False)

    def test_invoke_compile(self):
        """
        This test case just invoke the compilation process for the above
        defined network definition.
        """
        pass


class test_StructuralPlasticityModelDelay(unittest.TestCase):
    """
    This class tests the *Structural Plasticity* feature, which can optionally
    be enabled.

    In the synapse description an user can define prune and create conditions.
    This class behaves like test_StructuralPlasticityModel apart from using
    synaptic delays.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test, this was an example defintion
        provided by the reporter.
        """
        setup(structural_plasticity=True)

        value_synapse = Synapse(
            parameters="""
                tau_utility = 1000
            """,
            equations="""
                tau_utility * dutility/dt = pre.r * post.r : init=1.0
            """,
            creating="pre.r * post.r > 0.9 : proba = 1.0, w = 1.0",
            pruning="utility < 0.0 : proba = 0.5",
            operation="max"
        )

        v = Population(geometry=5, neuron=LeakyIntegrator())

        value_proj = Projection(pre=v, post=v, target="exc", synapse=value_synapse)
        value_proj.connect_fixed_number_pre(number=1, weights=1.0, delays=2.0)

        # build the network
        cls.test_net = Network()
        cls.test_net.add([v, value_proj])
        cls.test_net.compile(silent=True)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        setup(structural_plasticity=True)
        self.test_net.reset()

    @classmethod
    def tearDownClass(cls):
        """
        Remove the structural_plasticity global flag to not interfere with
        further tests.
        """
        setup(structural_plasticity=False)
        clear()

    def test_invoke_compile(self):
        """
        This test case just invoke the compilation process for the above
        defined network definition.
        """
        pass


class test_StructuralPlasticityEnvironment(unittest.TestCase):
    """
    This class tests the *Structural Plasticity* feature, which can optinally
    be enabled.

    This feature allows the user to manually manipulate *Dentrite* objects by
    adding or removing synapses within them. Both functions *prune_synapse()*
    and *create_synapse()* are tested.

    These functions are called from Python environment code.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        setup(structural_plasticity=True)

        neuron = Neuron(
            parameters="tau = 10",
            equations="r += 1/tau * t"
        )

        pop1 = Population((8), neuron)

        proj = Projection(
             pre=pop1,
             post=pop1,
             target="exc",
        )

        proj2 = Projection(
             pre=pop1,
             post=pop1,
             target="exc",
        )

        proj.connect_all_to_all(weights=1.0)
        proj2.connect_one_to_one(weights=1.0)

        cls.test_net = Network()
        cls.test_net.add([pop1, proj, proj2])
        cls.test_net.compile(silent=True)

        cls.test_proj = cls.test_net.get(proj)
        cls.test_proj2 = cls.test_net.get(proj2)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset(projections=True, synapses=True)

    @classmethod
    def tearDownClass(cls):
        """
        Remove the structural_plasticity global flag to not interfere with
        further tests.
        """
        setup(structural_plasticity=False)
        clear()

    def test_prune(self):
        """
        First we check if the synapses, which are defined by the
        *connect_all_to_all()* function, exist within a specific *Dendrite*.
        Also all weights of the synapses within the *Dendrite* are checked.

        Then, we delete 3 synapses by calling *prune_synapse()* and call the
        *rank* method on the *Dendrite* to check, if corresponding synapses
        are really missing.

        Once again, we check the *weights* to see, if the size of the array
        fits.

        As we use an all2all on same population, rank 3 is omited.
        """
        self.assertEqual(self.test_proj.dendrite(3).pre_ranks,
                         [0, 1, 2, 4, 5, 6, 7])
        numpy.testing.assert_allclose(self.test_proj.dendrite(3).w,
                                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.test_proj.dendrite(3).prune_synapse(2)
        self.test_proj.dendrite(3).prune_synapse(4)
        self.test_proj.dendrite(3).prune_synapse(6)

        self.assertEqual(self.test_proj.dendrite(3).pre_ranks, [0, 1, 5, 7])
        numpy.testing.assert_allclose(self.test_proj.dendrite(3).w,
                                      [1.0, 1.0, 1.0, 1.0])

    def test_create(self):
        """
        First, we check if there is only one synapse returned by the *rank*
        method called on a specific *Dendrite* like defined in the
        *connect_one_to_one()* function. Additionally, we  check the *weight*
        of that single synapse.

        Then, we create 3 additional synapses by calling *create_synapse()*
        call the *rank* method on the *Dendrite* to check, if corresponding
        synapses are listed.

        Once again, we check the *weights* to see, if the size of the returned
        array fits and the values match the second argument given to
        *create_synapse()*.
        """
        self.assertEqual(self.test_proj2.dendrite(3).pre_ranks, [3])
        numpy.testing.assert_allclose(self.test_proj2.dendrite(3).w, [1.0])

        self.test_proj2.dendrite(3).create_synapse(2, 2.0)
        self.test_proj2.dendrite(3).create_synapse(4, 2.0)
        self.test_proj2.dendrite(3).create_synapse(6, 2.0)

        self.assertEqual(self.test_proj2.dendrite(3).pre_ranks, [2, 3, 4, 6])
        numpy.testing.assert_allclose(self.test_proj2.dendrite(3).w,
                                      [2.0, 1.0, 2.0, 2.0])

    def test_prune_complete_dendrite(self):
        """
        We remove all synapses in a dendrite.
        """
        self.test_proj2.dendrite(5).prune_synapse(5)
