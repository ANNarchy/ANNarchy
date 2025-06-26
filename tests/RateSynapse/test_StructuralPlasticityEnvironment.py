"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
import numpy

from ANNarchy import Network, Neuron, Synapse, Uniform

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
        neuron = Neuron(
            parameters="tau = 10",
            equations="r += 1/tau * t"
        )

        synapse = Synapse(
            parameters="""
                tau = 20.0
                alpha = 0.0
            """
        )

        cls.test_net = Network()
        cls.test_net.config(structural_plasticity=True)

        pop1 = cls.test_net.create(geometry=(10), neuron=neuron)

        cls.test_proj = cls.test_net.connect(
            pre=pop1,
            post=pop1,
            target="exc",
        )
        cls.test_proj.all_to_all(weights=1.0)

        cls.test_proj2 = cls.test_net.connect(
            pre=pop1,
            post=pop1,
            target="exc",
        )
        cls.test_proj2.one_to_one(weights=1.0)

        cls.test_proj3 = cls.test_net.connect(
            pre=pop1,
            post=pop1,
            target="exc",
            synapse=synapse
        )
        cls.test_proj3.one_to_one(weights=1.0)
        cls.test_proj3.alpha = Uniform(0.1,1)

        cls.test_net.compile(silent=True)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset(projections=True, synapses=True)

    def test_prune_single_synapse(self):
        """
        First we check if the synapses, which are defined by the
        `all_to_all()` function, exist within a specific `Dendrite`.
        Also all weights of the synapses within the *Dendrite* are checked.

        Then, we delete 3 synapses by calling *prune_synapse()* and call the
        *rank* method on the *Dendrite* to check, if corresponding synapses
        are really missing.

        Once again, we check the *weights* to see, if the size of the array
        fits.

        Please note, as we use an all2all on same population, rank 3 is omited.
        """
        # check all-to-all pattern
        self.assertEqual(self.test_proj.dendrite(3).pre_ranks, [0,1,2,4,5,6,7,8,9])
        numpy.testing.assert_allclose(self.test_proj.dendrite(3).w, [1.0] * (self.test_proj.pre.size-1))

        # Remove some synapses
        self.test_proj.dendrite(3).prune_synapse(2)
        self.test_proj.dendrite(3).prune_synapse(4)
        
        # Rank-order is not important
        self.test_proj.dendrite(3).prune_synapse(8)
        self.test_proj.dendrite(3).prune_synapse(6)

        # Check correctness
        self.assertEqual(self.test_proj.dendrite(3).pre_ranks, [0, 1, 5, 7, 9])
        numpy.testing.assert_allclose(self.test_proj.dendrite(3).w,
                                      [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_prune_multiple_synapses(self):
        """
        First we check if the synapses, which are defined by the
        *all_to_all()* function, exist within a specific *Dendrite*.
        Also all weights of the synapses within the *Dendrite* are checked.

        Then, we delete 3 synapses by calling *prune_synapse()* and call the
        *rank* method on the *Dendrite* to check, if corresponding synapses
        are really missing.

        Once again, we check the *weights* to see, if the size of the array
        fits.

        Please note, as we use an all2all on same population, rank 3 is omited.
        """
        self.assertEqual(self.test_proj.dendrite(3).pre_ranks,
                         [0, 1, 2, 4, 5, 6, 7, 8, 9])
        numpy.testing.assert_allclose(self.test_proj.dendrite(3).w,
                                      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.test_proj.dendrite(3).prune_synapses([2, 4, 6])
        self.test_proj.dendrite(3).prune_synapses([8, 9])

        self.assertEqual(self.test_proj.dendrite(3).pre_ranks, [0, 1, 5, 7])
        numpy.testing.assert_allclose(self.test_proj.dendrite(3).w,
                                      [1.0, 1.0, 1.0, 1.0])

    def test_create_single_synapse(self):
        """
        First, we check if there is only one synapse returned by the *rank*
        method called on a specific *Dendrite* like defined in the
        *one_to_one()* function. Additionally, we  check the *weight*
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

    def test_create_single_synapse(self):
        """
        First, we check if there is only one synapse returned by the *rank*
        method called on a specific *Dendrite* like defined in the
        *one_to_one()* function. Additionally, we  check the *weight*
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

    def test_create_multiple_default_synapses(self):
        """
        First, we check if there is only one synapse returned by the *rank*
        method called on a specific *Dendrite* like defined in the
        *one_to_one()* function. Additionally, we  check the *weight*
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

        self.test_proj2.dendrite(3).create_synapses([2, 4, 6], [2.0, 3.0, 4.0])

        self.assertEqual(self.test_proj2.dendrite(3).pre_ranks, [2, 3, 4, 6])
        numpy.testing.assert_allclose(self.test_proj2.dendrite(3).w,
                                      [2.0, 1.0, 3.0, 4.0])

    def test_create_multiple_default_synapses_2(self):
        """
        Check for two successive calls, which fill current gaps, to check
        if the sorting is always ensured.
        """
        self.test_proj2.dendrite(4).create_synapses([2, 6], [2.0, 4.0])

        # First Check
        self.assertEqual(self.test_proj2.dendrite(4).pre_ranks, [2, 4, 6])
        numpy.testing.assert_allclose(self.test_proj2.dendrite(4).w,
                                      [2.0, 1.0, 4.0])
        
        # Fill some gaps
        self.test_proj2.dendrite(4).create_synapses([3,5,7], [3.0, 5.0, 7.0])

        # Check again
        self.assertEqual(self.test_proj2.dendrite(4).pre_ranks, [2, 3, 4, 5, 6, 7])

    def test_create_multiple_modified_synapses(self):
        """
        Contrary to *test_create_multiple_default_synapses*, this projection uses
        a synapse type with additional parameters.
        """
        # initial state
        self.assertEqual(self.test_proj3.dendrite(3).tau, [20.0])

        # add 3 synapses
        self.test_proj3.dendrite(3).create_synapses([2, 4, 6], [2.0, 3.0, 4.0])

        # check non-default values
        self.assertEqual(self.test_proj3.dendrite(3).tau, [20.0, 20.0, 20.0, 20.0])

        # Check if alpha defined as Uniform [0.1, 1) is drawn correctly
        rand_val = numpy.array(self.test_proj3.dendrite(3).alpha)
        self.assertTrue( (rand_val >= 0.1).all() )
        self.assertTrue( (rand_val < 1.0).all() )

    def test_prune_complete_dendrite(self):
        """
        We remove all synapses in a dendrite. The connectivity was constructed
        based on one-to-one so only one synapse exists.
        """
        self.test_proj2.dendrite(5).prune_synapse(5)
