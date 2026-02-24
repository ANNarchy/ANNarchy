"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Neuron, Synapse, Network
from ANNarchy.intern.Messages import ANNarchyException


class test_DendriteDefaultSynapse(unittest.TestCase):
    """
    This class tests the *Dendrite* object, which gathers all synapses
    belonging to a post-synaptic neuron in a *Projection*:

        * access to parameters
        * the *rank* method
        * the *size* method

    This test class considers the default synapse which contains only a
    synaptic weight *w*.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(parameters="tau = 10", equations="r += 1/tau * t")

        neuron2 = Neuron(
            parameters="tau = 10: population", equations="r += 1/tau * t: init = 1.0"
        )

        cls._network = Network()
        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=8, neuron=neuron2)

        cls._proj = cls._network.connect(pre=pop1, post=pop2, target="exc")
        cls._proj.all_to_all(
            weights=1.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
            force_multiple_weights=True,
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
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_none(self):
        """
        If a non-existent *Dendrite* is accessed, an error should be thrown.
        This is tested here.
        """
        with self.assertRaises(ANNarchyException) as cm:
            d = self._proj.dendrite(14)

    def test_pre_ranks(self):
        """
        Tests the *pre_ranks* method, which returns the ranks of the
        pre-synaptic neurons belonging to the accessed *Dendrite*.
        """
        self.assertEqual(self._proj.dendrite(5).pre_ranks, [0, 1, 2, 3, 4])

    def test_dendrite_size(self):
        """
        Tests the *size* method, which returns the number of pre-synaptic
        neurons belonging to the accessed *Dendrite*.
        """
        self.assertEqual(self._proj.dendrite(3).size, 5)

    def test_get_dendrite_weights(self):
        """
        Tests the direct access of the parameter *w* (weights) of a *Dendrite*.
        """
        numpy.testing.assert_allclose(
            self._proj.dendrite(7).w, [1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_set_weights(self):
        """
        Tests the setting of the parameter *w* (weights) of a *Dendrite*.
        """
        self._proj.dendrite(6).w = 2.0
        numpy.testing.assert_allclose(
            self._proj.dendrite(6).w, [2.0, 2.0, 2.0, 2.0, 2.0]
        )

    def test_set_weights_2(self):
        """
        Tests the setting of the parameter *w* (weights) of a specific synapse
        in a *Dendrite*.
        """
        self._proj.dendrite(6)[2].w = 3.0
        numpy.testing.assert_allclose(
            self._proj.dendrite(6).w, [2.0, 2.0, 3.0, 2.0, 2.0]
        )


class test_DendriteModifiedSynapse(unittest.TestCase):
    """
    This class tests the *Dendrite* object, which gathers all synapses
    belonging to a post-synaptic neuron in a *Projection*:

        * access to parameters
        * the *rank* method
        * the *size* method

    In this case, we modify the synapse by adding an equation and two
    parameters.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(parameters="tau = 10", equations="r += 1/tau * t")

        neuron2 = Neuron(
            parameters="tau = 10: population", equations="r += 1/tau * t: init = 1.0"
        )

        Oja = Synapse(
            parameters="""
                tau = 5000.0 : postsynaptic
                alpha = 8.0
            """,
            equations="""
                tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w
            """,
        )

        cls._network = Network()

        pop1 = cls._network.create(geometry=5, neuron=neuron)
        pop2 = cls._network.create(geometry=8, neuron=neuron2)

        cls._proj = cls._network.connect(pre=pop1, post=pop2, target="exc", synapse=Oja)

        cls._proj.all_to_all(
            weights=1.0,
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
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_none(self):
        """
        If a non-existent *Dendrite* is accessed, an error should be thrown.
        This is tested here.
        """
        with self.assertRaises(ANNarchyException) as cm:
            d = self._proj.dendrite(14)

    def test_pre_ranks(self):
        """
        Tests the *pre_ranks* method, which returns the ranks of the
        pre-synaptic neurons belonging to the accessed *Dendrite*.
        """
        self.assertEqual(self._proj.dendrite(5).pre_ranks, [0, 1, 2, 3, 4])

    def test_dendrite_size(self):
        """
        Tests the *size* method, which returns the number of pre-synaptic
        neurons belonging to the accessed *Dendrite*.
        """
        self.assertEqual(self._proj.dendrite(3).size, 5)

    def test_get_dendrite_tau(self):
        """
        Tests the direct access of the parameter *tau* of a *Dendrite*.
        """
        numpy.testing.assert_allclose(self._proj.dendrite(1).tau, 5000.0)

    def test_get_dendrite_alpha(self):
        """
        Tests the direct access of the variable *alpha* of a *Dendrite*.
        """
        numpy.testing.assert_allclose(
            self._proj.dendrite(0).alpha, [8.0, 8.0, 8.0, 8.0, 8.0]
        )

    def test_get_dendrite_weights(self):
        """
        Tests the direct access of the parameter *w* (weights) of a *Dendrite*.
        """
        numpy.testing.assert_allclose(
            self._proj.dendrite(7).w, [1.0, 1.0, 1.0, 1.0, 1.0]
        )

    def test_set_tau(self):
        """
        Tests the setting of the parameter *tau* for the whole *Projection*
        through a single value.
        """
        self._proj.tau = 6000.0
        numpy.testing.assert_allclose(self._proj.dendrite(0).tau, 6000.0)

    def test_set_tau_2(self):
        """
        Tests the setting of the parameter *tau* for a single dendrite with a
        single value.
        """
        old_value = self._proj.tau
        old_value[1] = 7000.0

        self._proj.dendrite(1).tau = 7000.0
        numpy.testing.assert_allclose(self._proj.dendrite(1).tau, 7000.0)
        numpy.testing.assert_allclose(self._proj.tau, old_value)

    def test_set_alpha(self):
        """
        Tests the setting of the parameter *alpha* of a *Dendrite*.
        """
        self._proj.dendrite(4).alpha = 9.0
        numpy.testing.assert_allclose(
            self._proj.dendrite(4).alpha, [9.0, 9.0, 9.0, 9.0, 9.0]
        )

    def test_set_alpha_2(self):
        """
        Tests the setting of the parameter *alpha* of a specific synapse in a
        *Dendrite*.
        """
        self._proj.dendrite(4)[1].alpha = 10.0
        numpy.testing.assert_allclose(
            self._proj.dendrite(4).alpha, [9.0, 10.0, 9.0, 9.0, 9.0]
        )

    def test_set_weights(self):
        """
        Tests the setting of the parameter *w* (weights) of a *Dendrite*.
        """
        self._proj.dendrite(6).w = 2.0
        numpy.testing.assert_allclose(
            self._proj.dendrite(6).w, [2.0, 2.0, 2.0, 2.0, 2.0]
        )

    def test_set_weights_2(self):
        """
        Tests the setting of the parameter *w* (weights) of a specific synapse
        in a *Dendrite*.
        """
        self._proj.dendrite(6)[2].w = 3.0
        numpy.testing.assert_allclose(
            self._proj.dendrite(6).w, [2.0, 2.0, 3.0, 2.0, 2.0]
        )

    def test_set_with_dict(self):
        """
        Test the setting of attributes using a dictionary.
        """
        new_value = self._proj.tau
        new_value[1] = 7000.0

        update = dict({"tau": 7000})
        self._proj.dendrite(1).set(update)
        numpy.testing.assert_allclose(self._proj.tau, new_value)

    def test_get_by_name(self):
        """
        Test the retrieval of an attribute by the name.
        """
        val = self._proj.dendrite(1).get("tau")
        numpy.testing.assert_allclose(val, 5000.0)
