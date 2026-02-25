"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Neuron, Synapse, Network, Izhikevich, Uniform


class test_PreSpike:
    """
    This class tests the functionality of neurons with a defined *pre-spike*
    equations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        # Each time step emit an event
        SpkNeuron = Neuron(parameters="v = 1", equations="mp = g_exc", spike="v > 0")

        # decrease weight until zero.
        BoundedSynapse = Synapse(
            parameters="g = 1.0",
            pre_spike="""
                w -= 1.0 : min=0.0
            """,
        )

        cls._network = Network()

        pop = cls._network.create(geometry=3, neuron=SpkNeuron)

        cls.test_proj = cls._network.connect(pop, pop, "exc", synapse=BoundedSynapse)
        cls.test_proj.all_to_all(
            5.0, storage_format=cls.storage_format, storage_order=cls.storage_order
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
        In our *setUp()* method we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_w(self):
        """
        Test if the weight value is decreased and check the boundary.
        """
        # is correctly initialized
        numpy.testing.assert_allclose(self.test_proj.dendrite(0).w, [5.0, 5.0])

        # w should has decreased
        self._network.simulate(5)
        numpy.testing.assert_allclose(self.test_proj.dendrite(0).w, [1.0, 1.0])

        # w should not decrease further
        self._network.simulate(5)
        numpy.testing.assert_allclose(self.test_proj.dendrite(0).w, [0.0, 0.0])


class test_PostSpike:
    """
    This class tests the functionality of neurons with a defined
    *post_spike* equations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        # Each time step emit an event
        SpkNeuron = Neuron(parameters="v = 1", equations="mp = g_exc", spike="v > 0")

        # increase weight towards a limit
        BoundedSynapse = Synapse(
            parameters="""
                constant = 1.0
            """,
            post_spike="""
                w += 1.0 * constant : max=10.0
            """,
        )

        cls._network = Network()

        pop = cls._network.create(geometry=3, neuron=SpkNeuron)

        cls.test_proj = cls._network.connect(pop, pop, "exc", synapse=BoundedSynapse)
        cls.test_proj.all_to_all(
            5.0, storage_format=cls.storage_format, storage_order=cls.storage_order
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
        In our *setUp()* method we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_w(self):
        """
        Test if the weight value is decreased and check the boundary.
        """
        # is correctly initialized
        numpy.testing.assert_allclose(self.test_proj.dendrite(0).w, [5.0, 5.0])

        # w should have increased
        self._network.simulate(5)
        numpy.testing.assert_allclose(self.test_proj.dendrite(0).w, [10.0, 10.0])

        # w should not increase further
        self._network.simulate(5)
        numpy.testing.assert_allclose(self.test_proj.dendrite(0).w, [10.0, 10.0])


class test_TimeDependentUpdate:
    """
    Whereas the other two classes test only the update using local variables,
    the update rules in this class uses the time points of pre- and post-synaptic event
    """

    @classmethod
    def setUpClass(cls):
        STDP = Synapse(
            parameters="""
                tau_pre = 10.0 : projection
                tau_post = 10.0 : projection
                cApre = 0.01 : projection
                cApost = 0.0105 : projection
                wmax = 0.01 : projection
            """,
            pre_spike="""
                g_target += w
                w = clip(w - cApost * exp((t_post - t)/tau_post) , 0.0 , wmax)
            """,
            post_spike="""
                w = clip(w + cApre * exp((t_pre - t)/tau_pre) , 0.0 , wmax)
            """,
        )

        cls._network = Network()

        pop = cls._network.create(geometry=100, neuron=Izhikevich)

        proj = cls._network.connect(pop, pop, "exc", STDP)
        proj.all_to_all(
            Uniform(-1.0, 1.0),
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
        In our *setUp()* method we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_invoke_compile(self):
        """
        Test if the compilation succeeds.
        """
        pass
