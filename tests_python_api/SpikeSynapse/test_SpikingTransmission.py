"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import DiscreteUniform, Network, Neuron
from ANNarchy.intern.Messages import InvalidConfiguration

class test_SpikeTransmissionNoDelay():
    """
    A pre-synaptic event should increase the conductance of the post-
    synaptic neuron by the value *w* in default case.
    """
    @classmethod
    def setUpClass(cls):
        """
        Build up the network
        """
        simple_emit = Neuron(
            spike = "t==1",
        )
        simple_recv = Neuron(
            equations = """
                g_exc = 0
            """,
            spike = "g_exc>30"
        )

        cls._network = Network()

        # simple in/out populations
        in_pop = cls._network.create(geometry=5, neuron=simple_emit)
        out_pop = cls._network.create(geometry=2, neuron=simple_recv)

        # create the projections for the test cases (TC)
        # TC: no delay
        proj = cls._network.connect(pre=in_pop, post=out_pop, target="exc")
        proj.all_to_all(weights=1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        # Monitor to record the currents
        cls._mon_g_exc = cls._network.monitor(out_pop, ["g_exc"])

        # build network and store required object
        # instances
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

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
        # back to initial values
        self._network.reset(populations=True, projections=True)
        # clear monitors must be done seperate
        self._mon_g_exc.get()

    def test_non_delay(self):
        """
        The spikes are emitted at t==1 so the g_exc should be increased in
        t == 2. And then again 0, as we reset g_exc.
        """
        self._network.simulate(5)
        g_exc_data = self._mon_g_exc.get('g_exc')
        numpy.testing.assert_allclose(g_exc_data, [[0., 0.], [0., 0.], [5., 5.], [0., 0.], [0., 0.]])


class test_SpikeTransmissionUniformDelay():
    """
    A pre-synaptic event should increase the conductance of the post-
    synaptic neuron by the value *w* in default case.

    TODO: possible test-case
    - setting new delays and compute psp
    - other patterns? this might be useful in future, when we have cpp-side pattern generators
    """
    @classmethod
    def setUpClass(cls):
        """
        Build up the network
        """
        simple_emit = Neuron(
            spike = "t==1",
        )
        simple_recv = Neuron(
            equations = """
                g_exc = 0
            """,
            spike = "g_exc>30"
        )

        cls._network = Network()

        # simple in/out populations
        in_pop = cls._network.create(geometry=5, neuron=simple_emit)
        out_pop = cls._network.create(geometry=2, neuron=simple_recv)

        # TC: uniform delay
        proj = cls._network.connect(pre=in_pop, post=out_pop, target="exc")
        proj.all_to_all(
            weights=1.0, delays=3.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        # Monitor to record the currents
        cls._mon_g_exc = cls._network.monitor(out_pop, ["g_exc"])

        # build network and store required object
        # instances
        try:
            cls._network.compile(silent=True, directory=TARGET_FOLDER)
        except InvalidConfiguration:
            cls._network = None

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        if cls._network is not None:
            del cls._network

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        if self._network is not None:
            # back to initial values
            self._network.reset(populations=True, projections=True)
            # clear monitors must be done seperate
            self._mon_g_exc.get()

    def test_uniform_delay(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        if self._network is not None:
            self._network.simulate(6)
            g_exc_data = self._mon_g_exc.get('g_exc')
            # The spikes are emitted at t==1 and 3 ms delay so the g_exc should be
            # increased in t == 4 (1ms delay is always). And then again 0, as we
            # reset g_exc.
            numpy.testing.assert_allclose(g_exc_data, [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [5., 5.], [0., 0.]] )

class test_SpikeTransmissionNonUniformDelay():
    """
    A pre-synaptic event should increase the conductance of the post-
    synaptic neuron by the value *w* in default case.

    TODO: possible test-case
    - setting new delays and compute psp
    - other patterns? this might be useful in future, when we have cpp-side pattern generators
    """
    @classmethod
    def setUpClass(cls):
        """
        Build up the network
        """
        simple_emit = Neuron(
            spike = "t==1",
        )
        simple_recv = Neuron(
            equations = """
                g_exc = 0
            """,
            spike = "g_exc>30"
        )

        cls._network = Network()

        # simple in/out populations
        in_pop = cls._network.create(geometry=5, neuron=simple_emit)
        out_pop = cls._network.create(geometry=2, neuron=simple_recv)

        # TC: non-uniform delay
        cls.test_proj = cls._network.connect(pre=in_pop, post=out_pop, target="exc")
        cls.test_proj.all_to_all(
            weights=1.0, delays=DiscreteUniform(2,10),
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        # Monitor to record the currents
        cls._mon_g_exc = cls._network.monitor(out_pop, ["g_exc"])

        # build network and store required object
        # instances
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

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
        # back to initial values
        self._network.reset(populations=True, projections=True)
        # clear monitors must be done seperate
        self._mon_g_exc.get()

    def test_nonuniform_delay(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        self.test_proj._set_delay([[1.0, 2.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 2.0, 3.0]])
        self._network.simulate(5)
        g_exc_data = self._mon_g_exc.get('g_exc')
        # 1st neuron gets 2 events at t==2, 2 events at t==3 and 1 event at t==4
        # 2nd neuron gets 1 event at t==2, 2 events at t==3, and 2 evets at t==4
        numpy.testing.assert_allclose(g_exc_data, [[0., 0.], [0., 0.], [2., 1.], [2., 2.], [1., 2.]])
