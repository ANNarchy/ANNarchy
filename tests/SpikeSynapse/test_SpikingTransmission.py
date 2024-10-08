"""
    test_SpikingTransmission.py

    This file is part of ANNarchy.

    Copyright (C) 2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    Copyright (C) 2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>, Alex Schwarz

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
from ANNarchy import (clear, DiscreteUniform, Monitor, Network, Neuron,
                      Population, Projection)
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

        # simple in/out populations
        in_pop = Population(5, neuron=simple_emit)
        out_pop = Population(2, neuron=simple_recv)

        # create the projections for the test cases (TC)
        # TC: no delay
        proj = Projection(pre=in_pop, post=out_pop, target="exc")
        proj.connect_all_to_all(weights=1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        # Monitor to record the currents
        m = Monitor(out_pop, ["g_exc"])

        # build network and store required object
        # instances
        net = Network()
        net.add([in_pop, out_pop, proj, m])
        cls.test_net = net
        cls.test_net.compile(silent=True)
        cls.test_g_exc_m = net.get(m)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net
        clear()

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        # back to initial values
        self.test_net.reset(populations=True, projections=True)
        # clear monitors must be done seperate
        self.test_g_exc_m.get()

    def test_non_delay(self):
        """
        The spikes are emitted at t==1 so the g_exc should be increased in
        t == 2. And then again 0, as we reset g_exc.
        """
        self.test_net.simulate(5)
        g_exc_data = self.test_g_exc_m.get('g_exc')
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

        # simple in/out populations
        in_pop = Population(5, neuron=simple_emit)
        out_pop = Population(2, neuron=simple_recv)

        # TC: uniform delay
        proj = Projection(pre=in_pop, post=out_pop, target="exc")
        proj.connect_all_to_all(weights=1.0, delays=3.0,
                                  storage_format=cls.storage_format,
                                  storage_order=cls.storage_order)

        # Monitor to record the currents
        m = Monitor(out_pop, ["g_exc"])

        # build network and store required object
        # instances
        net = Network()
        net.add([in_pop, out_pop, proj, m])
        cls.test_net = net
        try:
            cls.test_net.compile(silent=True)
            cls.test_g_exc_m = net.get(m)
        except InvalidConfiguration:
            cls.test_net = None
            clear()

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        if cls.test_net is not None:
            del cls.test_net
            clear()

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        if self.test_net is not None:
            # back to initial values
            self.test_net.reset(populations=True, projections=True)
            # clear monitors must be done seperate
            self.test_g_exc_m.get()

    def test_uniform_delay(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        if self.test_net is not None:
            self.test_net.simulate(6)
            g_exc_data = self.test_g_exc_m.get('g_exc')
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

        # simple in/out populations
        in_pop = Population(5, neuron=simple_emit)
        out_pop = Population(2, neuron=simple_recv)

        # TC: non-uniform delay
        proj = Projection(pre=in_pop, post=out_pop, target="exc")
        proj.connect_all_to_all(weights=1.0, delays=DiscreteUniform(2,10),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        # Monitor to record the currents
        m = Monitor(out_pop, ["g_exc"])

        # build network and store required object
        # instances
        net = Network()
        net.add([in_pop, out_pop, proj, m])
        cls.test_net = net
        cls.test_net.compile(silent=True)
        cls.test_g_exc_m = net.get(m)
        cls.test_proj = net.get(proj)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net
        clear()

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        # back to initial values
        self.test_net.reset(populations=True, projections=True)
        # clear monitors must be done seperate
        self.test_g_exc_m.get()

    def test_nonuniform_delay(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        self.test_proj._set_delay([[1.0, 2.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 2.0, 3.0]])
        self.test_net.simulate(5)
        g_exc_data = self.test_g_exc_m.get('g_exc')
        # 1st neuron gets 2 events at t==2, 2 events at t==3 and 1 event at t==4
        # 2nd neuron gets 1 event at t==2, 2 events at t==3, and 2 evets at t==4
        numpy.testing.assert_allclose(g_exc_data, [[0., 0.], [0., 0.], [2., 1.], [2., 2.], [1., 2.]])
