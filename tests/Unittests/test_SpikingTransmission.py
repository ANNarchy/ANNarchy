"""
    test_SpikingTransmission.py

    This file is part of ANNarchy.

    Copyright (C) 2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import Neuron, Population, Projection, Network, Monitor, Uniform

class TestNonDelay(unittest.TestCase):
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
            equations = """g_exc = 0""",
            spike = "g_exc>30"
        )

        in_pop = Population(5, neuron=simple_emit)
        out_pop = Population(2, neuron=simple_recv)
        proj = Projection(pre=in_pop, post=out_pop, target="exc")
        proj.connect_all_to_all(weights=1.0)
        m = Monitor(out_pop, ["g_exc"])

        net = Network(everything=True)

        cls.test_net = net
        cls.test_net.compile()
        cls.test_g_exc_m = net.get(m)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_all_to_all(self):
        """
        The spikes are emitted at t==1 so the g_exc should be increased in
        t == 2. And then again 0, as we reset g_exc.
        """
        self.test_net.simulate(5)

        g_exc_data = self.test_g_exc_m.get('g_exc')
        self.assertTrue( numpy.allclose( g_exc_data, [[0., 0.], [0., 0.], [5., 5.], [0., 0.], [0., 0.]] ) )

class TestUniformDelay(unittest.TestCase):
    """
    A pre-synaptic event should increase the conductance of the post-
    synaptic neuron by the value *w* in default case. In this class,
    we add an uniform delay to the connection.
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
            equations = """g_exc = 0""",
            spike = "g_exc>30"
        )

        in_pop = Population(5, neuron=simple_emit)
        out_pop = Population(2, neuron=simple_recv)
        proj = Projection(pre=in_pop, post=out_pop, target="exc")
        proj.connect_all_to_all(weights=1.0, delays=2.0)
        m = Monitor(out_pop, ["g_exc"])

        net = Network(everything=True)

        cls.test_net = net
        cls.test_net.compile()
        cls.test_g_exc_m = net.get(m)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_all_to_all(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        self.test_net.simulate(5)

        g_exc_data = self.test_g_exc_m.get('g_exc')

        #The spikes are emitted at t==1 and 2 ms delay so the g_exc should be increased in
        #t == 3 (1ms delay is always). And then again 0, as we reset g_exc.
        self.assertTrue( numpy.allclose( g_exc_data, [[0., 0.], [0., 0.], [0., 0.], [5., 5.], [0., 0.]] ) )

class TestNonuniformDelay(unittest.TestCase):
    """
    A pre-synaptic event should increase the conductance of the post-
    synaptic neuron by the value *w* in default case. In this class,
    we add an uniform delay to the connection.
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
            equations = """g_exc = 0""",
            spike = "g_exc>30"
        )

        in_pop = Population(5, neuron=simple_emit)
        out_pop = Population(2, neuron=simple_recv)
        proj = Projection(pre=in_pop, post=out_pop, target="exc")
        proj.connect_all_to_all(weights=1.0, delays=Uniform(2,10))
        m = Monitor(out_pop, ["g_exc"])

        net = Network(everything=True)

        cls.test_net = net
        cls.test_net.compile()
        cls.test_proj = net.get(proj)
        cls.test_g_exc_m = net.get(m)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset()

    def test_all_to_all(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        self.test_proj._set_delay([[1.0, 2.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 2.0, 3.0]])

        self.test_net.simulate(5)

        g_exc_data = self.test_g_exc_m.get('g_exc')

        # 1st neuron gets 2 events at t==2, 2 events at t==3 and 1 event at t==4
        # 2nd neuron gets 1 event at t==2, 2 events at t==3, and 2 evets at t==4
        self.assertTrue( numpy.allclose( g_exc_data, [[0., 0.], [0., 0.], [2., 1.], [2., 2.], [1., 2.]] ) )
