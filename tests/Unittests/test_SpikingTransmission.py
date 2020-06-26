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

class test_LILConnectivity(unittest.TestCase):
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
                g_exc1 = 0
                g_exc2 = 0
                g_exc3 = 0
            """,
            spike = "g_exc1>30"
        )

        # simple in/out populations
        in_pop = Population(5, neuron=simple_emit)
        out_pop = Population(2, neuron=simple_recv)

        # create the projections for the test cases (TC)
        # TC: no delay
        proj = Projection(pre=in_pop, post=out_pop, target="exc1")
        proj.connect_all_to_all(weights=1.0)
        # TC: uniform delay
        proj_u = Projection(pre=in_pop, post=out_pop, target="exc2")
        proj_u.connect_all_to_all(weights=1.0, delays=2.0)
        # TC: non-uniform delay
        proj_nu = Projection(pre=in_pop, post=out_pop, target="exc3")
        proj_nu.connect_all_to_all(weights=1.0, delays=Uniform(2,10))

        # Monitor to record the currents
        m = Monitor(out_pop, ["g_exc1", "g_exc2", "g_exc3"])

        # build network and store required object
        # instances
        net = Network(everything=True)
        cls.test_net = net
        cls.test_net.compile(silent=True)
        cls.test_g_exc_m = net.get(m)
        cls.test_proj = net.get(proj_nu)

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
        g_exc_data = self.test_g_exc_m.get('g_exc1')
        self.assertTrue( numpy.allclose( g_exc_data, [[0., 0.], [0., 0.], [5., 5.], [0., 0.], [0., 0.]] ) )

    def test_uniform_delay(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        self.test_net.simulate(5)
        g_exc_data = self.test_g_exc_m.get('g_exc2')
        #The spikes are emitted at t==1 and 2 ms delay so the g_exc should be increased in
        #t == 3 (1ms delay is always). And then again 0, as we reset g_exc.
        self.assertTrue( numpy.allclose( g_exc_data, [[0., 0.], [0., 0.], [0., 0.], [5., 5.], [0., 0.]] ) )

    def test_nonuniform_delay(self):
        """
        Test the receiving of spikes emitted at t == 1
        """
        self.test_proj._set_delay([[1.0, 2.0, 3.0, 2.0, 1.0], [3.0, 2.0, 1.0, 2.0, 3.0]])
        self.test_net.simulate(5)
        g_exc_data = self.test_g_exc_m.get('g_exc3')
        # 1st neuron gets 2 events at t==2, 2 events at t==3 and 1 event at t==4
        # 2nd neuron gets 1 event at t==2, 2 events at t==3, and 2 evets at t==4
        self.assertTrue( numpy.allclose( g_exc_data, [[0., 0.], [0., 0.], [2., 1.], [2., 2.], [1., 2.]] ) )
