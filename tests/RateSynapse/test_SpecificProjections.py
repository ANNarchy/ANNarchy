"""

    test_SpecificProjection.py

    This file is part of ANNarchy.

    Copyright (C) 2019 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,

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
import math

from ANNarchy import CurrentInjection, Network, Neuron

class test_CurrentInjection(unittest.TestCase):
    """
    Test the implementation of the specialized projection
    'CurrentInjection'. Based on the example in the documentation.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test. Adapted the example
        from documentation.
        """

        SimpleSpike = Neuron(
            equations="mp=g_exc",
            spike="mp >= 1.0",
            reset=""
        )

        cls.test_net = Network()

        inp = cls.test_net.create(geometry=1, neuron=Neuron(equations="r=sin(t)"))
        cls.output = cls.test_net.create(geometry=1, neuron=SimpleSpike)
        cls.m = cls.test_net.monitor(cls.output, "mp")

        proj = cls.test_net.connect(CurrentInjection(inp, cls.output, 'exc'))
        proj.connect_current()

        cls.test_net.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self.test_net.reset()

    def test_compile(self):
        """
        Enforce compilation of the network.
        """
        pass

    def test_run_one_loop(self):
        """
        Test the membrane potential after one full loop
        """
        self.test_net.simulate(11)

        rec_data = self.m.get("mp")[:,0]
        # there is 1 dt delay between the input and output
        target = [0] + [math.sin(x) for x in range(10)]

        numpy.testing.assert_allclose(rec_data, target)
