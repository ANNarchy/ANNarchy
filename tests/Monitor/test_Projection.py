"""

    test_Projection.py

    This file is part of ANNarchy.

    Copyright (C) 2024-25 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
from ANNarchy import Network, Neuron, Synapse

class test_ProjectionMonitor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        simple_pre_neuron = Neuron(
            equations="r = t"
        )

        simple_post_neuron = Neuron(
            equations="r = 10 - t"
        )

        simple_synapse = Synapse(
            equations="y = pre.r * post.r * w"
        )

        cls._network = Network()

        p0 = cls._network.create(geometry=2, neuron=simple_pre_neuron)
        p1 = cls._network.create(geometry=1, neuron=simple_post_neuron)

        proj = cls._network.connect(p0, p1, "exc", simple_synapse)
        proj.connect_all_to_all(0.5)

        cls._mon_m = cls._network.monitor(proj, ['y'])
        
        cls._network.compile(silent=True)

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

    def tearDown(self):
        """
        Since all tests are independent, after every test we use the *get()*
        method for every monotor to clear all recordings.
        """
        self._mon_m.get()
    
    def test_y_sim_10(self):
        """
        We compute a variable *y*, which changes over time dependent on the
        pre- and post-synaptic state variables. This test shows if the recorded
        values of *y* are as expected.
        """
        self._network.simulate(10)

        data_m = self._mon_m.get()['y']
        target_m = numpy.array(
            [[[0.0, 0.0]],
            [[4.5, 4.5]],
            [[8.0, 8.0]],
            [[10.5, 10.5]],
            [[12.0, 12.0]],
            [[12.5, 12.5]],
            [[12.0, 12.0]],
            [[10.5, 10.5]],
            [[8.0, 8.0]],
            [[4.5, 4.5]]], dtype=object
        )
        
        equal = True
        for t_idx in range(len(data_m)):
            for neur_idx in range(len(data_m[t_idx])):
                tmp_x = numpy.array(target_m[t_idx][neur_idx], dtype=float)
                tmp_y = numpy.array(data_m[t_idx][neur_idx], dtype=float)
                if not numpy.allclose(tmp_x, tmp_y):
                    equal = False

        numpy.testing.assert_equal(equal, True)