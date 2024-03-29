"""

    test_IndividualNeuron.py

    This file is part of ANNarchy.

    Copyright (C) 2023  Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import clear, Network, Neuron, Population

neuron = Neuron(
    parameters = "tau = 10",
    equations="r += 1/tau * t"
)

pop1 = Population((8, 8), neuron)

class test_IndividualNeuron(unittest.TestCase):
    """
    This class tests the functionality of the *IndividualNeuron* object, which
    hold references to one specific neuron of a *Population*.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        cls.test_net = Network()
        cls.test_net.add([pop1])
        cls.test_net.compile(silent=True)

        cls.net_pop1 = cls.test_net.get(pop1)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net
        clear()

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before
        every test.
        """
        self.test_net.reset()

    def test_get_r(self):
        """
        Tests the direct access of the variable *r* of a *IndividualNeuron*
        object.
        """
        numpy.testing.assert_allclose((self.net_pop1[2, 2] +
                                       self.net_pop1[3, 3] +
                                       self.net_pop1[4, 4]).r, [0.0, 0.0, 0.0])

    def test_set_r(self):
        """
        Tests the setting of *r* through direct access.
        """
        (self.net_pop1[2, 2] + self.net_pop1[3, 3] + self.net_pop1[4, 4]).r = 1.0
        numpy.testing.assert_allclose((self.net_pop1[2, 2] +
                                       self.net_pop1[3, 3] +
                                       self.net_pop1[4, 4]).r, [1.0, 1.0, 1.0])

    def test_rank_assignment(self):
        """
        Test the correct assignment of ranks
        """
        view = self.net_pop1[2, 4]
        numpy.testing.assert_equal(view.rank, 20)   # 2 * 8 + 4