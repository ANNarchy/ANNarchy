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

from ANNarchy import Network, Neuron, clear

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
        neuron = Neuron(
            parameters = "tau = 10",
            equations="r += 1/tau * t"
        )

        cls._network = Network()
        cls._population = cls._network.create(geometry=(8, 8), neuron=neuron)
        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before
        every test.
        """
        self._network.reset()

    def test_get_r(self):
        """
        Tests the direct access of the variable *r* of a *IndividualNeuron*
        object.
        """
        numpy.testing.assert_allclose((self._population[2, 2] +
                                       self._population[3, 3] +
                                       self._population[4, 4]).r, [0.0, 0.0, 0.0])

    def test_set_r(self):
        """
        Tests the setting of *r* through direct access.
        """
        (self._population[2, 2] + self._population[3, 3] + self._population[4, 4]).r = 1.0
        numpy.testing.assert_allclose((self._population[2, 2] +
                                       self._population[3, 3] +
                                       self._population[4, 4]).r, [1.0, 1.0, 1.0])

    def test_rank_assignment(self):
        """
        Test the correct assignment of ranks
        """
        view = self._population[2, 4]
        numpy.testing.assert_equal(view.rank, 20)   # 2 * 8 + 4