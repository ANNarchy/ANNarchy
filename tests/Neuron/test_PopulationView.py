"""

    test_PopulationView.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    Copyright (C) 2022  Alex Schwarz <alex.schwarz@informatik.tu-chemnitz.de>

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

from ANNarchy import Network, Neuron

neuron = Neuron(
    parameters = "tau = 10",
    equations="r += 1/tau * t"
)

class test_PopulationView(unittest.TestCase):
    """
    This class tests the functionality of the *PopulationView* object, which
    hold references to different neurons of the same *Population*.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
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
        Tests the direct access of the variable *r* of a *PopulationView*
        object.
        """
        numpy.testing.assert_allclose((self._population[:, 2]).r, numpy.zeros(8))

    def test_set_r(self):
        """
        Tests the setting of *r* through direct access.
        """
        self._population[:, 2].r = 1.0
        numpy.testing.assert_allclose(self._population[:, 2].r, numpy.ones(8))

    def test_rank_assignment_column(self):
        """
        Test the correct assignment of ranks of a sliced column
        """
        view = self._population[:, 4]
        numpy.testing.assert_allclose(view.ranks, [4,12,20,28,36,44,52,60])   # row_rank * 8 + 4

    def test_rank_assignment_row(self):
        """
        Test the correct assignment of ranks of a sliced row
        """
        view = self._population[2, :]
        numpy.testing.assert_allclose(view.ranks, [16,17,18,19,20,21,22,23])   # 2 * 8 + column_rank

    def test_index_type(self):
        """
        In ANNarchy 4.7.2 we changed the data type from list to ndarray. This test should uncover
        "old" data definitions.
        """
        view = self._population[2, :]
        numpy.testing.assert_equal(isinstance(view.ranks, numpy.ndarray), True)