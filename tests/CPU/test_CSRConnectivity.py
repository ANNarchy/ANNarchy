"""

    test_CSRConnectivity.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import Izhikevich, Population, Projection, Network

pop1 = Population((3, 3), Izhikevich)
pop2 = Population((3, 3), Izhikevich)

proj = Projection(
    pre = pop1,
    post = pop2,
    target = "exc",
)

proj.connect_all_to_all(weights = 0.1, storage_format="csr")

# TODO: PopulationViews

class test_CSRConnectivity(unittest.TestCase):
    """
    This class tests the functionality of the connectivity patterns within *Projections*.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([pop1, pop2, proj])
        self.test_net.compile(silent=True, debug_build=True, clean=True)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before every test.
        """
        self.test_net.reset()

    def test_all_to_all(self):
        """
        Tests the *all_to_all* connectivity pattern, in which every pre-synaptic neuron
        is connected to every post-synaptic neuron.

        We test correctness of ranks and weight values.
        """
        tmp = self.test_net.get(proj)

        self.assertEqual(tmp.dendrite(3).rank, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(numpy.allclose(tmp.dendrite(3).w, numpy.ones((8, 1)) * 0.1))
