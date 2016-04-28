"""

    test_Connectivity.py

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

from ANNarchy import *

neuron = Neuron(
    equations="r = 1"
)

neuron2 = Neuron(
    equations="r = sum(exc)"
)

pop1 = Population((3, 3), neuron)
pop2 = Population((3, 3), neuron2)

proj1 = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
)

proj2 = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
)

proj1.connect_one_to_one(weights = 0.1)
proj2.connect_all_to_all(weights = 0.1)

compile(clean=True)

#TODO: one2one, all2all for PopulationViews

class test_Connectivity(unittest.TestCase):
    """
    This class tests the functionality of the connectivity patterns within *Projections*.
    """
    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before every test.
        """
        reset()

    def test_one_to_one(self):
        """
        Tests the *one_to_one* connectivity pattern, in which every pre-synaptic neuron
        is connected to its ranked equivalent post-synaptic neuron.

        We test correctness of ranks and weight values.
        """
        self.assertEqual(proj1.dendrite(3).rank, [3])
        self.assertTrue(numpy.allclose(proj1.dendrite(3).w, [0.1]))

    def test_all_to_all(self):
        """
        Tests the *all_to_all* connectivity pattern, in which every pre-synaptic neuron
        is connected to every post-synaptic neuron.

        We test correctness of ranks and weight values.
        """
        self.assertEqual(proj2.dendrite(3).rank, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(numpy.allclose(proj2.dendrite(3).w, np.ones((8,1)) * 0.1))
