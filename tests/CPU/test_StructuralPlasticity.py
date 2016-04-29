"""

    test_StructuralPlasticity.py

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

setup(structural_plasticity=True)

neuron = Neuron(
    parameters = "tau = 10",
    equations="r += 1/tau * t"
)

neuron2 = Neuron(
    parameters = "tau = 10: population",
    equations="r += 1/tau * t: init = 1.0"
)

Oja = Synapse(
    parameters="""
        tau = 5000.0
        alpha = 8.0
    """,
    equations = """
        r = t
    """
)


pop1 = Population((8), neuron)
pop2 = Population((8), neuron2)


proj = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)

proj2 = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)

proj.connect_all_to_all(weights = 1.0)
proj2.connect_one_to_one(weights = 1.0)

compile(clean=True, silent=True)


class test_StructuralPlasticity(unittest.TestCase):
    """
    This class tests the *Structural Plasticity* feature, which can optinally be enabled.
    This feature allows the user to manually manipulate *Dentrite* objects by adding or removing synapses within them.
    Both functions *prune_synapse()* and *create_synapse()* are tested.
    """
    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        reset()

    def test_prune(self):
        """
        First we check if the synapses, which are defined by the *connect_all_to_all()* function, exist within a specific *Dendrite*.
        Also all weights of the synapses within the *Dendrite* are checked.
        Then, we delete 3 synapses by calling *prune_synapse()* and call the *rank* method on the *Dendrite* to check, if corresponding synapses are really missing.
        Once again, we check the *weights* to see, if the size of the array fits.
        """

        self.assertEqual(proj.dendrite(3).rank, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(numpy.allclose(proj.dendrite(3).w, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        proj.dendrite(3).prune_synapse(2)
        proj.dendrite(3).prune_synapse(4)
        proj.dendrite(3).prune_synapse(6)

        self.assertEqual(proj.dendrite(3).rank, [0, 1, 3, 5, 7])
        self.assertTrue(numpy.allclose(proj.dendrite(3).w, [1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_create(self):
        """
        First, we check if there is only one synapse returned by the *rank* method called on a specific *Dendrite* like defined in the *connect_one_to_one()* function.
        We also check the *weight* of that single synapse.
        Then, we create 3 additional synapses by calling *create_synapse()* call the *rank* method on the *Dendrite* to check, if corresponding synapses are listed.
        Once again, we check the *weights* to see, if the size of the returned array fits and the values match the second argument given to *create_synapse()*.
        """
        self.assertEqual(proj2.dendrite(3).rank, [3])
        self.assertTrue(numpy.allclose(proj2.dendrite(3).w, [1.0]))

        proj2.dendrite(3).create_synapse(2, 2.0)
        proj2.dendrite(3).create_synapse(4, 2.0)
        proj2.dendrite(3).create_synapse(6, 2.0)

        self.assertEqual(proj2.dendrite(3).rank, [2, 3, 4, 6])
        self.assertTrue(numpy.allclose(proj2.dendrite(3).w, [2.0, 1.0, 2.0, 2.0]))

if __name__ == '__main__':
    unittest.main()
