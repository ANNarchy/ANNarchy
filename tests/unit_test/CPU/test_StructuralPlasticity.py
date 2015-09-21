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

proj.connect_all_to_all(weights = 1.0)


setup(structural_plasticity=True)
compile(clean=True)


class test_StructuralPlasticity(unittest.TestCase):
       

    def test_delete_create(self):
        """
        checks if *Synapse* is there, deletes it and creates it with a different weight
        """
        
        self.assertEqual(proj.dendrite(3).rank, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(numpy.allclose(proj.dendrite(3).w, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        proj.dendrite(3).prune_synapse(2)
        proj.dendrite(3).prune_synapse(4)
        proj.dendrite(3).prune_synapse(6)

        self.assertEqual(proj.dendrite(3).rank, [0, 1, 3, 5, 7])

        proj.dendrite(3).create_synapse(2, 2.0)
        proj.dendrite(3).create_synapse(4, 2.0)
        proj.dendrite(3).create_synapse(6, 2.0)

        self.assertEqual(proj.dendrite(3).rank, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(numpy.allclose(proj.dendrite(3).w, [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]))
