"""

    test_ProjectionCUDA.py

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
setup(paradigm="cuda")


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


pop1 = Population((8, 8), neuron)
pop2 = Population((8, 8), neuron2)


proj = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)

proj.connect_all_to_all(weights = 1.0)

compile(clean=True)

class test_Projection(unittest.TestCase):




    def test_get_tau(self):
        """
        test access to parameter tau
        """
        self.assertTrue(numpy.allclose(proj.tau, 5000.0))

    def test_get_tau_2(self):
        """
        test access to parameter tau with different method
        """
        self.assertTrue(numpy.allclose(proj.get('tau'), 5000.0))

    def test_get_alpha(self):
        """
        test access to parameter alpha
        """
        self.assertTrue(numpy.allclose(proj.alpha, 8.0))

    def test_get_alpha_2(self):
        """
        test access to parameter alpha with different method
        """
        self.assertTrue(numpy.allclose(proj.get('alpha'), 8.0))

    def test_get_size(self):
        """
        test to get size
        """
        self.assertEqual(proj.size, 64)

    def test_get_post_ranks(self):
        """
        test to get post_ranks
        """
        self.assertEqual(proj.post_ranks, range(64))
