"""

    test_Dendrite.py

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
        tau = 5000.0 : postsynaptic
        alpha = 8.0
    """,
    equations = """
        tau * dw/dt = pre.r * post.r - alpha * post.r^2 * w
    """
)


pop1 = Population(5, neuron)
pop2 = Population(8, neuron2)


proj = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)

proj.connect_all_to_all(weights = 1.0)

compile(clean=True)


class test_Dendrite(unittest.TestCase):

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        reset()

    def test_none(self):
        """
        tests None object display if non-existent *Dendrite* is accessed
        """
        self.assertEqual(proj.dendrite(14), None)

    def test_rank(self):
        """
        tests correct display of the list of pre-synaptic neuron ranks
        """
        self.assertEqual(proj.dendrite(5).rank, [0, 1, 2, 3, 4])

    def test_dendrite_size(self):
        """
        tests if *Dendrite* size is displayed correctly
        """
        self.assertEqual(proj.dendrite(3).size, 5)

    def test_get_dendrite_tau(self):
        """
        tests if list of tau of pre-synaptic neurons is correctly displayed
        """
        self.assertTrue(numpy.allclose(proj.dendrite(1).tau, 5000.0))

    def test_get_dendrite_tau_2(self):
        """
        tests if list of tau of pre-synaptic neurons is correctly displayed (with a different method)
        """
        self.assertTrue(numpy.allclose(proj.dendrite(1).get('tau'), 5000.0))


    def test_get_dendrite_alpha(self):
        """
        tests if list of alpha of pre-synaptic neurons is correctly displayed
        """
        self.assertTrue(numpy.allclose(proj.dendrite(0).alpha, [8.0, 8.0, 8.0, 8.0, 8.0]))

    def test_get_dendrite_weights(self):
        """
        tests if list of weights is correctly displayed
        """
        self.assertTrue(numpy.allclose(proj.dendrite(7).w, [1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_set_tau(self):
        """
        tests if tau is correcly set
        """
        proj.tau=6000.0
        self.assertTrue(numpy.allclose(proj.dendrite(0).tau, 6000.0))

    def test_set_tau(self):
        """
        tests if tau is correcly set.

        HD (22th Sep. 2015): this test currently fail, it is not yet clear if it is an error or not.
        """
        proj.tau = [5000.0, 6000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0]
        self.assertTrue(numpy.allclose(proj.dendrite(1).tau, 6000.0))

    def test_set_alpha(self):
        """
        tests if alpha is correcly set
        """
        proj.dendrite(4).alpha=9.0
        self.assertTrue(numpy.allclose(proj.dendrite(4).alpha, [9.0, 9.0, 9.0, 9.0, 9.0]))

    def test_set_alpha_2(self):
        """
        tests if alpha is correcly set
        """
        proj.dendrite(4)[1].alpha=10.0
        self.assertTrue(numpy.allclose(proj.dendrite(4).alpha, [9.0, 10.0, 9.0, 9.0, 9.0]))

    def test_set_weights(self):
        """
        tests if weights are correcly set
        """
        proj.dendrite(6).w=2.0
        self.assertTrue(numpy.allclose(proj.dendrite(6).w, [2.0, 2.0, 2.0, 2.0, 2.0]))

    def test_set_weights_2(self):
        """
        tests if weights are correcly set
        """
        proj.dendrite(6)[2].w=3.0
        self.assertTrue(numpy.allclose(proj.dendrite(6).w, [2.0, 2.0, 3.0, 2.0, 2.0]))
