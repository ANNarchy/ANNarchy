"""

    test_CustomFunc.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
    Julien Vitay <julien.vitay@gmail.com>

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

from ANNarchy import Neuron, Population, Synapse, Projection, Network, add_function

add_function("glob_pos(x) = pos(x)")

class test_CustomFunc(object):
    """
    This class tests the definition of custom functions, they
    can defined on three levels:

        * globally
        * within neurons
        * within synapses
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations = "r = transfer_function(sum(exc), 0.0)",
            functions = "transfer_function(x, t) = if x > t: if x > 2*t : (x - 2*t)^2 else: x - t else: 0."
        )

        neuron2 = Neuron(
            equations = "r = glob_pos(sum(exc))"
        )

        synapse = Synapse(
            equations="w += hebb(pre.r, post.r)",
            functions="hebb(x, y) = x * y"
        )

        pop = Population(10, neuron)
        pop2 = Population(10, neuron2)
        proj = Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, pop2, proj])
        cls.test_net.compile(silent=True)

        cls.net_pop = cls.test_net.get(pop)
        cls.net_proj = cls.test_net.get(proj)

    @classmethod
    def tearDownClass(cls):
        del cls.test_net

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def test_neuron(self):
        """
        Custom func defined within a neuron object, providing numpy.array data.
        """
        self.assertTrue(numpy.allclose(self.net_pop.transfer_function(numpy.array([0., 1., 2., 3.]), numpy.array([2., 2., 2., 2.])), [0, 0, 0, 1]))

    def test_neuron2(self):
        """
        Custom func defined within a neuron object, providing simple lists.
        """
        self.assertTrue(numpy.allclose(self.net_pop.transfer_function([0., 1., 2., 3.], [2., 2., 2., 2.]), [0, 0, 0, 1]))

    def test_synapse(self):
        """
        Custom func defined within a synapse object, providing simple lists.
        """
        self.assertTrue(numpy.allclose(self.net_proj.hebb(numpy.array([0., 1., 2., 3.]), numpy.array([0., 1., 2., 3.])), [0, 1, 4, 9]))

    def test_synapse2(self):
        """
        Custom func defined within a synapse object, providing simple lists.
        """
        self.assertTrue(numpy.allclose(self.net_proj.hebb([0., 1., 2., 3.], [0., 1., 2., 3.]), [0, 1, 4, 9]))
