"""

    test_Projection.py

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
from scipy import sparse

from ANNarchy import Neuron, Synapse, Population, Projection, Network

class test_Projection(unittest.TestCase):
    """
    Tests the functionality of the *Projection* object using a list-in-list
    representation (currently the default in ANNarchy). We test:

        *access to parameters
        *method to get the ranks of post-synaptic neurons recieving synapses
        *method to get the number of post-synaptic neurons recieving synapses
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        simple = Neuron(
            parameters = "r=0",
        )

        Oja = Synapse(
            parameters="""
                tau = 5000.0
                alpha = 8.0 : postsynaptic
                """,
            equations = """
                dw/dt = -w
                """
        )

        pop1 = Population((8), neuron=simple)
        pop2 = Population((4), neuron=simple)

        # define a sparse matrix
        weight_matrix = sparse.lil_matrix((4,8))
        # HD (01.07.20): its not possible to use slicing here, as it produces
        #                FutureWarnings in scipy/numpy (for version >= 1.17)
        for i in range(8):
            weight_matrix[1, i] = 0.2
        for i in range(2,6):
            weight_matrix[3, i] = 0.5

        # we need to flip the matrix (see 2.9.3.2 in documentation)
        cls.weight_matrix = weight_matrix.T

        # set the pre-defined matrix
        proj = Projection(
            pre = pop1,
            post = pop2,
            target = "exc",
            synapse = Oja
        )
        proj.connect_from_sparse(cls.weight_matrix,
                                 storage_format=cls.storage_format,
                                 storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, proj])
        cls.test_net.compile(silent=True)

        cls.net_proj = cls.test_net.get(proj)

    def setUp(self):
        """
        In our *setUp()* function we reset the network before every test.
        """
        self.test_net.reset()

    def test_get_w(self):
        """
        Test the direct access to the synaptic weight.
        """
        # test row 1 (idx 0) with 8 elements should be 0.2
        numpy.testing.assert_allclose(self.net_proj.w[0], 0.2)

        # test row 3 (idx 1) with 8 elements should be 0.5
        numpy.testing.assert_allclose(self.net_proj.w[1], 0.5)

    def test_get_dendrite_w(self):
        """
        Test the access through dendrite to the synaptic weight.
        """
        # test row 1 with 8 elements should be 0.2
        numpy.testing.assert_allclose(self.net_proj.dendrite(1).w, 0.2)

        # test row 3 with 4 elements should be 0.5
        numpy.testing.assert_allclose(self.net_proj.dendrite(3).w, 0.5)

    def test_get_tau(self):
        """
        Tests the direct access to the parameter *tau* of our *Projection*.
        """
        # test row 1 (idx 0) with 8 elements
        numpy.testing.assert_allclose(self.net_proj.tau[0], 5000.0)
        # test row 3 (idx 1) with 4 elements
        numpy.testing.assert_allclose(self.net_proj.tau[1], 5000.0)

    def test_get_tau_2(self):
        """
        Tests the access to the parameter *tau* of our *Projection* with the *get()* method.
        """
        numpy.testing.assert_allclose(self.net_proj.get('tau')[0], 5000.0)
        numpy.testing.assert_allclose(self.net_proj.get('tau')[1], 5000.0)

    def test_get_alpha(self):
        """
        Tests the direct access to the parameter *alpha* of our *Projection*.
        """
        numpy.testing.assert_allclose(self.net_proj.alpha, 8.0)

    def test_get_alpha_2(self):
        """
        Tests the access to the parameter *alpha* of our *Projection* with the
        *get()* method.
        """
        numpy.testing.assert_allclose(self.net_proj.get('alpha'), 8.0)

    def test_get_size(self):
        """
        Tests the *size* method, which returns the number of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self.net_proj.size, 2)

    def test_get_post_ranks(self):
        """
        Tests the *post_ranks* method, which returns the ranks of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self.net_proj.post_ranks, [1, 3])

class test_SliceProjections(unittest.TestCase):
    """
    Test signal transmission for differently sliced rate-coded PopulationViews.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        preNeuron = Neuron(parameters="r=0.0")
        rNeuron = Neuron(equations="""
            p = sum(prev)
            q = sum(postv)
            r = sum(bothv)
        """)

        pop1 = Population(geometry=8, neuron=preNeuron)
        pop2 = Population(geometry=6, neuron=rNeuron)
        pre_slice = pop1[1:4]
        post_slice = pop2[1:3]

        # see test_pre_view()
        pre_view = Projection(pre_slice, pop2, target="prev")
        pre_view.connect_all_to_all(weights=0.1, storage_format=cls.storage_format, storage_order=cls.storage_order)

        # see test_post_view()
        post_view = Projection(pop1, post_slice, target="postv")
        post_view.connect_all_to_all(weights=0.1, storage_format=cls.storage_format, storage_order=cls.storage_order)

        # see test_both_view()
        both_view = Projection(pre_slice, post_slice, target="bothv")
        both_view.connect_all_to_all(weights=0.1, storage_format=cls.storage_format, storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, pre_view, post_view, both_view])
        cls.test_net.compile(silent=True)

        cls.pop1 = cls.test_net.get(pop1)
        cls.pop2 = cls.test_net.get(pop2)
        cls.prev = cls.test_net.get(pre_view)
        cls.post = cls.test_net.get(post_view)
        cls.both = cls.test_net.get(both_view)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self.test_net.reset(populations=True, projections=True)
        self.test_net.disable_learning()

    def test_compile(self):
        """
        Test Compile.
        """
        pass

    def test_pre_view(self):
        """
        Connection of all post-synaptic neurons with a presynaptic PopulationView

        pre-slice: sum over only 3 neurons == 6 times 0.1 -> expect 0.6
        post-all: all neurons receive the input
        """
        self.pop1.r = numpy.arange(self.pop1.size)
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.pop2.sum("prev"), 0.6)

    def test_post_view(self):
        """
        Connection of a postsynaptic PopulationView with all post-synaptic neurons 

        pre-all: sum over all values == 28 times 0.1 -> expect 2.8
        post-slice: only neurons with rank 1 and 2 receive input, the rest is zero
        """
        self.pop1.r = numpy.arange(self.pop1.size)
        self.test_net.simulate(1)
        numpy.testing.assert_allclose(self.pop2.sum("postv"), [0.0, 2.8, 2.8, 0.0, 0.0, 0.0])

    def test_both_view(self):
        """
        Connection with a pre- and postsynaptic PopulationView

        pre-slice: sum over only 3 neurons == 6 times 0.1 -> expect 0.6
        post-slice: only neurons with rank 1 and 2 receive input, the rest is zero
        """
        self.pop1.r = numpy.arange(self.pop1.size)
        self.test_net.simulate(1)

        # only neurons with rank 1 and 2 receive input, the rest is zero
        numpy.testing.assert_allclose(self.pop2.sum("bothv"), [0.0, 0.6, 0.6, 0.0, 0.0, 0.0])
