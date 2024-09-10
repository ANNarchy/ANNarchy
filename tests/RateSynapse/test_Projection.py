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
import numpy
from scipy import sparse

from ANNarchy import Neuron, Synapse, Population, Projection, Network

class test_Projection():
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
