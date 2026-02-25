"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy
from scipy import sparse

from conftest import TARGET_FOLDER
from ANNarchy import Neuron, Synapse, Network


class test_DefaultProjection(unittest.TestCase):
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
            parameters="r=0",
        )

        # define a sparse matrix
        weight_matrix = sparse.lil_matrix((4, 8))
        # HD (01.07.20): its not possible to use slicing here, as it produces
        #                FutureWarnings in scipy/numpy (for version >= 1.17)
        for i in range(8):
            weight_matrix[1, i] = 0.2
        for i in range(2, 6):
            weight_matrix[3, i] = 0.5

        # we need to flip the matrix (see 2.9.3.2 in documentation)
        cls.weight_matrix = weight_matrix.T

        cls._network = Network()
        pop1 = cls._network.create(geometry=(8), neuron=simple)
        pop2 = cls._network.create(geometry=(4), neuron=simple)

        cls._proj = cls._network.connect(pre=pop1, post=pop2, target="exc")
        cls._proj.from_sparse(
            cls.weight_matrix,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    def setUp(self):
        """
        In our *setUp()* function we reset the network before every test.
        """
        self._network.reset()

    def test_get_w(self):
        """
        Test the direct access to the synaptic weight.
        """
        # test row 1 (idx 0) with 8 elements should be 0.2
        numpy.testing.assert_allclose(self._proj.w[0], 0.2)

        # test row 3 (idx 1) with 8 elements should be 0.5
        numpy.testing.assert_allclose(self._proj.w[1], 0.5)

    def test_get_dendrite_w(self):
        """
        Test the access through dendrite to the synaptic weight.
        """
        # test row 1 with 8 elements should be 0.2
        numpy.testing.assert_allclose(self._proj.dendrite(1).w, 0.2)

        # test row 3 with 4 elements should be 0.5
        numpy.testing.assert_allclose(self._proj.dendrite(3).w, 0.5)

    def test_get_size(self):
        """
        Tests the *size* method, which returns the number of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self._proj.size, 2)

    def test_get_post_ranks(self):
        """
        Tests the *post_ranks* method, which returns the ranks of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self._proj.post_ranks, [1, 3])


class test_ModifiedProjection(unittest.TestCase):
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
        simple_neuron = Neuron(
            parameters="r=0",
        )

        simple_synapse = Synapse(
            parameters="""
                loc_var        = 5000.0
                semi_glob_var  = 8.0    : postsynaptic
                glob_var       = 8.0    : projection
                """,
            equations="""
                dx/dt = -x              : init = loc_var
                dw/dt = -w
                """,
        )

        # define a sparse matrix
        weight_matrix = sparse.lil_matrix((4, 8))
        # HD (01.07.20): its not possible to use slicing here, as it produces
        #                FutureWarnings in scipy/numpy (for version >= 1.17)
        for i in range(8):
            weight_matrix[1, i] = 0.2
        for i in range(2, 6):
            weight_matrix[3, i] = 0.5

        # we need to flip the matrix (see 2.9.3.2 in documentation)
        cls.weight_matrix = weight_matrix.T

        cls._network = Network()
        pop1 = cls._network.create(geometry=(8), neuron=simple_neuron)
        pop2 = cls._network.create(geometry=(4), neuron=simple_neuron)

        cls._proj = cls._network.connect(
            pre=pop1, post=pop2, target="exc", synapse=simple_synapse
        )
        cls._proj.from_sparse(
            cls.weight_matrix,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    def setUp(self):
        """
        In our *setUp()* function we reset the network before every test.
        """
        self._network.reset()

    def test_get_w(self):
        """
        Test the direct access to the synaptic weight.
        """
        # test row 1 (idx 0) with 8 elements should be 0.2
        numpy.testing.assert_allclose(self._proj.w[0], 0.2)

        # test row 3 (idx 1) with 8 elements should be 0.5
        numpy.testing.assert_allclose(self._proj.w[1], 0.5)

    def test_get_dendrite_w(self):
        """
        Test the access through dendrite to the synaptic weight.
        """
        # test row 1 with 8 elements should be 0.2
        numpy.testing.assert_allclose(self._proj.dendrite(1).w, 0.2)

        # test row 3 with 4 elements should be 0.5
        numpy.testing.assert_allclose(self._proj.dendrite(3).w, 0.5)

    def test_get_loc_var(self):
        """
        Tests the direct access to the parameter *loc_var* of our *Projection*.
        """
        # test row 1 (idx 0) with 8 elements
        numpy.testing.assert_allclose(self._proj.loc_var[0], 5000.0)
        # test row 3 (idx 1) with 4 elements
        numpy.testing.assert_allclose(self._proj.loc_var[1], 5000.0)

    def test_get_loc_var_2(self):
        """
        Tests the access to the parameter *loc_var* of our *Projection* with the *get()* method.
        """
        numpy.testing.assert_allclose(self._proj.get("loc_var")[0], 5000.0)
        numpy.testing.assert_allclose(self._proj.get("loc_var")[1], 5000.0)

    def test_get_loc_var_from_other(self):
        """
        Tests the access to a variable inited from parameter.
        """
        numpy.testing.assert_allclose(self._proj.x[0], 5000.0)
        numpy.testing.assert_allclose(self._proj.get("x")[0], 5000.0)

    def test_get_semi_glob_var(self):
        """
        Tests the direct access to the parameter *semi_glob_var* of our *Projection*.
        """
        numpy.testing.assert_allclose(self._proj.semi_glob_var, 8.0)

    def test_get_semi_glob_var_2(self):
        """
        Tests the access to the parameter *semi_glob_var* of our *Projection* with the
        *get()* method.
        """
        numpy.testing.assert_allclose(self._proj.get("semi_glob_var"), 8.0)

    def test_get_glob_var(self):
        """
        Tests the direct access to the parameter *glob_var* of our *Projection*.
        """
        numpy.testing.assert_allclose(self._proj.glob_var, 8.0)

    def test_get_glob_var_2(self):
        """
        Tests the access to the parameter *glob_var* of our *Projection* with the
        *get()* method.
        """
        numpy.testing.assert_allclose(self._proj.get("glob_var"), 8.0)

    def test_get_size(self):
        """
        Tests the *size* method, which returns the number of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self._proj.size, 2)

    def test_get_post_ranks(self):
        """
        Tests the *post_ranks* method, which returns the ranks of post-synaptic
        neurons recieving synapses.
        """
        self.assertEqual(self._proj.post_ranks, [1, 3])


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
        rNeuron = Neuron(
            equations="""
            p = sum(prev)
            q = sum(postv)
            r = sum(bothv)
        """
        )

        cls._network = Network()

        cls._pop1 = cls._network.create(geometry=8, neuron=preNeuron)
        cls._pop2 = cls._network.create(geometry=6, neuron=rNeuron)

        # see test_pre_view()
        cls._prev = cls._network.connect(cls._pop1[1:4], cls._pop2, target="prev")
        cls._prev.all_to_all(
            weights=0.1,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )

        # see test_post_view()
        cls._post = cls._network.connect(cls._pop1, cls._pop2[1:3], target="postv")
        cls._post.all_to_all(
            weights=0.1,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )

        # see test_both_view()
        cls._both = cls._network.connect(cls._pop1[1:4], cls._pop2[1:3], target="bothv")
        cls._both.all_to_all(
            weights=0.1,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        self._network.reset(populations=True, projections=True)
        self._network.disable_learning()

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
        self._pop1.r = numpy.arange(self._pop1.size)
        self._network.simulate(1)
        numpy.testing.assert_allclose(self._pop2.sum("prev"), 0.6)

    def test_post_view(self):
        """
        Connection of a postsynaptic PopulationView with all post-synaptic neurons

        pre-all: sum over all values == 28 times 0.1 -> expect 2.8
        post-slice: only neurons with rank 1 and 2 receive input, the rest is zero
        """
        self._pop1.r = numpy.arange(self._pop1.size)
        self._network.simulate(1)
        numpy.testing.assert_allclose(
            self._pop2.sum("postv"), [0.0, 2.8, 2.8, 0.0, 0.0, 0.0]
        )

    def test_both_view(self):
        """
        Connection with a pre- and postsynaptic PopulationView

        pre-slice: sum over only 3 neurons == 6 times 0.1 -> expect 0.6
        post-slice: only neurons with rank 1 and 2 receive input, the rest is zero
        """
        self._pop1.r = numpy.arange(self._pop1.size)
        self._network.simulate(1)

        # only neurons with rank 1 and 2 receive input, the rest is zero
        numpy.testing.assert_allclose(
            self._pop2.sum("bothv"), [0.0, 0.6, 0.6, 0.0, 0.0, 0.0]
        )
