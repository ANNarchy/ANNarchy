"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron, Uniform

# neuron defintions common used for test cases
neuron = Neuron(parameters="""tau = 10""", equations="""r += t/tau""")

neuron2 = Neuron(parameters="tau = 10: population", equations="r += t/tau : init = 1.0")


class test_Population2D(unittest.TestCase):
    """
    Test several functions of the *Population* object in this particular
    test, we focus on two-dimensional case:

        * access methods for variables and parameters
        * coordinate transformations
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        cls._network = Network()
        cls._population_1 = cls._network.create(geometry=(3, 3), neuron=neuron)
        cls._population_2 = cls._network.create(geometry=(3, 3), neuron=neuron2)
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically
        to reset the network after every test.
        """
        self._network.reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this
        test we prove coordinate to rank transformation.
        """
        self.assertSequenceEqual(self._population_1.coordinates_from_rank(2), (0, 2))
        self.assertSequenceEqual(self._population_1.coordinates_from_rank(6), (2, 0))

    def test_rank_from_coordinates(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this
        test we prove rank to coordinate transformation.
        """
        self.assertEqual(self._population_1.rank_from_coordinates((0, 2)), 2)
        self.assertEqual(self._population_1.rank_from_coordinates((2, 0)), 6)

    #
    # Parameters
    #
    def test_get_tau(self):
        """
        Test retrieval of parameter *tau* from population *tc2_pop1* by
        directly access.  As population has the size 9 there should be 9
        entries with value 10.
        """
        numpy.testing.assert_allclose(
            self._population_1.tau,
            [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
        )

    def test_get_tau2(self):
        """
        Test retrieval of parameter *tau* from population *tc2_pop1* by *get()*
        method.  As population has the size 9 there should be 9 entries with
        value 10.
        """
        numpy.testing.assert_allclose(
            self._population_1.get("tau"),
            [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
        )

    def test_get_neuron_tau(self):
        """
        Tests retrieval of parameter *tau* from a specific neuron from
        population *tc2_pop1* by direct access.
        """
        numpy.testing.assert_allclose(self._population_1.neuron(1).tau, 10.0)

    def test_set_tau(self):
        """
        Assigned a new value, all instances will change.
        """
        self._population_1.tau = 5.0
        numpy.testing.assert_allclose(
            self._population_1.tau, [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
        )

    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change.
        """
        self._population_1.set({"tau": 5.0})
        numpy.testing.assert_allclose(
            self._population_1.tau, [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
        )

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally.
        One can use *PopulationView* to update more specific.
        """
        self._population_1[1:3, 1].tau = 5.0
        numpy.testing.assert_allclose(
            self._population_1.tau,
            [[10.0, 10.0, 10.0], [10.0, 5.0, 10.0], [10.0, 5.0, 10.0]],
        )

    def test_get_tau_population(self):
        """
        Test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(self._population_2.tau, 10.0)

    def test_popattributes(self):
        """
        Tests the listing of *Population* attributes.
        """
        self.assertEqual(
            self._population_1.attributes, ["tau", "r"], "failed listing attributes"
        )
        self.assertEqual(
            self._population_1.parameters, ["tau"], "failed listing parameters"
        )
        self.assertEqual(
            self._population_1.variables, ["r"], "failed listing variables"
        )

    #
    # Variables
    #
    def test_get_r(self):
        """
        By default all variables are initialized with zero, which is tested
        here by retrieving *r* directly.
        """
        numpy.testing.assert_allclose(
            self._population_1.r, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )

    def test_get_r2(self):
        """
        Tests the retrieval of the variable *r* through the *get()* method.
        """
        numpy.testing.assert_allclose(
            self._population_1.get("r"),
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

    def test_get_neuron_r(self):
        """
        Tests the retrieval of the variable *r* from a specific neuron by
        direct access.
        """
        numpy.testing.assert_allclose(self._population_1.neuron(0).r, 0.0)

    def test_get_r_with_init(self):
        """
        By default all variables are initialized with zero, we now modified
        this with init = 1.0 and test it.
        """
        numpy.testing.assert_allclose(
            self._population_2.r, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )

    def test_set_r(self):
        """
        Test the setting of the variable *r* by direct access.
        """
        self._population_1.r = 1.0
        numpy.testing.assert_allclose(
            self._population_1.r, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )

    def test_set_r_2(self):
        """
        Here we set only a change the variable of a selected field of neurons.
        The rest should stay the same.
        """
        self._population_1[1:3, 1].r = 2.0
        numpy.testing.assert_allclose(
            self._population_1.r, [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
        )

    def test_set_r_uniform(self):
        """
        Test the setting of the variable *r* by the *Uniform()* method.  This
        method assigns a random value (within a chosen interval) to the
        variable of each neuron.
        """
        self._population_1.r = Uniform(0.0, 1.0).get_values(9)
        self.assertTrue(
            any(self._population_1[0:3, 0:3].r >= 0.0)
            and all(self._population_1[0:3, 0:3].r <= 1.0)
        )

    def test_set_r3(self):
        """
        Test the setting of the variable *r* by the *set()* method.
        """
        self._population_1.set({"r": 1.0})
        numpy.testing.assert_allclose(
            self._population_1.r, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )

    #
    # Reset-Test
    #
    def test_reset(self):
        """
        Tests the functionality of the *reset()* method, which we use in our
        *setUp()* function.
        """
        self._population_1.tau = 5.0
        numpy.testing.assert_allclose(
            self._population_1.tau, [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
        )
        self._network.reset()
        numpy.testing.assert_allclose(
            self._population_1.tau,
            [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]],
        )


class test_Population2D_NonSquared(unittest.TestCase):
    """
    Test several functions of the *Population* object in this particular test,
    we focus on assymetrical two-dimensional case:

        * access methods for variables and parameters
        * coordinate transformations

    TODO: I don't recall why we test a squared and a non-squared 2-dimensional
          population. I leave it for now ... (HD: 24th July 2025)
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        cls._network = Network()
        cls._population_1 = cls._network.create(geometry=(3, 2), neuron=neuron)
        cls._population_2 = cls._network.create(geometry=(3, 2), neuron=neuron2)
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this
        test we prove coordinate to rank transformation.
        """
        self.assertSequenceEqual(self._population_1.coordinates_from_rank(2), (1, 0))
        self.assertSequenceEqual(self._population_1.coordinates_from_rank(5), (2, 1))

    def test_rank_from_coordinates(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this
        test we prove rank to coordinate transformation.
        """
        self.assertEqual(self._population_1.rank_from_coordinates((1, 0)), 2)
        self.assertEqual(self._population_1.rank_from_coordinates((2, 1)), 5)

    #
    # Parameters
    #
    def test_get_tau(self):
        """
        Test retrieval of parameter *tau* from population *tc4_pop1* by
        directly access.  As population has the size 6 there should be 6
        entries with value 10.
        """
        numpy.testing.assert_allclose(
            self._population_1.tau, [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
        )

    def test_get_tau2(self):
        """
        Test retrieval of parameter *tau* from population *tc4_pop1* by *get()*
        method.  As population has the size 6 there should be 6 entries with
        value 10.
        """
        numpy.testing.assert_allclose(
            self._population_1.get("tau"), [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
        )

    def test_get_neuron_tau(self):
        """
        Tests retrieval of parameter *tau* from a specific neuron from
        population *tc4_pop1* by direct access.
        """

        numpy.testing.assert_allclose(self._population_1.neuron(1).tau, 10.0)

    def test_set_tau(self):
        """
        Assigned a new value, all instances will change.
        """
        self._population_1.tau = 5.0
        numpy.testing.assert_allclose(
            self._population_1.tau, [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
        )

    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change.
        """
        self._population_1.set({"tau": 5.0})
        numpy.testing.assert_allclose(
            self._population_1.tau, [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
        )

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally.
        One can use *PopulationView* to update more specific.
        """
        self._population_1[0:2, 1].tau = 5.0
        numpy.testing.assert_allclose(
            self._population_1.tau, [[10.0, 5.0], [10.0, 5.0], [10.0, 10.0]]
        )

    def test_get_tau_population(self):
        """
        Test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(self._population_2.tau, 10.0)

    def test_popattributes(self):
        """
        Tests the listing of *Population* attributes.
        """
        self.assertEqual(
            self._population_1.attributes, ["tau", "r"], "failed listing attributes"
        )
        self.assertEqual(
            self._population_1.parameters, ["tau"], "failed listing parameters"
        )
        self.assertEqual(
            self._population_1.variables, ["r"], "failed listing variables"
        )

    #
    # Variables
    #
    def test_get_r(self):
        """
        By default all variables are initialized with zero, which is tested
        here by retrieving *r* directly.
        """
        numpy.testing.assert_allclose(
            self._population_1.r, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )

    def test_get_r2(self):
        """
        Tests the retrieval of the variable *r* through the *get()* method.
        """
        numpy.testing.assert_allclose(
            self._population_1.get("r"), [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )

    def test_get_neuron_r(self):
        """
        Tests the retrieval of the variable *r* from a specific neuron by
        direct access.
        """
        numpy.testing.assert_allclose(self._population_1.neuron(0).r, 0.0)

    def test_get_r_with_init(self):
        """
        By default all variables are initialized with zero, we now modified
        this with init = 1.0 and test it.
        """
        numpy.testing.assert_allclose(
            self._population_2.r, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        )

    def test_set_r(self):
        """
        Test the setting of the variable *r* by direct access.
        """
        self._population_1.r = 1.0
        numpy.testing.assert_allclose(
            self._population_1.r, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        )

    def test_set_r_2(self):
        """
        Here we set only a change the variable of a selected field of neurons.
        The rest should stay the same.
        """
        self._population_1[0:2, 1].r = 2.0
        numpy.testing.assert_allclose(
            self._population_1.r, [[0.0, 2.0], [0.0, 2.0], [0.0, 0.0]]
        )

    def test_set_r_uniform(self):
        """
        Test the setting of the variable *r* by the *Uniform()* method. This
        method assigns a random value (within a chosen interval) to the
        variable of each neuron.
        """
        self._population_1.r = Uniform(0.0, 1.0).get_values(6)
        self.assertTrue(
            any(self._population_1[0:3, 0:2].r >= 0.0)
            and all(self._population_1[0:3, 0:2].r <= 1.0)
        )

    def test_set_r3(self):
        """
        Test the setting of the variable *r* by the *set()* method.
        """
        self._population_1.set({"r": 1.0})
        numpy.testing.assert_allclose(
            self._population_1.r, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        )

    #
    # Reset-Test
    #
    def test_reset(self):
        """
        Tests the functionality of the *reset()* method, which we use in our
        *setUp()* function.
        """
        self._population_1.tau = 5.0
        numpy.testing.assert_allclose(
            self._population_1.tau, [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]
        )
        self._network.reset()
        numpy.testing.assert_allclose(
            self._population_1.tau, [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]
        )
