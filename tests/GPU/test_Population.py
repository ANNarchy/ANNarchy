"""

    test_Population.py

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

# neuron defintions common used for test cases
neuron = Neuron(
    parameters = """tau = 10""",
    equations = """r += 1/tau * t"""
)

neuron2 = Neuron(
    parameters = "tau = 10: population",
    equations = "r += 1/tau * t: init = 1.0"
)

# Populations for TestCase1
tc1_pop1 = Population(3, neuron)
tc1_pop2 = Population(3, neuron2)

# Populations for TestCase2
tc2_pop1 = Population((3, 3), neuron)
tc2_pop2 = Population((3, 3), neuron2)

# Populations for TestCase3
tc3_pop1 = Population((3, 3, 3), neuron)
tc3_pop2 = Population((3, 3, 3), neuron2)

# Populations for TestCase4
tc4_pop1 = Population((3, 2), neuron)
tc4_pop2 = Population((3, 2), neuron2)

class test_Population1D(unittest.TestCase):
    """
    Test several functions of the *Population* object in this particular test, we focus on one-dimensional case:

        * access methods for variables and parameters
        * coordinate transformations
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([tc1_pop1, tc1_pop2])
        self.test_net.compile(silent=True)

        self.net_pop1 = self.test_net.get(tc1_pop1)
        self.net_pop2 = self.test_net.get(tc1_pop2)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset() # network reset

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this test we prove
        coordinate to rank transformation.
        """
        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(1), (1, ))

    def test_rank_from_coordinates(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this test we prove
        rank to coordinate transformation.
        """
        self.assertEqual(self.net_pop1.rank_from_coordinates((1, )), 1)

    #
    # Parameters
    #
    def test_get_tau(self):
        """
        Test retrieval of parameter *tau* from population *tc1_pop1* by directly access.
        As population has the size 3 there should be 3 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [10.0, 10.0, 10.0]))

    def test_get_tau2(self):
        """
        Test retrieval of parameter *tau* from population *tc1_pop1* by *get()* method.
        As population has the size 3 there should be 3 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('tau'), [10.0, 10.0, 10.0]))

    def test_get_neuron_tau(self):
        """
        Tests retrieval of parameter *tau* from a specific neuron from population *tc1_pop1* by direct access.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.neuron(1).tau, 10.0))

    def test_set_tau(self):
        """
        Assigned a new value, all instances will change.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [5.0, 5.0, 5.0]))

    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change.
        """
        self.net_pop1.set({'tau' : 7.0})
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [7.0, 7.0, 7.0]))

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally. 
        One can use *PopulationView* to update more specific.
        """

        self.net_pop1[1:3].tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [10.0, 5.0, 5.0]))


    def test_get_tau_population(self):
        """
        Test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(self.net_pop2.tau, 10.0)

    def test_popattributes(self):
        """
        Tests the listing of *Population* attributes.
        """
        self.assertEqual(self.net_pop1.attributes, ['tau', 'r'], 'failed listing attributes')
        self.assertEqual(self.net_pop1.parameters, ['tau'], 'failed listing parameters')
        self.assertEqual(self.net_pop1.variables, ['r'], 'failed listing variables')

    #
    # Variables
    #
    def test_get_r(self):
        """
        By default all variables are initialized with zero, which is tested here by retrieving *r* directly.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.r, [0.0, 0.0, 0.0]))

    def test_get_r2(self):
        """
        Tests the retrieval of the variable *r* through the *get()* method.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('r'), [0.0, 0.0, 0.0]))

    def test_get_neuron_r(self):
        """
        Tests the retrieval of the variable *r* from a specific neuron by direct access.
        """

        self.assertTrue(numpy.allclose(self.net_pop1.neuron(0).r, 0.0))

    def test_get_r_with_init(self):
        """
        By default all variables are initialized with zero, we now
        modified this with init = 1.0 and test it.
        """
        self.assertTrue(numpy.allclose(self.net_pop2.r, [1.0, 1.0, 1.0]))

    def test_set_r(self):
        """
        Test the setting of the variable *r* by direct access.
        """
        self.net_pop1.r = 1.0
        self.assertTrue(numpy.allclose(self.net_pop1.r, [1.0, 1.0, 1.0]))

    def test_set_r_2(self):
        """
        Here we set only a change the variable of a selected field of neurons. The rest should stay the same.
        """
        self.net_pop1[1:3].r = 2.0
        self.assertTrue(numpy.allclose(self.net_pop1.r, [0.0, 2.0, 2.0]))

    def test_set_r_uniform(self):
        """
        Test the setting of the variable *r* by the *Uniform()* method.
        This method assigns a random value (within a chosen interval) to the variable of each neuron.
        """
        self.net_pop1.r = Uniform(0.0, 1.0).get_values(3)
        self.assertTrue(any(self.net_pop1.r >= 0.0) and all(self.net_pop1.r <= 1.0))

    def test_set_r3(self):
        """
        Test the setting of the variable *r* by the *set()* method.
        """
        self.net_pop1.set({'r': 1.0})
        self.assertTrue(numpy.allclose(self.net_pop1.r, [1.0, 1.0, 1.0]))


    #
    # Reset-Test
    #
    def test_reset(self):
        """
        Tests the functionality of the *reset()* method, which we use in our *setUp()* function.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [5.0, 5.0, 5.0]))
        self.test_net.reset()
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [10.0, 10.0, 10.0]))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class test_Population2D(unittest.TestCase):
    """
    Test several functions of the *Population* object in this particular
    test, we focus on two-dimensional case:

        * access methods for variables and parameters
        * coordinate transformations
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([tc2_pop1, tc2_pop2])
        self.test_net.compile(silent=True)

        self.net_pop1 = self.test_net.get(tc2_pop1)
        self.net_pop2 = self.test_net.get(tc2_pop2)

    def setUp(self):
        """
        Automatically called before each test method, basically
        to reset the network after every test.
        """
        self.test_net.reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        ANNarchy allows two types of indexing, coordinates
        and ranks. In this test we prove coordinate to
        rank transformation.
        """
        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(2), (0, 2))
        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(6), (2, 0))

    def test_rank_from_coordinates(self):
        """

        ANNarchy allows two types of indexing, coordinates and ranks. In this test we prove
        rank to coordinate transformation.
        """
        self.assertEqual(self.net_pop1.rank_from_coordinates((0, 2)), 2)
        self.assertEqual(self.net_pop1.rank_from_coordinates((2, 0)), 6)

    #
    # Parameters
    #
    def test_get_tau(self):
        """

        Test retrieval of parameter *tau* from population *tc2_pop1* by directly access.
        As population has the size 9 there should be 9 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[10., 10., 10.],
                                                      [10., 10., 10.],
                                                      [10., 10., 10.]]))

    def test_get_tau2(self):
        """
        Test retrieval of parameter *tau* from population *tc2_pop1* by *get()* method.
        As population has the size 9 there should be 9 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('tau'), [[10., 10., 10.],
                                                             [10., 10., 10.],
                                                             [10., 10., 10.]]))

    def test_get_neuron_tau(self):
        """
        Tests retrieval of parameter *tau* from a specific neuron from population *tc2_pop1* by direct access.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.neuron(1).tau, 10.0))
        

    def test_set_tau(self):
        """

        Assigned a new value, all instances will change.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[5., 5., 5.],
                                                           [5., 5., 5.],
                                                           [5., 5., 5.]]))
            
    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change.
        """
        self.net_pop1.set({'tau' : 5.0})
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[5., 5., 5.],
                                                           [5., 5., 5.],
                                                           [5., 5., 5.]]))

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally. 
        One can use *PopulationView* to update more specific.
        """
        self.net_pop1[1:3, 1].tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[10., 10., 10.],
                                                           [10., 5., 10.],
                                                           [10., 5., 10.]]))

    def test_get_tau_population(self):
        """
        Test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(self.net_pop2.tau, 10.0)

    def test_popattributes(self):
        """
        Tests the listing of *Population* attributes.
        """
        self.assertEqual(self.net_pop1.attributes, ['tau', 'r'], 'failed listing attributes')
        self.assertEqual(self.net_pop1.parameters, ['tau'], 'failed listing parameters')
        self.assertEqual(self.net_pop1.variables, ['r'], 'failed listing variables')

    #
    # Variables
    #
    def test_get_r(self):
        """
        By default all variables are initialized with zero, which is tested here by retrieving *r* directly.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 0.,  0.,  0.],
                                                    [ 0.,  0.,  0.],
                                                    [ 0.,  0.,  0.]]))

    def test_get_r2(self):
        """
        Tests the retrieval of the variable *r* through the *get()* method.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('r'), [[ 0.,  0.,  0.],
                                                           [ 0.,  0.,  0.],
                                                           [ 0.,  0.,  0.]]))

    def test_get_neuron_r(self):
        """
        Tests the retrieval of the variable *r* from a specific neuron by direct access.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.neuron(0).r, 0.0))

    def test_get_r_with_init(self):
        """
        By default all variables are initialized with zero, we now
        modified this with init = 1.0 and test it.
        """
        self.assertTrue(numpy.allclose(self.net_pop2.r, [[ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.]]))

    def test_set_r(self):
        """
        Test the setting of the variable *r* by direct access.
        """
        self.net_pop1.r = 1.0
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.]]))
    def test_set_r_2(self):
        """
        Here we set only a change the variable of a selected field of neurons. The rest should stay the same.
        """
        self.net_pop1[1:3, 1].r = 2.0
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 0.,  0.,  0.],
                                                    [ 0.,  2.,  0.],
                                                    [ 0.,  2.,  0.]]))

    def test_set_r_uniform(self):
        """
        Test the setting of the variable *r* by the *Uniform()* method.
        This method assigns a random value (within a chosen interval) to the variable of each neuron.
        """
        self.net_pop1.r = Uniform(0.0, 1.0).get_values(9)
        self.assertTrue(any(self.net_pop1[0:3, 0:3].r >= 0.0) and all(self.net_pop1[0:3, 0:3].r <= 1.0))

    def test_set_r3(self):
        """
        Test the setting of the variable *r* by the *set()* method.
        """
        self.net_pop1.set({'r': 1.0})
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.]]))

    #
    #Reset-Test
    #
    def test_reset(self):
        """
        Tests the functionality of the *reset()* method, which we use in our *setUp()* function.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.]]))
        self.test_net.reset()
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 10.,  10.,  10.],
                                                      [ 10.,  10.,  10.],
                                                      [ 10.,  10.,  10.]]))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class test_Population3D(unittest.TestCase):
    """
    Test several functions of the *Population* object in this particular test, we focus on three-dimensional case:

        * access methods for variables and parameters
        * coordinate transformations

    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([tc3_pop1, tc3_pop2])
        self.test_net.compile(silent=True)

        self.net_pop1 = self.test_net.get(tc3_pop1)
        self.net_pop2 = self.test_net.get(tc3_pop2)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.

        """
        self.test_net.reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this test we prove
        coordinate to rank transformation.

        """

        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(2), (0, 0, 2))
        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(6), (0, 2, 0))
        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(18), (2, 0, 0))

    def test_rank_from_coordinates(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this test we prove
        rank to coordinate transformation.
        """
        self.assertEqual(self.net_pop1.rank_from_coordinates((0, 0, 2)), 2)
        self.assertEqual(self.net_pop1.rank_from_coordinates((0, 2, 0)), 6)
        self.assertEqual(self.net_pop1.rank_from_coordinates((2, 0, 0)), 18)

    #
    # Parameters
    #
    def test_get_tau(self):
        """
        Test retrieval of parameter *tau* from population *tc3_pop1* by directly access.
        As population has the size 27 there should be 27 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[[ 10., 10., 10.],
                                                            [ 10., 10., 10.],
                                                            [ 10., 10., 10.]],
                                                           [[ 10., 10., 10.],
                                                            [ 10., 10., 10.],
                                                            [ 10., 10., 10.]],
                                                           [[ 10., 10., 10.],
                                                            [ 10., 10., 10.],
                                                            [ 10., 10., 10.]]]))

    def test_get_tau2(self):
        """
        Test retrieval of parameter *tau* from population *tc3_pop1* by *get()* method.
        As population has the size 27 there should be 27 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('tau'), [[[10., 10., 10.],
                                                                   [10., 10., 10.],
                                                                   [10., 10., 10.]],
                                                                  [[10., 10., 10.],
                                                                   [10., 10., 10.],
                                                                   [10., 10., 10.]],
                                                                  [[10., 10., 10.],
                                                                   [10., 10., 10.],
                                                                   [10., 10., 10.]]]))

    def test_get_neuron_tau(self):
        """
        Tests retrieval of parameter *tau* from a specific neuron from population *tc3_pop1* by direct access.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.neuron(1).tau, 10.0))

    def test_set_tau(self):
        """

        Assigned a new value, all instances will change.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]],
                                                      [[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]],
                                                      [[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]]]))

    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change.
        """
        self.net_pop1.set({'tau' : 5.0})
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]],
                                                      [[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]],
                                                      [[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]]]))

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally.
        One can use *PopulationView* to update more specific.
        """
        self.net_pop1[0:3, 1, 1:3].tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[[ 10.,  10.,  10.],
                                                       [ 10.,  5.,  5.],
                                                       [ 10.,  10.,  10.]],
                                                      [[ 10.,  10.,  10.],
                                                       [ 10.,  5.,  5.],
                                                       [ 10.,  10.,  10.]],
                                                      [[ 10.,  10.,  10.],
                                                       [ 10.,  5.,  5.],
                                                       [ 10.,  10.,  10.]]]))

    def test_get_tau_population(self):
        """
        Test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(self.net_pop2.tau, 10.0)

    def test_popattributes(self):
        """
        Tests the listing of *Population* attributes.
        """
        self.assertEqual(self.net_pop1.attributes, ['tau', 'r'], 'failed listing attributes')
        self.assertEqual(self.net_pop1.parameters, ['tau'], 'failed listing parameters')
        self.assertEqual(self.net_pop1.variables, ['r'], 'failed listing variables')

    #
    # Variables
    #
    def test_get_r(self):
        """
        By default all variables are initialized with zero, which is tested here by retrieving *r* directly.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[[ 0.,  0.,  0.],
                                                     [ 0.,  0.,  0.],
                                                     [ 0.,  0.,  0.]],
                                                    [[ 0.,  0.,  0.],
                                                     [ 0.,  0.,  0.],
                                                     [ 0.,  0.,  0.]],
                                                    [[ 0.,  0.,  0.],
                                                     [ 0.,  0.,  0.],
                                                     [ 0.,  0.,  0.]]]))

    def test_get_r2(self):
        """
        Tests the retrieval of the variable *r* through the *get()* method.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('r'), [[[0., 0., 0.],
                                                                [0., 0., 0.],
                                                                [0., 0., 0.]],
                                                               [[0., 0., 0.],
                                                                [0., 0., 0.],
                                                                [0., 0., 0.]],
                                                               [[0., 0., 0.],
                                                                [0., 0., 0.],
                                                                [0., 0., 0.]]]))

    def test_get_neuron_r(self):
        """
        Tests the retrieval of the variable *r* from a specific neuron by direct access.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.neuron(18).r, 0.0))

    def test_get_r_with_init(self):
        """
        By default all variables are initialized with zero, we now
        modified this with init = 1.0 and test it.
        """
        self.assertTrue(numpy.allclose(self.net_pop2.r, [[[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]]]))

    def test_set_r(self):
        """
        Test the setting of the variable *r* by direct access.
        """
        self.net_pop1.r = 1.0
        self.assertTrue( numpy.allclose(self.net_pop1.r, [[[ 1.,  1.,  1.],
                                                           [ 1.,  1.,  1.],
                                                           [ 1.,  1.,  1.]],
                                                          [[ 1.,  1.,  1.],
                                                           [ 1.,  1.,  1.],
                                                           [ 1.,  1.,  1.]],
                                                          [[ 1.,  1.,  1.],
                                                           [ 1.,  1.,  1.],
                                                           [ 1.,  1.,  1.]]]))

    def test_set_r_2(self):
        """
        Here we set only a change the variable of a selected field of neurons. The rest should stay the same.
        """
        self.net_pop1[0:3, 1, 1:3].r=2.0
        self.assertTrue( numpy.allclose(self.net_pop1.r, [[[ 0.,  0.,  0.],
                                                           [ 0., 2., 2.],
                                                           [ 0., 0., 0.]],
                                                          [[ 0., 0., 0.],
                                                           [ 0., 2., 2.],
                                                           [ 0., 0., 0.]],
                                                          [[ 0., 0., 0.],
                                                           [ 0., 2., 2.],
                                                           [ 0., 0., 0.]]]))

    def test_set_r_uniform(self):
        """
        Test the setting of the variable *r* by the *Uniform()* method.
        This method assigns a random value (within a chosen interval) to the variable of each neuron.
        """
        self.net_pop1.r=Uniform(0.0, 1.0).get_values(27)
        self.assertTrue(any(self.net_pop1[0:3, 0:3, 0:3].r >= 0.0) and all(self.net_pop1[0:3, 0:3, 0:3].r <= 1.0))

    def test_set_r3(self):
        """
        Test the setting of the variable *r* by the *set()* method.
        """
        self.net_pop1.set({'r': 1.0})
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[[ 1., 1., 1.],
                                                          [ 1., 1., 1.],
                                                          [ 1., 1., 1.]],
                                                         [[ 1., 1., 1.],
                                                          [ 1., 1., 1.],
                                                          [ 1., 1., 1.]],
                                                         [[ 1., 1., 1.],
                                                          [ 1., 1., 1.],
                                                          [ 1., 1., 1.]]]))

    #
    #Reset-Test
    #
    def test_reset(self):
        """
        Tests the functionality of the *reset()* method, which we use in our *setUp()* function.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[[5., 5., 5.],
                                                            [5., 5., 5.],
                                                            [5., 5., 5.]],
                                                           [[5., 5., 5.],
                                                            [5., 5., 5.],
                                                            [5., 5., 5.]],
                                                           [[5., 5., 5.],
                                                            [5., 5., 5.],
                                                            [5., 5., 5.]]]))

        self.test_net.reset()
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[[10., 10., 10.],
                                                            [10., 10., 10.],
                                                            [10., 10., 10.]],
                                                           [[10., 10., 10.],
                                                            [10., 10., 10.],
                                                            [10., 10., 10.]],
                                                           [[10., 10., 10.],
                                                            [10., 10., 10.],
                                                            [10., 10., 10.]]]))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class test_Population2x3D(unittest.TestCase):
    """
    Test several functions of the *Population* object in this particular test, we focus on assymetrical two-dimensional case:

        * access methods for variables and parameters
        * coordinate transformations

    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        self.test_net = Network()
        self.test_net.add([tc4_pop1, tc4_pop2])
        self.test_net.compile(silent=True)

        self.net_pop1 = self.test_net.get(tc4_pop1)
        self.net_pop2 = self.test_net.get(tc4_pop2)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.

        """
        self.test_net.reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        ANNarchy allows two types of indexing, coordinates and ranks. In this test we prove
        coordinate to rank transformation.

        """
        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(2), (1, 0))
        self.assertSequenceEqual(self.net_pop1.coordinates_from_rank(5), (2, 1))

    def test_rank_from_coordinates(self):
        """

        ANNarchy allows two types of indexing, coordinates and ranks. In this test we prove
        rank to coordinate transformation.
        """
        self.assertEqual(self.net_pop1.rank_from_coordinates( (1, 0) ), 2)
        self.assertEqual(self.net_pop1.rank_from_coordinates( (2, 1) ), 5)

    #
    # Parameters
    #
    def test_get_tau(self):
        """

        Test retrieval of parameter *tau* from population *tc4_pop1* by directly access.
        As population has the size 6 there should be 6 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 10.,  10.],
                                                      [ 10.,  10.],
                                                      [ 10.,  10.]]))

    def test_get_tau2(self):
        """
        Test retrieval of parameter *tau* from population *tc4_pop1* by *get()* method.
        As population has the size 6 there should be 6 entries with value 10.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('tau'), [[ 10.,  10.],
                                                               [ 10.,  10.],
                                                             [ 10.,  10.]]))

    def test_get_neuron_tau(self):
        """
        Tests retrieval of parameter *tau* from a specific neuron from population *tc4_pop1* by direct access.
        """

        self.assertTrue(numpy.allclose(self.net_pop1.neuron(1).tau, 10.0))

    def test_set_tau(self):
        """

        Assigned a new value, all instances will change.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 5.,  5.],
                                                      [ 5.,  5.],
                                                      [ 5.,  5.]]))
            
    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change.
        """
        self.net_pop1.set({'tau' : 5.0})
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 5.,  5.],
                                                      [ 5.,  5.],
                                                      [ 5.,  5.]]))

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally. 
        One can use *PopulationView* to update more specific.
        """
        self.net_pop1[0:2, 1].tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 10.,  5.],
                                                      [ 10.,  5.],
                                                      [ 10.,  10.]]))

    def test_get_tau_population(self):
        """
        Test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(self.net_pop2.tau, 10.0)

    def test_popattributes(self):
        """
        Tests the listing of *Population* attributes.
        """
        self.assertEqual(self.net_pop1.attributes, ['tau', 'r'], 'failed listing attributes')
        self.assertEqual(self.net_pop1.parameters, ['tau'], 'failed listing parameters')
        self.assertEqual(self.net_pop1.variables, ['r'], 'failed listing variables')

    #
    # Variables
    #
    def test_get_r(self):
        """

        By default all variables are initialized with zero, which is tested here by retrieving *r* directly.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 0., 0.],
                                                    [ 0., 0.],
                                                    [ 0., 0.]]))

    def test_get_r2(self):
        """
        Tests the retrieval of the variable *r* through the *get()* method.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.get('r'), [[ 0., 0.],
                                                           [ 0., 0.],
                                                           [ 0., 0.]]))

    def test_get_neuron_r(self):
        """
        Tests the retrieval of the variable *r* from a specific neuron by direct access.
        """
        self.assertTrue(numpy.allclose(self.net_pop1.neuron(0).r, 0.0))

    def test_get_r_with_init(self):
        """
        By default all variables are initialized with zero, we now
        modified this with init = 1.0 and test it.
        """
        self.assertTrue(numpy.allclose(self.net_pop2.r, [[ 1., 1.],
                                                    [ 1., 1.],
                                                    [ 1., 1.]]))

    def test_set_r(self):
        """
        Test the setting of the variable *r* by direct access.
        """
        self.net_pop1.r = 1.0
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 1., 1.],
                                                    [ 1., 1.],
                                                    [ 1., 1.]]))
    def test_set_r_2(self):
        """
        Here we set only a change the variable of a selected field of neurons. The rest should stay the same.
        """
        self.net_pop1[0:2, 1].r = 2.0
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 0., 2.],
                                                    [ 0., 2.],
                                                    [ 0., 0.]]))
    
    def test_set_r_uniform(self):
        """
        Test the setting of the variable *r* by the *Uniform()* method.
        This method assigns a random value (within a chosen interval) to the variable of each neuron.
        """
        self.net_pop1.r=Uniform(0.0, 1.0).get_values(6)
        self.assertTrue(any(tc4_pop1[0:3, 0:2].r>=0.0) and all(tc4_pop1[0:3, 0:2].r<=1.0))

    def test_set_r3(self):
        """
        Test the setting of the variable *r* by the *set()* method.
        """
        self.net_pop1.set({'r': 1.0})
        self.assertTrue(numpy.allclose(self.net_pop1.r, [[ 1., 1.],
                                                    [ 1., 1.],
                                                    [ 1., 1.]]))

    #
    # Reset-Test
    #
    def test_reset(self):
        """
        Tests the functionality of the *reset()* method, which we use in our *setUp()* function.
        """
        self.net_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 5., 5.],
                                                      [ 5., 5.],
                                                      [ 5., 5.]]))
        self.test_net.reset()
        self.assertTrue(numpy.allclose(self.net_pop1.tau, [[ 10., 10.],
                                                      [ 10., 10.],
                                                      [ 10., 10.]]))
