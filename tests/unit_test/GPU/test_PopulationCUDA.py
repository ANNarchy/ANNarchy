"""

    test_PopulationCUDA.py

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

# neuron defintions common used for 
# test cases
neuron = Neuron(
    parameters = "tau = 10",
    equations="r += 1/tau * t"
)

neuron2 = Neuron(
    parameters = "tau = 10: population",
    equations="r += 1/tau * t: init = 1.0"
)

# Populations for TestCase1
tc1_pop1 = Population(3, neuron)
tc1_pop2 = Population(3, neuron2)

# Populations for TestCase2
tc2_pop1 = Population((3,3), neuron)
tc2_pop2 = Population((3,3), neuron2)

# Populations for TestCase3
tc3_pop1 = Population((3,3,3), neuron)
tc3_pop2 = Population((3,3,3), neuron2)

compile(clean=True)

#
# TODO: I would prefer seperate test classes for 1 up to 3 dimensions
#       getter/setter for higher dimensionial populations should use identity
#       matrices to test correct transformation of stored data. Maybe one could
#       use alternatively non-symetric population shapes.
#
#       Secondly, please setup a test for reset of populations.
#

#
# Comments: when refering to keywords, e. g. population use * * to mark them (will appear italic in the documentation
#           when refering to ANNarchy objects, use correct spelling, e. g. PopulationView
#

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class test_Population1D(unittest.TestCase):
    """
    Test *Population* data storage, access methods for one-dimensional populations.
    """
    def setUp(self):
        reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        tests coordinates_from_rank method
        """
        self.assertSequenceEqual(tc1_pop1.coordinates_from_rank(1), (1, ))
        
    def test_rank_from_coordinates(self):
        """
        tests rank_from_coordinates method
        """
        self.assertEqual(tc1_pop1.rank_from_coordinates((1, )), 1)

    #
    # Parameters
    #
    def test_get_tau(self):
        """
        test access to parameter
        """
        self.assertTrue(numpy.allclose(tc1_pop1.tau, [10.0, 10.0, 10.0]))

    def test_get_tau2(self):
        """
        test access to parameter
        """
        self.assertTrue(numpy.allclose(tc1_pop1.get('tau'), [10.0, 10.0, 10.0]))

    def test_get_neuron_tau(self):
        """
        tests access to single specific neurons in the *Population* (tau)
        """

        self.assertTrue(numpy.allclose(tc1_pop1.neuron(1).tau, 10.0))
        

    def test_set_tau(self):
        """
        Assigned a new value, all instances will change
        """
        tc1_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(tc1_pop1.tau, [5.0,5.0,5.0]))
            
    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change
        """
        tc1_pop1.set({'tau' : 7.0})
        self.assertTrue(numpy.allclose(tc1_pop1.tau, [7.0,7.0,7.0]))

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally. 
        One can use *PopulationView* to update more specific
        """

        tc1_pop1[1:3].tau = 5.0
        self.assertTrue(numpy.allclose(tc1_pop1.tau, [10.0,5.0,5.0]))


    def test_get_tau_population(self):
        """
        test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(tc1_pop2.tau, 10.0)

    def test_popattributes(self):
        """
        tests listing *Population* attributes
        """
        self.assertEqual(tc1_pop1.attributes, ['tau', 'r'], 'failed listing attributes')
        self.assertEqual(tc1_pop1.parameters, ['tau'], 'failed listing parameters')
        self.assertEqual(tc1_pop1.variables, ['r'], 'failed listing variables')
    


    #
    # Variables
    #
    def test_get_r(self):
        """
        default all variables are initialized with zero
        """
        self.assertTrue(numpy.allclose(tc1_pop1.r, [0.0,0.0,0.0]))

    def test_get_r2(self):
        """
        tests getting method
        """
        self.assertTrue(numpy.allclose(tc1_pop1.get('r'), [0.0,0.0,0.0]))

    def test_get_neuron_r(self):
        """
        tests access to single specific neurons in the *Population* (r)
        """

        self.assertTrue(numpy.allclose(tc1_pop1.neuron(0).r, 0.0))


        

    def test_get_r_with_init(self):
        """
        default all variables are initialized with zero, we now
        modified this with init = 1.0
        """
        self.assertTrue(numpy.allclose(tc1_pop2.r, [1.0,1.0,1.0]))

    def test_set_r(self):
        """
        tests setting of variable
        """
        tc1_pop1.r=1.0
        self.assertTrue(numpy.allclose(tc1_pop1.r, [1.0, 1.0, 1.0]))
        
        tc1_pop1[1:3].r=2.0
        self.assertTrue(numpy.allclose(tc1_pop1.r, [1.0, 2.0, 2.0]))
    
        tc1_pop1.r=2.0
        tc1_pop1.r=Uniform(0.0, 1.0).get_values(3)
        self.assertTrue(any(tc1_pop1.r>=0.0) and all(tc1_pop1.r<=1.0))


    def test_set_r2(self):
        """
        tests the setting method
        """
        tc1_pop1.set({'r': 1.0})
        self.assertTrue(numpy.allclose(tc1_pop1.r, [1.0, 1.0, 1.0]))


    #
    #Reset-Test
    #
    
    def test_reset(self):
        """
        tests if *Population* is properly reset if reset() is called
        """
        tc1_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(tc1_pop1.tau, [5.0,5.0,5.0]))
        reset()
        self.assertTrue(numpy.allclose(tc1_pop1.tau, [10.0, 10.0, 10.0]))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class test_Population2D(unittest.TestCase):
    """
    Test *Population* data storage, access methods two-dimensional populations.
    """
    def setUp(self):
        reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """
        tests coordinates_from_rank method:
        
        * largest x in first line
        * largest y in first column
        """
        self.assertSequenceEqual(tc2_pop1.coordinates_from_rank(2), (0, 2))
        self.assertSequenceEqual(tc2_pop1.coordinates_from_rank(6), (2, 0))
        
    def test_rank_from_coordinates(self):
        """
        tests rank_from_coordinates method:

        * largest x in first line
        * largest y in first column
        """
        self.assertEqual( tc2_pop1.rank_from_coordinates( (0, 2) ), 2)
        self.assertEqual( tc2_pop1.rank_from_coordinates( (2, 0) ), 6)

    #
    # Parameters
    #
    def test_get_tau(self):
        """
        test access to parameter
        """
        self.assertTrue(numpy.allclose(tc2_pop1.tau, [[ 10.,  10.,  10.],
                                                      [ 10.,  10.,  10.],
                                                      [ 10.,  10.,  10.]]))

    def test_get_tau2(self):
        """
        test access to parameter
        """
        self.assertTrue(numpy.allclose(tc2_pop1.get('tau'), [[ 10.,  10.,  10.],
                                                             [ 10.,  10.,  10.],
                                                             [ 10.,  10.,  10.]]))

    def test_get_neuron_tau(self):
        """
        tests access to single specific neurons in the *Population* (tau)
        """

        self.assertTrue(numpy.allclose(tc2_pop1.neuron(1).tau, 10.0))
        

    def test_set_tau(self):
        """

        Assigned a new value, all instances will change
        """
        tc2_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(tc2_pop1.tau, [[ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.]]))
            
    def test_set_tau_2(self):
        """
        Assigned a new value, all instances will change
        """
        tc2_pop1.set({'tau' : 5.0})
        self.assertTrue(numpy.allclose(tc2_pop1.tau, [[ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.]]))

    def test_set_tau_popview(self):
        """
        Assigned a new value, all instances will change normally. 
        One can use *PopulationView* to update more specific
        """

        tc2_pop1[1:3, 1].tau = 5.0
        self.assertTrue(numpy.allclose(tc2_pop1.tau, [[ 10.,  10.,  10.],
                                                      [ 10.,  5.,  10.],
                                                      [ 10.,  5.,  10.]]))


    def test_get_tau_population(self):
        """
        test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(tc2_pop2.tau, 10.0)

    def test_popattributes(self):
        """
        tests listing *Population* attributes
        """
        self.assertEqual(tc2_pop1.attributes, ['tau', 'r'], 'failed listing attributes')
        self.assertEqual(tc2_pop1.parameters, ['tau'], 'failed listing parameters')
        self.assertEqual(tc2_pop1.variables, ['r'], 'failed listing variables')
    


    #
    # Variables
    #
    def test_get_r(self):
        """
        default all variables are initialized with zero
        """
        self.assertTrue(numpy.allclose(tc2_pop1.r, [[ 0.,  0.,  0.],
                                                    [ 0.,  0.,  0.],
                                                    [ 0.,  0.,  0.]]))

    def test_get_r2(self):
        """
        tests getting method
        """
        self.assertTrue(numpy.allclose(tc2_pop1.get('r'), [[ 0.,  0.,  0.],
                                                           [ 0.,  0.,  0.],
                                                           [ 0.,  0.,  0.]]))

    def test_get_neuron_r(self):
        """
        tests access to single specific neurons in the *Population* (r)
        """

        self.assertTrue(numpy.allclose(tc2_pop1.neuron(0).r, 0.0))


        

    def test_get_r_with_init(self):
        """
        default all variables are initialized with zero, we now
        modified this with init = 1.0
        """
        self.assertTrue(numpy.allclose(tc2_pop2.r, [[ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.]]))

    def test_set_r(self):
        """
        tests setting of variable
        """
        tc2_pop1.r=1.0
        self.assertTrue(numpy.allclose(tc2_pop1.r, [[ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.]]))
        
        tc2_pop1[1:3, 1].r=2.0
        self.assertTrue(numpy.allclose(tc2_pop1.r, [[ 1.,  1.,  1.],
                                                    [ 1.,  2.,  1.],
                                                    [ 1.,  2.,  1.]]))
    
        tc2_pop1.r=2.0

        tc2_pop1.r=Uniform(0.0, 1.0).get_values(9)
        self.assertTrue(any(tc2_pop1[0:3, 0:3].r>=0.0) and all(tc2_pop1[0:3, 0:3].r<=1.0))


    def test_set_r2(self):
        """

        tests the setting method
        """
        tc2_pop1.set({'r': 1.0})
        self.assertTrue(numpy.allclose(tc2_pop1.r, [[ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.],
                                                    [ 1.,  1.,  1.]]))


    #
    #Reset-Test
    #
    
    def test_reset(self):
        """
        tests if *Population* is properly reset if reset() is called
        """
        tc2_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(tc2_pop1.tau, [[ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.],
                                                      [ 5.,  5.,  5.]]))
        reset()
        self.assertTrue(numpy.allclose(tc2_pop1.tau, [[ 10.,  10.,  10.],
                                                      [ 10.,  10.,  10.],
                                                      [ 10.,  10.,  10.]]))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class test_Population3D(unittest.TestCase):
    """
    Test *Population* data storage, access methods three-dimensional populations.
    """
    def setUp(self):
        reset()

    #
    # Coordinate transformations
    #
    def test_coordinates_from_rank(self):
        """

        tests coordinates_from_rank method:
        
        * largest x in first line
        * largest y in first column
        * largest z in first depth
        """

        self.assertSequenceEqual(tc3_pop1.coordinates_from_rank(2), (0, 0, 2))
        self.assertSequenceEqual(tc3_pop1.coordinates_from_rank(6), (0, 2, 0))
        self.assertSequenceEqual(tc3_pop1.coordinates_from_rank(18), (2, 0, 0))


    def test_rank_from_coordinates(self):
        """
        tests rank_from_coordinates method:


        * largest x in first line
        * largest y in first column
        * largest z in first depth
        """
        self.assertEqual( tc3_pop1.rank_from_coordinates( (0, 0, 2) ), 2)
        self.assertEqual( tc3_pop1.rank_from_coordinates( (0, 2, 0) ), 6)
        self.assertEqual( tc3_pop1.rank_from_coordinates( (2, 0, 0) ), 18)


    #
    # Parameters
    #
    def test_get_tau(self):
        """
        test access to parameter
        """

        self.assertTrue(numpy.allclose(tc3_pop1.tau, [[[ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.]],
                                                      [[ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.]],
                                                      [[ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.]]]))

    def test_get_tau2(self):
        """
        test access to parameter
        """
        self.assertTrue(numpy.allclose(tc3_pop1.get('tau'), [[[ 10.,  10.,  10.],
                                                              [ 10.,  10.,  10.],
                                                              [ 10.,  10.,  10.]],
                                                             [[ 10.,  10.,  10.],
                                                              [ 10.,  10.,  10.],
                                                              [ 10.,  10.,  10.]],
                                                             [[ 10.,  10.,  10.],
                                                              [ 10.,  10.,  10.],
                                                              [ 10.,  10.,  10.]]]))

    def test_get_neuron_tau(self):
        """
        tests access to single specific neurons in the *Population* (tau)
        """

        self.assertTrue(numpy.allclose(tc3_pop1.neuron(1).tau, 10.0))
        

    def test_set_tau(self):
        """

        Assigned a new value, all instances will change
        """
        tc3_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(tc3_pop1.tau, [[[ 5.,  5.,  5.],
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
        Assigned a new value, all instances will change

        """
        tc3_pop1.set({'tau' : 5.0})
        self.assertTrue(numpy.allclose(tc3_pop1.tau, [[[ 5.,  5.,  5.],
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
        One can use *PopulationView* to update more specific
        """

        tc3_pop1[0:3, 1, 1:3].tau = 5.0
        self.assertTrue(numpy.allclose(tc3_pop1.tau, [[[ 10.,  10.,  10.],
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
        test access to parameter, modified with *Population* keyword, as
        consequence there should be only one instance of tau.
        """
        self.assertEqual(tc3_pop2.tau, 10.0)

    def test_popattributes(self):
        """
        tests listing *Population* attributes
        """
        self.assertEqual(tc3_pop1.attributes, ['tau', 'r'], 'failed listing attributes')
        self.assertEqual(tc3_pop1.parameters, ['tau'], 'failed listing parameters')
        self.assertEqual(tc3_pop1.variables, ['r'], 'failed listing variables')
    


    #
    # Variables
    #
    def test_get_r(self):
        """
        default all variables are initialized with zero
        """
        self.assertTrue(numpy.allclose(tc3_pop1.r, [[[ 0.,  0.,  0.],
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
        tests getting method
        """
        self.assertTrue(numpy.allclose(tc3_pop1.get('r'), [[[ 0.,  0.,  0.],
                                                            [ 0.,  0.,  0.],
                                                            [ 0.,  0.,  0.]],
                                                           [[ 0.,  0.,  0.],
                                                            [ 0.,  0.,  0.],
                                                            [ 0.,  0.,  0.]],
                                                           [[ 0.,  0.,  0.],
                                                            [ 0.,  0.,  0.],
                                                            [ 0.,  0.,  0.]]]))

    def test_get_neuron_r(self):
        """
        tests access to single specific neurons in the *Population* (r)

        """

        self.assertTrue(numpy.allclose(tc3_pop1.neuron(18).r, 0.0))


        

    def test_get_r_with_init(self):
        """
        default all variables are initialized with zero, we now
        modified this with init = 1.0
        """
        self.assertTrue(numpy.allclose(tc3_pop2.r, [[[ 1.,  1.,  1.],
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
        tests setting of variable
        """
        tc3_pop1.r=1.0
        self.assertTrue(numpy.allclose(tc3_pop1.r, [[[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]]]))
        
        tc3_pop1[0:3, 1, 1:3].r=2.0

        self.assertTrue(numpy.allclose(tc3_pop1.r, [[[ 1.,  1.,  1.],
                                                     [ 1.,  2.,  2.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  2.,  2.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  2.,  2.],
                                                     [ 1.,  1.,  1.]]]))
    
        
        tc3_pop1.r=2.0
        tc3_pop1.r=Uniform(0.0, 1.0).get_values(27)
        self.assertTrue(any(tc3_pop1[0:3, 0:3, 0:3].r>=0.0) and all(tc3_pop1[0:3, 0:3, 0:3].r<=1.0))


    def test_set_r2(self):
        """
        tests the setting method
        """
        tc3_pop1.set({'r': 1.0})
        self.assertTrue(numpy.allclose(tc3_pop1.r, [[[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]],
                                                    [[ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.],
                                                     [ 1.,  1.,  1.]]]))


    #
    #Reset-Test
    #
    
    def test_reset(self):
        """
        tests if *Population* is properly reset if reset() is called
        """
        tc3_pop1.tau = 5.0
        self.assertTrue(numpy.allclose(tc3_pop1.tau, [[[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]],
                                                      [[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]],
                                                      [[ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.],
                                                       [ 5.,  5.,  5.]]]))
        reset()
        self.assertTrue(numpy.allclose(tc3_pop1.tau, [[[ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.]],
                                                      [[ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.]],
                                                      [[ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.],
                                                       [ 10.,  10.,  10.]]]))


