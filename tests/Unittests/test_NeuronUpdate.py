"""

    test_NeuronUpdate.py

    This file is part of ANNarchy.

    Copyright (C) 2016-2018 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import Neuron, Population, Network, Monitor, set_seed
set_seed(seed=1)

class test_NeuronUpdate(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        # neuron defintions common used for test cases
        local_eq = Neuron(
            equations="""
                noise = Uniform(0,1)
            	    r = t
            """
        )

        global_eq = Neuron(
            equations="""
                noise = Uniform(0,1) : population
                glob_r = t : population
                r = t
            """
        )

        mixed_eq = Neuron(
            parameters="glob_par = 1.0: population",
            equations="""
                r = t + glob_par
            """
        )

        bound_eq = Neuron(
            parameters="""
                min_r=1.0: population
                max_r=3.0: population
            """,
            equations="""
                r = t : min=min_r, max=max_r
            """
        )

        tc_loc_up_pop = Population(3, local_eq)
        tc_glob_up_pop = Population(3, global_eq)
        tc_mixed_up_pop = Population(3, mixed_eq)
        tc_bound_up_pop = Population(3, bound_eq)

        m = Monitor(tc_bound_up_pop, 'r')

        cls.test_net = Network()
        cls.test_net.add([tc_loc_up_pop, tc_glob_up_pop,
                          tc_mixed_up_pop, tc_bound_up_pop, m])
        cls.test_net.compile(silent=True)

        cls.net_loc_pop = cls.test_net.get(tc_loc_up_pop)
        cls.net_glob_pop = cls.test_net.get(tc_glob_up_pop)
        cls.net_mix_pop = cls.test_net.get(tc_mixed_up_pop)
        cls.net_bound_pop = cls.test_net.get(tc_bound_up_pop)
        cls.net_m = cls.test_net.get(m)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self.test_net.reset() # network reset

    def test_single_update_local(self):
        """
        Test the update of a local equation.
        """
        self.test_net.simulate(5)

        # after 5ms simulation, r should be at 4
        self.assertTrue(numpy.allclose(self.net_loc_pop.r, [4.0, 4.0, 4.0]))

    def test_single_update_global(self):
        """
        Test the update of a global equation.
        """
        self.test_net.simulate(5)

        # after 5ms simulation, glob_r should be at 4
        self.assertTrue(numpy.allclose(self.net_glob_pop.glob_r, [4.0]))

    def test_single_update_mixed(self):
        """
        Test the update of a local equation which depends on a global parameter.
        """
        self.test_net.simulate(5)

        # after 5ms simulation, glob_r should be at 4 + glob_var lead to 5
        self.assertTrue(numpy.allclose(self.net_mix_pop.r, [5.0]))

    def test_bound_update(self):
        """
        Test the update of a local equation and given boundaries.
        """
        self.test_net.simulate(5)

        r = self.net_m.get('r')
        self.assertTrue(numpy.allclose(r[:,0], [1,1,2,3,3]))

class test_GlobalOps_1D(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects. Currently the following methods
    are supported:

        * mean()
        * max()
        * min()
        * norm1()
        * norm2()

    They are used in the equations of our neuron definition.
    This particular test focuses on a one-dimensional *Population*.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r=0
            """,
            equations="""
                mean_r = mean(r)
                max_r = max(r)
                min_r = min(r)
                l1 = norm1(r)
                l2 = norm2(r)
            """
        )

        pop = Population(6, neuron)

        self.test_net = Network()
        self.test_net.add([pop])
        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(pop)

    @classmethod
    def tearDownClass(cls):
        del cls.test_net

    def setUp(self):
        """
        In our *setUp()* function we set the variable *r*.
        We also call *simulate()* to calculate mean/max/min.
        """
        # reset() set all variables to init value (default 0), which is
        # unfortunately meaningless for mean/max/min. So we set here some
        # better values
        self.net_pop.r = [2.0, 1.0, 0.0, -5.0, -3.0, -1.0]

        # 1st step: calculate mean/max/min and store in intermediate
        #           variables
        # 2nd step: write intermediate variables to accessible variables.
        self.test_net.simulate(2)

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def test_get_mean_r(self):
        """
        Tests the result of *mean(r)* for *pop*.
        """
        self.assertTrue( numpy.allclose( self.net_pop.mean_r, -1.0 ) )

    def test_get_max_r(self):
        """
        Tests the result of *max(r)* for *pop*.
        """
        self.assertTrue( numpy.allclose( self.net_pop.max_r, 2.0) )

    def test_get_min_r(self):
        """
        Tests the result of *min(r)* for *pop*.
        """
        self.assertTrue( numpy.allclose( self.net_pop.min_r, -5.0) )

    def test_get_l1_norm(self):
        """
        Tests the result of *norm1(r)* (L1 norm) for *pop*.
        """
        self.assertTrue(numpy.allclose( self.net_pop.l1, 12.0))

    def test_get_l2_norm(self):
        """
        Tests the result of *norm2(r)* (L2 norm) for *pop*.
        """
        # compute control value
        l2norm = numpy.linalg.norm( self.net_pop.r, ord=2)

        # test
        self.assertTrue(numpy.allclose( self.net_pop.l2, l2norm))

class test_GlobalOps_2D(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects. Currently the following methods
    are supported:

        * mean()
        * max()
        * min()
        * norm1()
        * norm2()

    They are used in the equations of our neuron definition.
    This particular test focuses on a two-dimensional *Population*.
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r=0
            """,
            equations="""
                mean_r = mean(r)
                max_r = max(r)
                min_r = min(r)
                l1 = norm1(r)
                l2 = norm2(r)
            """
        )

        pop = Population((2, 3), neuron)

        self.test_net = Network()
        self.test_net.add([pop])

        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(pop)

    @classmethod
    def tearDownClass(cls):
        del cls.test_net

    def setUp(self):
        """
        In our *setUp()* function we set the variable *r*.
        We also call *simulate()* to calculate mean/max/min.
        """
        # reset() set all variables to init value (default 0), which is
        # unfortunately meaningless for mean/max/min. So we set here some
        # better values
        self.net_pop.r = [[ 2.0,  1.0,  0.0],
                          [-5.0, -3.0, -1.0]]

        # 1st step: calculate mean/max/min and store in intermediate
        #           variables
        # 2nd step: write intermediate variables to accessible variables.
        self.test_net.simulate(2)

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def test_get_mean_r(self):
        """
        Tests the result of *mean(r)* for *pop*.
        """
        self.assertTrue( numpy.allclose( self.net_pop.mean_r, -1.0 ) )

    def test_get_max_r(self):
        """
        Tests the result of *max(r)* for *pop*.
        """
        self.assertTrue( numpy.allclose( self.net_pop.max_r, 2.0) )

    def test_get_min_r(self):
        """
        Tests the result of *min(r)* for *pop*.
        """
        self.assertTrue( numpy.allclose( self.net_pop.min_r, -5.0) )

    def test_get_l1_norm(self):
        """
        Tests the result of *norm1(r)* for *pop*.
        """
        self.assertTrue(numpy.allclose( self.net_pop.l1, 12.0))
