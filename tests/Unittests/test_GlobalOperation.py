"""

    test_GlobalOperation.py

    This file is part of ANNarchy.

    Copyright (C) 2018-2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
    Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>

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
#
# Originally wrote by JG
#
# Changes 2018 (HD):
#
# - model definition is now part of each test class instead of global definition
# - added test case for usage in synaptic equations
import unittest
import numpy

from ANNarchy import *

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

        self.test_net = Network(True)

        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(pop)

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
        l2norm = np.linalg.norm( self.net_pop.r, ord=2)

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

        self.test_net = Network(True)

        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(pop)

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

class test_SynapticAccess(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects.

    This particular test focuses on the usage of them in synaptic learning rules
    (for instance covariance).
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r=0
            """
        )

        cov = Synapse(
            parameters="""
                tau = 5000.0
            """,
            equations="""
                tau * dw/dt = (pre.r - mean(pre.r) ) * (post.r - mean(post.r) )
            """
        )

        pre = Population(6, neuron)
        post = Population(1, neuron)
        proj = Projection(pre, post, "exc", synapse = cov).connect_all_to_all(weights=1.0)

        self.test_net = Network(True)

        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(post)

    def test_compile(self):
        """
        Tests the result of *norm1(r)* for *pop*.
        """
        pass
