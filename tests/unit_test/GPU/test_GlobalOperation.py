"""

    test_GlobalOperation.py

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

neuron = Neuron(
    parameters = """
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

pop1 = Population (6, neuron)
pop2 = Population ((2, 3), neuron)

compile(clean=True)

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

    def setUp(self):
        """
        In our *setUp()* function we set the variable *r*.
        We also call *simulate()* to calculate mean/max/min.
        """
        # reset() set all variables to init value (default 0), which is 
        # unfortunatly meaningless for mean/max/min. So we set here some
        # better values
        pop1.r = [2.0, 1.0, 0.0, -5.0, -3.0, -1.0]

        # 1st step: calculate mean/max/min and store in intermediate
        #           variables
        # 2nd step: write intermediate variables to accessible variables.
        simulate(2)

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        reset()

    def test_get_mean_r(self):
        """
        Tests the result of *mean(r)* for *pop1*.
        """
        self.assertTrue( numpy.allclose( pop1.mean_r, -1.0 ) )

    def test_get_max_r(self):
        """
        Tests the result of *max(r)* for *pop1*.
        """
        self.assertTrue( numpy.allclose(pop1.max_r, 2.0) )

    def test_get_min_r(self):
        """
        Tests the result of *min(r)* for *pop1*.
        """
        self.assertTrue( numpy.allclose(pop1.min_r, -5.0) )

    def test_get_l1_norm(self):
        """
        Tests the result of *norm1(r)* (L1 norm) for *pop1*.
        """
        self.assertTrue(numpy.allclose(pop1.l1, 12.0))

    def test_get_l2_norm(self):
        """
        Tests the result of *norm2(r)* (L2 norm) for *pop1*.
        """
        # compute control value
        l2norm = np.linalg.norm(pop1.r, ord=2)

        # test
        self.assertTrue(numpy.allclose(pop1.l2, l2norm))


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

    def setUp(self):
        """
        In our *setUp()* function we set the variable *r*.
        We also call *simulate()* to calculate mean/max/min.
        
        """
        # reset() set all variables to init value (default 0), which is 
        # unfortunatly meaningless for mean/max/min. So we set here some
        # better values
        pop2.r = [[2.0, 1.0, 0.0],
                  [-5.0, -3.0, -1.0]]

        # 1st step: calculate mean/max/min and store in intermediate
        #           variables
        # 2nd step: write intermediate variables to accessible variables.
        simulate(2)

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        reset()

    def test_get_mean_r(self):
        """
        Tests the result of *mean(r)* for *pop2*.
        """       
        self.assertTrue( numpy.allclose( pop2.mean_r, -1.0 ) )

    def test_get_max_r(self):
        """
        Tests the result of *max(r)* for *pop2*.
        """
        self.assertTrue( numpy.allclose(pop2.max_r, 2.0) )

    def test_get_min_r(self):
        """
        Tests the result of *min(r)* for *pop2*.
        """
        self.assertTrue( numpy.allclose(pop2.min_r, -5.0) )
        
    def test_get_l1_norm(self):
        """
        Tests the result of *norm1(r)* for *pop2*.
        """
        self.assertTrue(numpy.allclose(pop2.l1, 12.0))

    def test_get_l2_norm(self):
        """
        Tests the result of *norm2(r)* for *pop2*.
        """
        # compute control value
        l2norm = np.linalg.norm(pop2.r, ord=2)
        
        # test
        self.assertTrue(numpy.allclose(pop2.l2, l2norm))
