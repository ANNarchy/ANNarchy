"""

    test_RateTransmission.py

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

neuron = Neuron(
    equations="r = 1+t: init = -1"
)

neuron2 = Neuron(
    equations="""
	sum1 = sum(one2one)
        sum2 = sum(all2all)
        sum3 = sum(del_one2one)
        r =  sum1 + sum2 + sum3
    """
)

pop1 = Population((3, 3), neuron)
pop2 = Population((3, 3), neuron2)

proj = Projection(
     pre = pop1,
     post = pop2,
     target = "one2one"
)

proj2 = Projection(
     pre = pop1,
     post = pop2,
     target = "all2all"
)


proj3 = Projection(
     pre = pop1,
     post = pop2,
     target = "del_one2one"
)

proj.connect_one_to_one(weights = 1.0)
proj2.connect_all_to_all(weights = 1.0)
proj3.connect_one_to_one(weights = 1.0, delays = 10.0)

compile(clean=True, silent=True)


class test_RateTransmission(unittest.TestCase):

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        reset()

    def test_one_to_one(self):
        """
        tests functionality of the one_to_one connectivity pattern
        """
        # sum up r = 1
        simulate(2)
        self.assertTrue(numpy.allclose(pop2.sum1, 1.0))

    def test_all_to_all(self):
        """
        tests functionality of the all_to_all connectivity pattern
        """
        # sum up r = 1, 9 neurons
        simulate(2)
        self.assertTrue(numpy.allclose(pop2.sum2, 9.0))

    def test_delay(self):
        """
        tests the delay functionality. 
        """
        # The first ten steps, we have 
        # initialization value
        simulate(10)
        self.assertTrue(numpy.allclose(pop2.sum3, -1.0))

        # at 11th step we have the first queue 
        # value in our case t = 0
        simulate(1)
        self.assertTrue(numpy.allclose(pop2.sum3, 1.0))

        # at 20th -> t = 9
        simulate(9)
        self.assertTrue(numpy.allclose(pop2.sum3, 10.0))

if __name__ == '__main__':
    unittest.main()