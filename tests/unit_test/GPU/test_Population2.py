"""

    test_Population2.py

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

compile(clean=True)
pop1.r = [2.0, 1.0, 0.0, -5.0, -3.0, -1.0]
simulate(2)


class test_Population2(unittest.TestCase):


    def test_get_mean_r(self):
        """
        tests access to mean_r
        """
        self.assertTrue(numpy.allclose(pop1.mean_r, -1.0))

    def test_get_max_r(self):
        """
        tests access to max_r
        """
        self.assertTrue(numpy.allclose(pop1.max_r, 2.0))

    def test_get_min_r(self):
        """
        tests access to min_r
        """
        self.assertTrue(numpy.allclose(pop1.min_r, -5.0))

    def test_get_l1_norm(self):
        """
        tests access to l1_norm_r
        """
        self.assertTrue(numpy.allclose(pop1.l1, -6.0))

    def test_get_l2_norm(self):
        """
        tests access to l2_norm_r
        """
        # compute control value
        l2norm = np.linalg.norm(pop1.r, ord=2)

        # test
        self.assertTrue(numpy.allclose(pop1.l2, l2norm))
