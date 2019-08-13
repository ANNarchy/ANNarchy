"""

    test_BuiltinFunctions.py

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

from ANNarchy import *

class test_BuiltinFunctions(unittest.TestCase):
    """
    Test the correct evaluation of builtin functions
    """
    @classmethod
    def setUpClass(self):
        """
        Compile the network for this test
        """
        BuiltinFuncs = Neuron(
            parameters="""
                base = 2.0
            """,
            equations = """
                r = modulo(t,3)
                pr = power(base,3)
            """
        )

        pop1 = Population(1, BuiltinFuncs)
        mon = Monitor(pop1, ['r', 'pr'])

        self.test_net = Network()
        self.test_net.add([pop1, mon])
        self.test_net.compile(silent=True)

        self.test_mon = self.test_net.get(mon)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset()

    def tearDown(self):
        """
        Since all tests are independent, after every test we use the *get()* method for every monotor to clear all recordings.
        """
        self.test_mon.get()

    def test_modulo(self):
        """
        Test modulo function.
        """
        self.test_net.simulate(10)
        data_m = self.test_mon.get()
        self.assertTrue(np.allclose(data_m['r'], [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0]]))

    def test_integer_power(self):
        """
        Test integer power function.
        """
        self.test_net.simulate(1)
        data_m = self.test_mon.get()
        self.assertTrue(np.allclose(data_m['pr'], [[8.0]]))
