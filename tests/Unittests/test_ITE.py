"""

    test_ITE.py

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
import numpy as np

class test_ITE(unittest.TestCase):
    """
    Test the correct evaluation of if-then-else (ITE) statements.

    TODO: until now, only the successful compilation is tested, some value tests need to be added ...
    """
    @classmethod
    def setUpClass(self):
        """
        Setup and compile the network for this tests
        """
        SimpleITE = Neuron(
            equations = """
                r = if (t is 1): 1.0 else: 0.0
            """
        )

        NestedITE = Neuron(
            parameters = """
                state = 1.0
            """,
            equations = """
                r = if (t <= 5): 
                        0 
                    else: 
                        if (t >= 7):  
                            2
                        else: 
                            1
            """
        )

        ITEinODE = Neuron(
            equations = """
                dr/dt = if t > 5: 1 else: 0
            """
        )

        SimpleITE2 = Neuron(
            equations = """
                r = ite(t is 1, 1.0, 0.0)
            """
        )

        NestedITE2 = Neuron(
            parameters = """
                state = 2.0
            """,
            equations = """
                r = ite(state >= 2, 2, ite(state == 1, 1, 0))
            """
        )

        ITEinODE2 = Neuron(
            equations = """
                dr/dt = ite((t > 5) and (t <20), 1, 0)
            """
        )

        pop1 = Population(1, SimpleITE)
        pop2 = Population(1, NestedITE)
        pop3 = Population(1, ITEinODE)
        pop4 = Population(1, SimpleITE2)
        pop5 = Population(1, NestedITE2)
        pop6 = Population(1, ITEinODE2)

        self.test_net = Network()
        self.test_net.add([pop1, pop2, pop3, pop4, pop5, pop6])
        self.test_net.compile(silent=True)

        self.net_pop1 = self.test_net.get(pop1)
        self.net_pop2 = self.test_net.get(pop2)
        self.net_pop3 = self.test_net.get(pop3)
        self.net_pop4 = self.test_net.get(pop4)
        self.net_pop5 = self.test_net.get(pop5)
        self.net_pop6 = self.test_net.get(pop6)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset() # network reset

    def test_works(self):
        """
        Tests if the ITE statements worked..
        """
        self.test_net.simulate(10)

        self.assertTrue(np.allclose(self.net_pop1.r[0], 0.0))
        self.assertTrue(np.allclose(self.net_pop2.r[0], 2.0))
        self.assertTrue(np.allclose(self.net_pop3.r[0], 4.0))
        self.assertTrue(np.allclose(self.net_pop4.r[0], 0.0))
        self.assertTrue(np.allclose(self.net_pop5.r[0], 2.0))
        self.assertTrue(np.allclose(self.net_pop6.r[0], 4.0))

if __name__ == '__main__':
    unittest.main()