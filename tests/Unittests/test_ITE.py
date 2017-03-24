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
                state = 0.0
            """,
            equations = """
                r = if (state >= 2): 2 else: if (state == 1): 1 else: 0
            """
        )

        ITEinODE = Neuron(
            equations = """
                dr/dt = if t < 10: 1 else: 0
            """
        )

        pop = Population(1, SimpleITE)
        pop2 = Population(1, NestedITE)
        pop3 = Population(1, ITEinODE)

        self.test_net = Network()
        self.test_net.add([pop, pop2])
        self.test_net.compile(silent=True)

        self.net_pop = self.test_net.get(pop)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset() # network reset

