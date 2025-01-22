"""

    test_BuiltinFunctions.py

    This file is part of ANNarchy.

    Copyright (C) 2016-2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import Monitor, Network, Neuron, Population

class test_BuiltinFunctions(unittest.TestCase):
    """
    Test the correct evaluation of builtin functions
    """
    @classmethod
    def setUpClass(cls):
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
                clip_below = clip(-2, -1, 1)
                clip_within = clip(0, -1, 1)
                clip_above = clip(2, -1, 1)
            """
        )

        cls._network = Network()
        pop1 = cls._network.create(geometry=1, neuron=BuiltinFuncs)
        cls._test_mon = cls._network.monitor(pop1, ['r', 'pr', 'clip_below', 'clip_within', 'clip_above'])
        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self._network.reset()

    def tearDown(self):
        """
        Since all tests are independent, after every test we use the *get()* method for every monotor to clear all recordings.
        """
        self._test_mon.get()

    def test_modulo(self):
        """
        Test modulo function.
        """
        self._network.simulate(10)
        data_m = self._test_mon.get('r')
        numpy.testing.assert_allclose(data_m, [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0]])

    def test_integer_power(self):
        """
        Test integer power function.
        """
        self._network.simulate(1)
        data_m = self._test_mon.get('pr')
        numpy.testing.assert_allclose(data_m, [[8.0]])

    def test_clip_below(self):
        """
        The clip(x, a, b) method ensures that x is within range [a,b]. This tests validates that x = -2 is clipped to -1
        """
        self._network.simulate(1)
        data_clip_below = self._test_mon.get('clip_below')
        numpy.testing.assert_allclose(data_clip_below, [[-1.0]])

    def test_clip_within(self):
        """
        The clip(x, a, b) method ensures that x is within range [a,b]. This tests validates that x = 0 retains.
        """
        self._network.simulate(1)
        data_clip_within = self._test_mon.get('clip_within')
        numpy.testing.assert_allclose(data_clip_within, [[0.0]])

    def test_clip_above(self):
        """
        The clip(x, a, b) method ensures that x is within range [a,b]. This tests validates that x = 2 is clipped to 1.
        """
        self._network.simulate(1)
        data_clip_above = self._test_mon.get('clip_above')
        numpy.testing.assert_allclose(data_clip_above, [[1.0]])
