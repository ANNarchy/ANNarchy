"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron, DiscreteUniform


class test_SingleCondition(unittest.TestCase):
    """
    Setup 1: multiple populations, only one tracked.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        simple_t = Neuron(
            equations="r += 1",
        )

        cls._network = Network()

        cls._pop1 = cls._network.create(geometry=15, neuron=simple_t)
        cls._pop2 = cls._network.create(
            geometry=15, neuron=simple_t, stop_condition="r > 5.0 : any"
        )
        cls._pop2.r = DiscreteUniform(-4, -2, rng=numpy.random.default_rng(56789))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()

    def test_early_stopping(self):
        """
        The second population will stop the simulation even though the first population
        fulfills the condition earlier.
        """
        stopped_at = self._network.simulate_until(
            max_duration=15.0, population=self._pop2
        )
        self.assertEqual(stopped_at, 8.0)


class test_TwoConditionWithOr(unittest.TestCase):
    """
    Setup 2: multiple populations, both tracked. The conditions are linked with 'or'
    operator.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        simple_t = Neuron(
            equations="r += 1",
        )

        cls._network = Network()

        cls._pop1 = cls._network.create(
            geometry=15, neuron=simple_t, stop_condition="r > 5.0 : any"
        )
        cls._pop2 = cls._network.create(
            geometry=15, neuron=simple_t, stop_condition="r > 5.0 : any"
        )
        cls._pop2.r = DiscreteUniform(-4, -2, rng=numpy.random.default_rng(56789))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()

    def test_early_stopping(self):
        """
        The first population will stop the simulation as it fulfills the condition earlier.
        """
        stopped_at = self._network.simulate_until(
            max_duration=15.0, population=[self._pop1, self._pop2], operator="or"
        )
        self.assertEqual(stopped_at, 6.0)


class test_TwoConditionWithAnd(unittest.TestCase):
    """
    Setup 2: multiple populations, both tracked. The conditions are linked with 'and'
    operator.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        simple_t = Neuron(
            equations="r += 1",
        )

        cls._network = Network()

        cls._pop1 = cls._network.create(
            geometry=15, neuron=simple_t, stop_condition="r > 5.0 : any"
        )
        cls._pop2 = cls._network.create(
            geometry=15, neuron=simple_t, stop_condition="r > 5.0 : any"
        )
        cls._pop2.r = DiscreteUniform(-4, -2, rng=numpy.random.default_rng(56789))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()

    def test_early_stopping(self):
        """
        The second population will stop the simulation as both conditions must be fulfilled.
        """
        stopped_at = self._network.simulate_until(
            max_duration=15.0, population=[self._pop1, self._pop2], operator="and"
        )
        self.assertEqual(stopped_at, 8.0)
