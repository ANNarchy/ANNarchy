"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron

class test_IndividualNeuron(unittest.TestCase):
    """
    This class tests the functionality of the *IndividualNeuron* object, which
    hold references to one specific neuron of a *Population*.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters = "tau = 10",
            equations="r += 1/tau * t"
        )

        cls._network = Network()
        cls._population = cls._network.create(geometry=(8, 8), neuron=neuron)
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network before
        every test.
        """
        self._network.reset()

    def test_get_r(self):
        """
        Tests the direct access of the variable *r* of a *IndividualNeuron*
        object.
        """
        numpy.testing.assert_allclose((self._population[2, 2] +
                                       self._population[3, 3] +
                                       self._population[4, 4]).r, [0.0, 0.0, 0.0])

    def test_set_r(self):
        """
        Tests the setting of *r* through direct access.
        """
        (self._population[2, 2] + self._population[3, 3] + self._population[4, 4]).r = 1.0
        numpy.testing.assert_allclose((self._population[2, 2] +
                                       self._population[3, 3] +
                                       self._population[4, 4]).r, [1.0, 1.0, 1.0])

    def test_rank_assignment(self):
        """
        Test the correct assignment of ranks
        """
        view = self._population[2, 4]
        numpy.testing.assert_equal(view.rank, 20)   # 2 * 8 + 4