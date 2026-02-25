"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron


class test_NeuronUpdate(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        # neuron defintions common used for test cases
        local_eq = Neuron(
            equations="""
                noise = Uniform(0,1)
            	    r = t
            """
        )

        global_eq = Neuron(
            equations="""
                noise = Uniform(0,1) : population
                glob_r = t : population
                r = t
            """
        )

        mixed_eq = Neuron(
            parameters="glob_par = 1.0: population",
            equations="""
                r = t + glob_par
            """,
        )

        bound_eq = Neuron(
            parameters="""
                min_r=1.0: population
                max_r=3.0: population
            """,
            equations="""
                r = t : min=min_r, max=max_r
            """,
        )

        cls._network = Network()

        cls._local_attr = cls._network.create(3, local_eq)
        cls._global_attr = cls._network.create(3, global_eq)
        cls._multi_attr = cls._network.create(3, mixed_eq)
        cls._bound_attr = cls._network.create(3, bound_eq)

        cls.net_m = cls._network.monitor(cls._bound_attr, "r")

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """Delete class instance."""
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset()  # network reset

    def test_single_update_local(self):
        """
        Test the update of a local equation.
        """
        self._network.simulate(5)

        # after 5ms simulation, r should be at 4
        numpy.testing.assert_allclose(self._local_attr.r, [4.0, 4.0, 4.0])

    def test_single_update_global(self):
        """
        Test the update of a global equation.
        """
        self._network.simulate(5)

        # after 5ms simulation, glob_r should be at 4
        numpy.testing.assert_allclose(self._global_attr.glob_r, [4.0])

    def test_single_update_mixed(self):
        """
        Test the update of a local equation which depends on a global parameter.
        """
        self._network.simulate(5)

        # after 5ms simulation, glob_r should be at 4 + glob_var lead to 5
        numpy.testing.assert_allclose(self._multi_attr.r, [5.0, 5.0, 5.0])

    def test_bound_update(self):
        """
        Test the update of a local equation and given boundaries.
        """
        self._network.simulate(5)

        r = self.net_m.get("r")
        numpy.testing.assert_allclose(r[:, 0], [1, 1, 2, 3, 3])
