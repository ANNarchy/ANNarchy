"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
import numpy
from ANNarchy import Neuron, Network

class test_ITE(unittest.TestCase):
    """
    Test the correct evaluation of if-then-else (ITE) statements.

    TODO: until now, only the successful compilation is tested, some value
    tests need to be added ...
    """
    @classmethod
    def setUpClass(cls):
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

        cls._network = Network()

        cls.net_pop1 = cls._network.create(geometry=1, neuron=SimpleITE)
        cls.net_pop2 = cls._network.create(geometry=1, neuron=NestedITE)
        cls.net_pop3 = cls._network.create(geometry=1, neuron=ITEinODE)
        cls.net_pop4 = cls._network.create(geometry=1, neuron=SimpleITE2)
        cls.net_pop5 = cls._network.create(geometry=1, neuron=NestedITE2)
        cls.net_pop6 = cls._network.create(geometry=1, neuron=ITEinODE2)

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
        self._network.reset() # network reset

    def test_works(self):
        """
        Tests if the ITE statements worked..
        """
        self._network.simulate(10)

        numpy.testing.assert_allclose(self.net_pop1.r[0], 0.0)
        numpy.testing.assert_allclose(self.net_pop2.r[0], 2.0)
        numpy.testing.assert_allclose(self.net_pop3.r[0], 4.0)
        numpy.testing.assert_allclose(self.net_pop4.r[0], 0.0)
        numpy.testing.assert_allclose(self.net_pop5.r[0], 2.0)
        numpy.testing.assert_allclose(self.net_pop6.r[0], 4.0)
