"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron, Synapse, Uniform


class test_Explicit(object):
    """
    Test the code generation for equations evaluated with explicit scheme
    """

    @classmethod
    def setUpClass(cls):
        neuron = Neuron(
            parameters="""
                tau = 10.0
                V_l = 1.0
            """,
            equations="""
                tau * dv/dt = V_l - v + g_exc
                tau * dg_exc/dt = - g_exc
            """,
            spike="v > 1",
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection", equations="tau * dw/dt = -w"
        )

        cls._network = Network()

        pop = cls._network.create(geometry=10, neuron=neuron)
        proj = cls._network.connect(pre=pop, post=pop, target="exc", synapse=synapse)
        proj.all_to_all(weights=Uniform(1.0, 2.0))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        self._network.reset()

    def test_work(self):
        self._network.simulate(1.0)


class test_Implicit(object):
    """
    Test the code generation for equations evaluated with implicit scheme
    """

    @classmethod
    def setUpClass(cls):
        neuron = Neuron(
            parameters="""
                tau = 10.0
                V_l = 1.0
            """,
            equations="""
                tau * dv/dt = V_l - v + g_exc : implicit
            """,
            spike="v > 1",
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="tau * dw/dt = -w : implicit",
        )

        cls._network = Network()

        pop = cls._network.create(geometry=10, neuron=neuron)
        proj = cls._network.connect(pre=pop, post=pop, target="exc", synapse=synapse)
        proj.all_to_all(weights=Uniform(1.0, 2.0))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        self._network.reset()

    def test_work(self):
        self._network.simulate(1.0)


class test_ImplicitCoupled(object):
    """
    Test the code generation for coupled equations evaluated with implicit scheme
    """

    @classmethod
    def setUpClass(cls):
        neuron = Neuron(
            parameters="""
                tau = 10.0
                V_l = 1.0
            """,
            equations="""
                tau * dv/dt = V_l - v + g_exc : implicit
                tau * dg_exc/dt = - g_exc : implicit
            """,
            spike="v > 1",
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="""
                tau * dw/dt = -w + u : implicit
                tau * du/dt = -u +1 : implicit
                """,
        )

        cls._network = Network()

        pop = cls._network.create(geometry=10, neuron=neuron)
        proj = cls._network.connect(pre=pop, post=pop, target="exc", synapse=synapse)
        proj.all_to_all(weights=Uniform(1.0, 2.0))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        self._network.reset()

    def test_work(self):
        self._network.simulate(1.0)


class test_Midpoint(object):
    """
    Test the code generation for equations evaluated with midpoint scheme.
    """

    @classmethod
    def setUpClass(cls):
        neuron = Neuron(
            parameters="""
                tau = 10.0
                V_l = 1.0
            """,
            equations="""
                tau * dv/dt = V_l - v + g_exc : midpoint
            """,
            spike="v > 1",
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="tau * dw/dt = -w : midpoint",
        )

        cls._network = Network()

        pop = cls._network.create(geometry=10, neuron=neuron)
        proj = cls._network.create(pre=pop, post=pop, target="exc", synapse=synapse)
        proj.all_to_all(weights=Uniform(1.0, 2.0))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        self._network.reset()

    def test_work(self):
        self._network.simulate(1.0)


class test_MidpointCoupled(object):
    """
    Test the code generation for coupled equations evaluated with midpoint scheme
    """

    @classmethod
    def setUpClass(cls):
        neuron = Neuron(
            parameters="""
                tau = 10.0
                V_l = 1.0
            """,
            equations="""
                tau * dv/dt = V_l - v + g_exc : midpoint
                tau * dg_exc/dt = - g_exc : midpoint
            """,
            spike="v > 1",
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="""
                tau * dw/dt = -w + u : midpoint
                tau * du/dt = -u +1 : midpoint
                """,
        )

        cls._network = Network()

        pop = cls._network.create(geometry=10, neuron=neuron)
        proj = cls._network.connect(pre=pop, post=pop, target="exc", synapse=synapse)
        proj.all_to_all(weights=Uniform(1.0, 2.0))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        self._network.reset()

    def test_work(self):
        self._network.simulate(1.0)


class test_Exponential(object):
    """
    Test the code generation for equations evaluated with exponential scheme
    """

    @classmethod
    def setUpClass(cls):
        neuron = Neuron(
            parameters="""
                tau = 10.0
                V_l = 1.0
            """,
            equations="""
                tau * dv/dt = V_l - v + g_exc : exponential
                tau * dg_exc/dt = - g_exc : exponential
            """,
            spike="v > 1",
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="""
                tau * dw/dt = -w + u : exponential
                tau * du/dt = -u +1 : exponential
                """,
        )

        cls._network = Network()

        pop = cls._network.create(geometry=10, neuron=neuron)
        proj = cls._network.connect(pre=pop, post=pop, target="exc", synapse=synapse)
        proj.all_to_all(weights=Uniform(1.0, 2.0))

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        self._network.reset()

    def test_work(self):
        self._network.simulate(1.0)


class test_Precision(unittest.TestCase):
    """
    Test the precision of the numerical methods for the alpha function
    """

    @classmethod
    def setUpClass(cls):
        explicit = Neuron(
            parameters="""
            tau = 10.0
            I = 0.0
            """,
            equations="""
            tau * dv/dt = -v -u + I : explicit
            tau * du/dt = - u + I : explicit
            r = pos(v)
            """,
        )

        implicit = Neuron(
            parameters="""
            tau = 10.0
            I = 0.0
            """,
            equations="""
            tau * dv/dt = -v -u + I : implicit
            tau * du/dt = - u + I : implicit
            r = pos(v)
            """,
        )

        midpoint = Neuron(
            parameters="""
            tau = 10.0
            I = 0.0
            """,
            equations="""
            tau * dv/dt = -v -u + I : midpoint
            tau * du/dt = - u + I : midpoint
            r = pos(v)
            """,
        )

        exponential = Neuron(
            parameters="""
            tau = 10.0
            I = 0.0
            """,
            equations="""
            tau * dv/dt = -v -u + I : exponential
            tau * du/dt = - u + I : exponential
            r = pos(v)
            """,
        )

        cls._network = Network()

        pop_explicit = cls._network.create(geometry=1, neuron=explicit, name="explicit")
        pop_implicit = cls._network.create(geometry=1, neuron=implicit, name="implicit")
        pop_midpoint = cls._network.create(geometry=1, neuron=midpoint, name="midpoint")
        pop_exponential = cls._network.create(
            geometry=1, neuron=exponential, name="exponential"
        )

        cls.m_explicit = cls._network.monitor(pop_explicit, ["v", "u"])
        cls.m_implicit = cls._network.monitor(pop_implicit, ["v", "u"])
        cls.m_midpoint = cls._network.monitor(pop_midpoint, ["v", "u"])
        cls.m_exponential = cls._network.monitor(pop_exponential, ["v", "u"])

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        self._network.reset()

    def test_precision(self):
        """
        Makes sure the precision of the numerical methods is good enough (errors come from the methods themselves, not ANNarchy).
        """

        self._network.simulate(20.0)
        self._network.get_population("explicit").I = 1.0
        self._network.get_population("implicit").I = 1.0
        self._network.get_population("midpoint").I = 1.0
        self._network.get_population("exponential").I = 1.0
        self._network.simulate(80)

        data_explicit = self.m_explicit.get("v")[:, 0]
        data_implicit = self.m_implicit.get("v")[:, 0]
        data_midpoint = self.m_midpoint.get("v")[:, 0]
        data_exponential = self.m_exponential.get("v")[:, 0]

        data_mean = (
            data_explicit + data_implicit + data_midpoint + data_exponential
        ) / 4.0

        error_explicit = numpy.max(numpy.abs(data_explicit - data_mean))
        error_implicit = numpy.max(numpy.abs(data_implicit - data_mean))
        error_midpoint = numpy.max(numpy.abs(data_midpoint - data_mean))
        error_exponential = numpy.max(numpy.abs(data_exponential - data_mean))

        self.assertTrue(error_explicit < 0.05)
        self.assertTrue(error_implicit < 0.05)
        self.assertTrue(error_midpoint < 0.05)
        self.assertTrue(error_exponential < 0.05)
