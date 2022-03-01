"""

    test_NumericalMethod.py

    This file is part of ANNarchy.

    Copyright (C) 2017-2019 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
            spike="v > 1"
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="tau * dw/dt = -w")

        pop = Population(10, neuron)
        proj = Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(Uniform(1.0, 2.0),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network(True)

    def setUp(self):
        self.test_net.compile(silent=True)

    def test_work(self):
        self.test_net.simulate(1.0)


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
            spike="v > 1"
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="tau * dw/dt = -w : implicit")

        pop = Population(10, neuron)
        proj = Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(Uniform(1.0, 2.0),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])

    def setUp(self):
        self.test_net.compile(silent=True)

    def test_work(self):
        self.test_net.simulate(1.0)

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
            spike="v > 1"
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="""
                tau * dw/dt = -w + u : implicit
                tau * du/dt = -u +1 : implicit
                """
        )

        pop = Population(10, neuron)
        proj = Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(Uniform(1.0, 2.0),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])

    def setUp(self):
        self.test_net.compile(silent=True)

    def test_work(self):
        self.test_net.simulate(1.0)



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
            spike="v > 1"
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="tau * dw/dt = -w : midpoint")

        pop = Population(10, neuron)
        proj = Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(Uniform(1.0, 2.0),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])

    def setUp(self):
        self.test_net.compile(silent=True)

    def test_work(self):
        self.test_net.simulate(1.0)

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
            spike="v > 1"
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="""
                tau * dw/dt = -w + u : midpoint
                tau * du/dt = -u +1 : midpoint
                """
        )

        pop = Population(10, neuron)
        proj = Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(Uniform(1.0, 2.0),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])

    def setUp(self):
        self.test_net.compile(silent=True)

    def test_work(self):
        self.test_net.simulate(1.0)


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
            spike="v > 1"
        )

        synapse = Synapse(
            parameters="tau = 10.0 : projection",
            equations="""
                tau * dw/dt = -w + u : exponential
                tau * du/dt = -u +1 : exponential
                """
        )

        pop = Population(10, neuron)
        proj = Projection(pop, pop, 'exc', synapse)
        proj.connect_all_to_all(Uniform(1.0, 2.0),
                                storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([pop, proj])


    def setUp(self):
        self.test_net.compile(silent=True)

    def test_work(self):
        self.test_net.simulate(1.0)

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
            """
        )
        pop_explicit = Population(1, explicit, name="explicit")
        m_explicit = Monitor(pop_explicit, ['v', 'u'])

        implicit = Neuron(
            parameters="""
            tau = 10.0
            I = 0.0
            """,
            equations="""
            tau * dv/dt = -v -u + I : implicit
            tau * du/dt = - u + I : implicit
            r = pos(v)
            """
        )
        pop_implicit = Population(1, implicit, name="implicit")
        m_implicit = Monitor(pop_implicit, ['v', 'u'])

        midpoint = Neuron(
            parameters="""
            tau = 10.0
            I = 0.0
            """,
            equations="""
            tau * dv/dt = -v -u + I : midpoint
            tau * du/dt = - u + I : midpoint
            r = pos(v)
            """
        )
        pop_midpoint = Population(1, midpoint, name="midpoint")
        m_midpoint = Monitor(pop_midpoint, ['v', 'u'])

        exponential = Neuron(
            parameters="""
            tau = 10.0
            I = 0.0
            """,
            equations="""
            tau * dv/dt = -v -u + I : exponential
            tau * du/dt = - u + I : exponential
            r = pos(v)
            """
        )
        pop_exponential = Population(1, exponential, name="exponential")
        m_exponential = Monitor(pop_exponential, ['v', 'u'])

        cls.test_net = Network()
        cls.test_net.add([pop_explicit, m_explicit, pop_implicit, m_implicit, pop_midpoint, m_midpoint, pop_exponential, m_exponential])

        cls.m_explicit = cls.test_net.get(m_explicit)
        cls.m_implicit = cls.test_net.get(m_implicit)
        cls.m_midpoint = cls.test_net.get(m_midpoint)
        cls.m_exponential = cls.test_net.get(m_exponential)


    def setUp(self):
        self.test_net.compile(silent=True)

    def test_precision(self):
        """
        Makes sure the precision of the numerical methods is good enough (errors come from the methods themselves, not ANNarchy).
        """

        self.test_net.simulate(20.)
        self.test_net.get_population('explicit').I = 1.0
        self.test_net.get_population('implicit').I = 1.0
        self.test_net.get_population('midpoint').I = 1.0
        self.test_net.get_population('exponential').I = 1.0
        self.test_net.simulate(80)

        data_explicit = self.m_explicit.get('v')[:, 0]
        data_implicit = self.m_implicit.get('v')[:, 0]
        data_midpoint = self.m_midpoint.get('v')[:, 0]
        data_exponential = self.m_exponential.get('v')[:, 0]

        data_mean = (data_explicit+data_implicit+data_midpoint+data_exponential)/4.

        error_explicit = np.max(np.abs(data_explicit - data_mean))
        error_implicit = np.max(np.abs(data_implicit - data_mean))
        error_midpoint = np.max(np.abs(data_midpoint - data_mean))
        error_exponential = np.max(np.abs(data_exponential - data_mean))

        self.assertTrue(error_explicit < 0.05)
        self.assertTrue(error_implicit < 0.05)
        self.assertTrue(error_midpoint < 0.05)
        self.assertTrue(error_exponential < 0.05)

        

if __name__ == '__main__':
    unittest.main()
