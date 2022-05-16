"""

    test_Report.py

    This file is part of ANNarchy.

    Copyright (C) 2022 Alex Schwarz <alex.schwarz@informatik.tu-chemnitz.de>

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
import os
import io
import unittest
from unittest.mock import patch
import ANNarchy as ann

class test_Report_Rate(unittest.TestCase):
    """
    Test the report function for a rate based network.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        ann.Constant('mu', 0.2)
        ann.add_function("f(x) = 2/(x*x)")
        DefaultNeuron = ann.Neuron(
            parameters = """
                tau = 10.0 : population
                baseline = 1.0
                condition = True
            """,
            equations= """
                noise = Uniform(-0.1, 0.1)
                tau * dmp/dt + mp = f(sum(exc)) +  neg(- f2(sum(inh)))
                                    + baseline + noise : implicit
                r = pos(mp)
            """,
            functions="""
                f2(x) = if x>0: x else:x/2
            """
        )

        emptyNeuron1 = ann.Neuron(equations="r=0")
        emptyNeuron2 = ann.Neuron(parameters="r=0")

        Oja = ann.Synapse(
            parameters = """
                eta = 10.0
                tau = 10.0 : postsynaptic
            """,
            equations = """
                tau * dalpha/dt + alpha = pos(post.r - 1.0) : postsynaptic
                eta * dw/dt = pre.r * post.r - alpha * post.r^2 * w : min=0.0
                mu * dx/dt = pre.sum(exc) * post.sum(exc)
            """,
            psp = """
                w * pre.r
            """
        )

        pop1 = ann.Population(name='pop1', neuron=emptyNeuron1, geometry=1)
        pop2 = ann.Population(name='pop2', neuron=DefaultNeuron, geometry=1)
        pop3 = ann.Population(name='pop3', neuron=emptyNeuron2, geometry=(2,2))
        proj1 = ann.Projection(pre=pop1, post=pop2, target='exc', synapse=Oja)
        proj1.connect_one_to_one(weights=ann.Uniform(-0.5, 0.5))
        proj2 = ann.Projection(pre=pop1, post=pop3, target='exc',
                               synapse=ann.Hebb)
        proj2.connect_all_to_all(1.0)
        proj3 = ann.Projection(pre=pop1, post=pop3, target='exc')
        proj3.connect_all_to_all(1.0)

        ann.Monitor(pop2,'r')

    def test_tex_report(self):
        """
        Run the report function to generate a .tex file. Delete it afterwards.
        """
        fn = "./report.tex"
        with patch('sys.stdout', new=io.StringIO()):
            ann.report(filename=fn)
        self.assertTrue(os.path.isfile(fn))
        os.remove(fn)

    def test_md_report(self):
        """
        Run the report function to generate a .md file. Delete it afterwards.
        """
        fn = "./report.md"
        with patch('sys.stdout', new=io.StringIO()):
            ann.report(filename=fn)
        self.assertTrue(os.path.isfile(fn))
        os.remove(fn)

class test_Report_Spiking(unittest.TestCase):
    """
    Test the report function for a spiking based network.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        ann.Constant('b', 0.2)
        ann.add_function("f(x) = 5.0 * x + 140")
        Izhikevich = ann.Neuron(
            parameters = """
                noise = 0.0
                a = 0.02 : population
                c = -65.0
                d = 8.0
                v_thresh = 30.0
                i_offset = 0.0
            """,
            equations = """
                I = g_exc - g_inh + noise * Normal(0.0, 1.0) + i_offset
                dv/dt = f2(v) + f(v) - u + I : init = -65.0
                du/dt = a * (b*v - u) : init= -13.0
            """,
            spike = "v > v_thresh",
            reset = "v = c; u += d",
            refractory = 0.0,
            functions="f2(x) = 0.04 * x^2"
        )

        STP = ann.Synapse(
            parameters = """
                tau_rec = 100.0 : projection
                tau_facil = 0.01 : projection
                U = 0.5
            """,
            equations = """
                dx/dt = (1 - x)/tau_rec : init = 1.0, event-driven
                du/dt = (U - u)/tau_facil : init = 0.5, event-driven
            """,
            pre_spike="""
                g_target += w * u * x
                x *= (1 - u)
                u += U * (1 - u)
                y = pre.u + post.I
            """,
            post_spike="y += pre.u + post.I",
            psp="g_exc * post.u"
        )

        pop1 = ann.Population(name='pop1', neuron=ann.IF_curr_exp, geometry=1)
        pop2 = ann.Population(name='pop2', neuron=Izhikevich, geometry=1)
        pop3 = ann.Population(name='pop3', neuron=Izhikevich, geometry=(2,2))
        proj1 = ann.Projection(pre=pop1, post=pop2, target='exc', synapse=STP)
        proj1.connect_one_to_one(weights=ann.Uniform(-0.5, 0.5))
        proj2 = ann.Projection(pre=pop1, post=pop3, target='exc',
                               synapse=ann.STDP)
        proj2.connect_all_to_all(1.0)

        ann.Monitor(pop2,'r')

    def test_tex_report(self):
        """
        Run the report function to generate a .tex file. Delete it afterwards.
        """
        fn = "./report.tex"
        with patch('sys.stdout', new=io.StringIO()):
            ann.report(filename=fn)
        self.assertTrue(os.path.isfile(fn))
        os.remove(fn)

    def test_md_report(self):
        """
        Run the report function to generate a .md file. Delete it afterwards.
        """
        fn = "./report.md"
        with patch('sys.stdout', new=io.StringIO()):
            ann.report(filename=fn)
        self.assertTrue(os.path.isfile(fn))
        os.remove(fn)
