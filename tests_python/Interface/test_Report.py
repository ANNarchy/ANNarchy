"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import os
import io
import unittest
from unittest.mock import patch

from ANNarchy import Network, Hebb, IF_curr_exp, Neuron, STDP, Synapse, report, Uniform

class test_Report_Rate(unittest.TestCase):
    """
    Test the report function for a rate based network.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        DefaultNeuron = Neuron(
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
                f(x) = 2/(x*x)
                f2(x) = if x>0: x else:x/2
            """
        )

        emptyNeuron1 = Neuron(equations="r=0")
        emptyNeuron2 = Neuron(parameters="r=0")

        Oja = Synapse(
            parameters = """
                eta = 10.0
                tau = 10.0 : postsynaptic
                mu = 0.2 : postsynaptic
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

        cls._network = Network()

        pop1 = cls._network.create(name='pop1', neuron=emptyNeuron1, geometry=1)
        pop2 = cls._network.create(name='pop2', neuron=DefaultNeuron, geometry=1)
        pop3 = cls._network.create(name='pop3', neuron=emptyNeuron2, geometry=(2,2))
        proj1 = cls._network.connect(pre=pop1, post=pop2, target='exc', synapse=Oja)
        proj1.one_to_one(weights=Uniform(-0.5, 0.5))
        proj2 = cls._network.connect(pre=pop1, post=pop3, target='exc',
                               synapse=Hebb)
        proj2.all_to_all(1.0)
        proj3 = cls._network.connect(pre=pop1, post=pop3, target='exc')
        proj3.all_to_all(1.0)

        cls._network.monitor(pop2,'r')

    @classmethod
    def tearDownClass(cls):
        """ Clear ANNarchy Network after tests are run."""
        del cls._network

    def test_tex_report(self):
        """
        Run the report function to generate a .tex file. Delete it afterwards.
        """
        fn = "./report.tex"
        with patch('sys.stdout', new=io.StringIO()):
            report(network=self._network, filename=fn)
        self.assertTrue(os.path.isfile(fn))
        os.remove(fn)

    def test_md_report(self):
        """
        Run the report function to generate a .md file. Delete it afterwards.
        """
        fn = "./report.md"
        with patch('sys.stdout', new=io.StringIO()):
            report(network=self._network, filename=fn)
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
        
        Izhikevich = Neuron(
            parameters = """
                noise = 0.0
                a = 0.02 : population
                b = 0.2 : population
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
            functions="""
                f(x) = 5.0 * x + 140
                f2(x) = 0.04 * x^2
                """
        )

        STP = Synapse(
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


        cls._network = Network()

        pop1 = cls._network.create(name='pop1', neuron=IF_curr_exp, geometry=1)
        pop2 = cls._network.create(name='pop2', neuron=Izhikevich, geometry=1)
        pop3 = cls._network.create(name='pop3', neuron=Izhikevich, geometry=(2,2))
        proj1 = cls._network.connect(pre=pop1, post=pop2, target='exc', synapse=STP)
        proj1.one_to_one(weights=Uniform(-0.5, 0.5))
        proj2 = cls._network.connect(pre=pop1, post=pop3, target='exc',
                               synapse=STDP)
        proj2.all_to_all(1.0)
        proj3 = cls._network.connect(pre=pop1, post=pop3, target='exc')
        proj3.all_to_all(1.0)

        cls._network.monitor(pop2,'spike')

    @classmethod
    def tearDownClass(cls):
        """ Clear ANNarchy Network after tests are run."""
        del cls._network

    def test_tex_report(self):
        """
        Run the report function to generate a .tex file. Delete it afterwards.
        """
        fn = "./report.tex"
        report(network=self._network, filename=fn)
        self.assertTrue(os.path.isfile(fn))
        os.remove(fn)

    def test_md_report(self):
        """
        Run the report function to generate a .md file. Delete it afterwards.
        """
        fn = "./report.md"
        report(network=self._network, filename=fn)
        self.assertTrue(os.path.isfile(fn))
        os.remove(fn)

if __name__ == '__main__':
    unittest.main(verbosity=2)
