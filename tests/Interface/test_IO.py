"""

    test_IO.py

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
import io
import os
from shutil import rmtree
import unittest
from unittest.mock import patch
import numpy
from ANNarchy import add_function, clear, Constant, Hebb, IF_curr_exp, \
    load_parameters, Monitor, Network, Neuron, STDP, Synapse, Population, \
    Projection, save_parameters, Uniform

class test_IO_Rate(unittest.TestCase):
    """
    Test basic input output operations for a rate based network.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        Constant('mu', 0.2)
        add_function("fa(x) = 2/(x*x)")
        DefaultNeuron = Neuron(
            parameters = """
                tau = 10.0 : population
                baseline = 1.0
                condition = True
            """,
            equations= """
                noise = Uniform(-0.1, 0.1)
                tau * dmp/dt + mp = fa(sum(exc)) +  neg(- fb(sum(inh)))
                                    + baseline + noise : implicit
                r = pos(mp)
            """,
            functions="""
                fb(x) = if x>0: x else:x/2
            """
        )

        emptyNeuron1 = Neuron(equations="r=0")
        emptyNeuron2 = Neuron(parameters="r=0")

        Oja = Synapse(
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

        pop1 = Population(name='pop1', neuron=emptyNeuron1, geometry=1)
        pop2 = Population(name='pop2', neuron=DefaultNeuron, geometry=1)
        pop3 = Population(name='pop3', neuron=emptyNeuron2, geometry=(2,2))
        proj1 = Projection(pre=pop1, post=pop2, target='exc')
        proj1.connect_one_to_one(weights=Uniform(-0.5, 0.5))
        proj2 = Projection(pre=pop1, post=pop3, target='exc',
                               synapse=Hebb)
        proj2.connect_all_to_all(1.0)
        proj3 = Projection(pre=pop2, post=pop3, target='exc', synapse=Oja)
        proj3.connect_all_to_all(1.0)

        m = Monitor(pop2,'r')

        cls.network = Network()
        cls.network.add([pop1, pop2, pop3, proj1, proj2, proj3, m])
        cls.network.compile(silent=True)
        cls.pop2 = cls.network.get(pop2)
        cls.proj1 = cls.network.get(proj1)
        cls.proj2 = cls.network.get(proj2)
        cls.proj3 = cls.network.get(proj3)

        cls.newp = [0.5, 5, 0.02, 3, 20]
        cls.savefolder = '_networksave/'
        os.mkdir(cls.savefolder)
        cls.save_extensions = ['.data', '.npz', '.txt.gz']

    @classmethod
    def tearDownClass(cls):
        """ Delete the save folder after all tests were run. """
        rmtree(cls.savefolder)
        clear()

    def setUp(self):
        """ Clear the network before every test. """
        self.network.reset()

    def set_parameters(self, parameters):
        """ Set the parameters of the network. """
        self.pop2.set({'baseline': parameters[0], 'r': parameters[1]})
        self.proj2.set({'eta': parameters[2], 'w': parameters[3]})
        self.proj3.set({'tau': parameters[4]})

    def get_parameters(self):
        """ Check if the parameters of the network are as expected. """
        return [self.pop2.get('baseline')[0], self.pop2.get('r')[0],
                self.proj2.get('eta'), self.proj2.get('w')[0][0],
                self.proj3.get('tau')[0]]

    def test_save_and_load(self):
        """
        Save the network in every loadable format and check if it can be loaded
        """
        for ext in self.save_extensions:
            with self.subTest(extension=ext):
                self.set_parameters(self.newp)
                with patch('sys.stdout', new=io.StringIO()): # suppress print
                    self.network.save(self.savefolder + "ratenet" + ext)
                self.network.reset()
                self.network.load(self.savefolder + "ratenet" + ext)
                numpy.testing.assert_allclose(self.get_parameters(), self.newp)

    def test_save_mat(self):
        """
        Save the network in the non-loadable .mat format
        """
        with patch('sys.stdout', new=io.StringIO()): # suppress print
            self.network.save(self.savefolder + "ratenet.mat")

    def test_parameters_save_and_load(self):
        """
        Save and load only the parameters of the network
        """
        ID = self.network.id
        self.set_parameters(self.newp)
        with patch('sys.stdout', new=io.StringIO()): # suppress print
            save_parameters(self.savefolder + "ratenet.json", net_id=ID)
        self.network.reset()
        load_parameters(self.savefolder + "ratenet.json", net_id=ID)
        numpy.testing.assert_allclose(self.get_parameters(), self.newp)

    def test_projection_save_and_load(self):
        """
        Save one projection from the network in every loadable format and check
        if it can be loaded
        """
        for ext in self.save_extensions:
            with self.subTest(extension=ext):
                with patch('sys.stdout', new=io.StringIO()): # suppress print
                    self.proj2.save(self.savefolder + "pr2rate" + ext)
                self.proj2.load(self.savefolder + "pr2rate" + ext)


class test_IO_Spiking(unittest.TestCase):
    """
    Test basic input output operations for a rate based network.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        Constant('b', 0.2)
        add_function("f(x) = 5.0 * x + 140")
        Izhikevich = Neuron(
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
            post_spike="y += post.I",
        )

        pop1 = Population(name='pop1', neuron=IF_curr_exp, geometry=1)
        pop2 = Population(name='pop2', neuron=Izhikevich, geometry=1)
        pop3 = Population(name='pop3', neuron=Izhikevich, geometry=(2,2))
        proj1 = Projection(pre=pop1, post=pop2, target='exc',
                               synapse=STDP)
        proj1.connect_one_to_one(weights=Uniform(-0.5, 0.5))
        proj2 = Projection(pre=pop2, post=pop3, target='exc', synapse=STP)
        proj2.connect_all_to_all(1.0)

        m = Monitor(pop2,'r')

        cls.network = Network()
        cls.network.add([pop1, pop2, pop3, proj1, proj2, m])
        cls.network.compile(silent=True)
        cls.pop2 = cls.network.get(pop2)
        cls.proj1 = cls.network.get(proj1)
        cls.proj2 = cls.network.get(proj2)

        cls.newp = [0.5, -60, 0.01, 5, 10]
        cls.savefolder = '_networksave/'
        os.mkdir(cls.savefolder)
        cls.save_extensions = ['.data', '.npz', '.txt.gz']

    def setUp(self):
        """ Clear the network before every test. """
        self.network.reset()

    @classmethod
    def tearDownClass(cls):
        """ Delete the save folder after all tests were run. """
        rmtree(cls.savefolder)
        clear()

    def set_parameters(self, parameters):
        """ Set the parameters of the network. """
        self.pop2.set({'d': parameters[0], 'v': parameters[1]})
        self.proj1.set({'tau': parameters[2]})
        self.proj2.set({'U': parameters[3], 'x': parameters[4]})

    def get_parameters(self):
        """ Return the parameters of the network. """
        return [self.pop2.get('d')[0], self.pop2.get('v')[0],
                self.proj1.get('tau'), self.proj2.get('U')[0][0],
                self.proj2.get('x')[0][0]]

    def test_save_and_load(self):
        """
        Save the network in every loadable format and check if it can be loaded
        """
        for ext in self.save_extensions:
            with self.subTest(extension=ext):
                self.set_parameters(self.newp)
                with patch('sys.stdout', new=io.StringIO()): # suppress print
                    self.network.save(self.savefolder + "spikenet" + ext)
                self.network.reset()
                self.network.load(self.savefolder + "spikenet" + ext)
                numpy.testing.assert_allclose(self.get_parameters(), self.newp)

    def test_save_mat(self):
        """
        Save the network in the non-loadable .mat format
        """
        with patch('sys.stdout', new=io.StringIO()): # suppress print
            self.network.save(self.savefolder + "spikenet.mat")

    def test_parameters_save_and_load(self):
        """
        Save the network in every loadable format and check if it can be loaded
        """
        ID = self.network.id
        self.set_parameters(self.newp)
        with patch('sys.stdout', new=io.StringIO()): # suppress print
            save_parameters(self.savefolder + "ratenet.json", net_id=ID)
        self.network.reset()
        load_parameters(self.savefolder + "ratenet.json", net_id=ID)
        numpy.testing.assert_allclose(self.get_parameters(), self.newp)


    def test_projection_save_and_load(self):
        """
        Save one projection from the network in every loadable format and check
        if it can be loaded
        """
        for ext in self.save_extensions:
            with self.subTest(extension=ext):
                with patch('sys.stdout', new=io.StringIO()): # suppress print
                    self.proj2.save(self.savefolder + "pr2spike" + ext)
                self.proj2.load(self.savefolder + "pr2spike" + ext)


if __name__ == "__main__":
    unittest.main()
