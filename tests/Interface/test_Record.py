"""

    test_Record.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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

from ANNarchy import Monitor, Neuron, Network, Projection, Population, Synapse

neuron = Neuron(
    equations="r = t"
)

neuron2 = Neuron(
    equations="""
        v = v + 1.0
    """,
    spike = "v == 5.0",
    reset = "v = 3.0"
)

neuron3 = Neuron(
    equations="""
        v = v + 1.0
    """,
    spike = "v == 3.0",
    reset = "v = 1.0 ",
    refractory = 3.0
)

Oja = Synapse(
    parameters="""
        tau = 5000.0
        alpha = 8.0
    """,
    equations = """
        w = t
    """
)

pop1 = Population(3, neuron)
pop2 = Population(5, neuron)
pop3 = Population(3, neuron2)
pop4 = Population(3, neuron3)

proj = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)

proj.connect_all_to_all(weights = 1.0)

m = Monitor(pop1, 'r')
n = Monitor(pop1[:2], 'r')
o = Monitor(pop1, 'r', period=10.0)
p = Monitor(pop1, 'r', start=False)
q = Monitor(pop1[0] + pop1[2], 'r')
r = Monitor(pop3, ['v', 'spike'])
s = Monitor(pop4, ['v', 'spike'])

class test_Record(unittest.TestCase):
    """
    This class tests the selective recording of the evolution of neural or
    synaptic variables during a simulation.  To do so, the *Monitor* object is
    used.  *Population*, *PopulationView* and *Dendrite* objects can be
    recorded.

    A number of *Monitors* is defined to test specific recording preferences.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        cls.test_net = Network()
        cls.test_net.add([pop1, pop2, pop3, pop4, proj, m, n, o, p, q, r, s])
        cls.test_net.compile(silent=True)

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self.test_net.reset()

    def tearDown(self):
        """
        Since all tests are independent, after every test we use the *get()*
        method for every monotor to clear all recordings.
        """
        self.test_net.get(m).get()
        self.test_net.get(n).get()
        self.test_net.get(o).get()
        self.test_net.get(p).get()
        self.test_net.get(q).get()
        self.test_net.get(r).get()
        self.test_net.get(s).get()

    def test_r_sim_10(self):
        """
        Tests the recording of the variable *r* of a *Population* of 3 neurons
        for 10 time steps.
        """
        self.test_net.simulate(10)
        data_m = self.test_net.get(m).get()
        numpy.testing.assert_allclose(data_m['r'], [[0.0, 0.0, 0.0],
                                                    [1.0, 1.0, 1.0],
                                                    [2.0, 2.0, 2.0],
                                                    [3.0, 3.0, 3.0],
                                                    [4.0, 4.0, 4.0],
                                                    [5.0, 5.0, 5.0],
                                                    [6.0, 6.0, 6.0],
                                                    [7.0, 7.0, 7.0],
                                                    [8.0, 8.0, 8.0],
                                                    [9.0, 9.0, 9.0]])

    def test_r_first_neurons(self):
        """
        Tests the recording of the variable *r* of the first 2 neurons of a
        *Population* for 10 time steps.
        """
        self.test_net.simulate(10)
        datan = self.test_net.get(n).get()
        numpy.testing.assert_allclose(datan['r'], [[0.0, 0.0], [1.0, 1.0],
                                                   [2.0, 2.0], [3.0, 3.0],
                                                   [4.0, 4.0], [5.0, 5.0],
                                                   [6.0, 6.0], [7.0, 7.0],
                                                   [8.0, 8.0], [9.0, 9.0]])

    def test_r_sim_100_p_10(self):
        """
        Tests the recording of the variable *r* of a *Population* of 3 neurons
        for 100 time steps and a set *period* of 10.0.
        """
        self.test_net.simulate(100)
        datao = self.test_net.get(o).get()
        numpy.testing.assert_allclose(datao['r'], [[0.0, 0.0, 0.0],
                                                   [10.0, 10.0, 10.0],
                                                   [20.0, 20.0, 20.0],
                                                   [30.0, 30.0, 30.0],
                                                   [40.0, 40.0, 40.0],
                                                   [50.0, 50.0, 50.0],
                                                   [60.0, 60.0, 60.0],
                                                   [70.0, 70.0, 70.0],
                                                   [80.0, 80.0, 80.0],
                                                   [90.0, 90.0, 90.0]])

    def test_startrec(self):
        """
        Tests the *start()* method of a *Monitor*, which *start* parameter has
        been set to "false".  That *Monitor* won't record until *start()* is
        called.
        """
        self.test_net.simulate(10)
        self.test_net.get(p).start()
        self.test_net.simulate(10)
        datap = self.test_net.get(p).get()
        numpy.testing.assert_allclose(datap['r'], [[10.0, 10.0, 10.0],
                                                   [11.0, 11.0, 11.0],
                                                   [12.0, 12.0, 12.0],
                                                   [13.0, 13.0, 13.0],
                                                   [14.0, 14.0, 14.0],
                                                   [15.0, 15.0, 15.0],
                                                   [16.0, 16.0, 16.0],
                                                   [17.0, 17.0, 17.0],
                                                   [18.0, 18.0, 18.0],
                                                   [19.0, 19.0, 19.0]])

    def test_a_pauserec(self):
        """
        Tests the *pause()* and *resume()* methods of a *Monitor*, which are
        designed so one can stop recording and resume whenever it is necessary.
        """
        self.test_net.get(m).pause()
        self.test_net.simulate(10)
        self.test_net.get(m).resume()
        self.test_net.simulate(10)
        datam = self.test_net.get(m).get()
        numpy.testing.assert_allclose(datam['r'], [[10.0, 10.0, 10.0],
                                                   [11.0, 11.0, 11.0],
                                                   [12.0, 12.0, 12.0],
                                                   [13.0, 13.0, 13.0],
                                                   [14.0, 14.0, 14.0],
                                                   [15.0, 15.0, 15.0],
                                                   [16.0, 16.0, 16.0],
                                                   [17.0, 17.0, 17.0],
                                                   [18.0, 18.0, 18.0],
                                                   [19.0, 19.0, 19.0]])

    def test_r_after_5(self):
        """
        Tests the access to a recording of the variable *r* made at a specific
        time step.
        """
        self.test_net.simulate(10)
        datam = self.test_net.get(m).get()
        numpy.testing.assert_allclose(datam['r'][5, :], [5.0, 5.0, 5.0])

    def test_r_from_rank(self):
        """
        Tests the access to the recording of the variable *r* belonging to a
        neuron, which is specified by rank.
        """
        self.test_net.simulate(10)
        datam = self.test_net.get(m).get()
        numpy.testing.assert_allclose(datam['r'][:, 1], [0.0, 1.0, 2.0, 3.0,
                                                         4.0, 5.0, 6.0, 7.0,
                                                         8.0, 9.0])

    def test_popview(self):
        """
        One can also record variables of a *PopulationView* object. This is
        tested here.
        """
        self.test_net.simulate(10)
        dataq = self.test_net.get(q).get()
        numpy.testing.assert_allclose(dataq['r'], [[0.0, 0.0], [1.0, 1.0],
                                                   [2.0, 2.0], [3.0, 3.0],
                                                   [4.0, 4.0], [5.0, 5.0],
                                                   [6.0, 6.0], [7.0, 7.0],
                                                   [8.0, 8.0], [9.0, 9.0]])

    def test_spike(self):
        """
        Tests if the time steps of *spikes* of a *Population* of spiking
        neurons are correctly recorded.
        """
        self.test_net.simulate(10)
        datar = self.test_net.get(r).get('spike')
        self.assertEqual(datar[0], [4, 6, 8])

    def test_r_ref(self):
        """
        Tests if the variable *v* of a *Population* consisting of neurons with
        a defined *refractory* period is correctly recorded.
        """
        self.test_net.simulate(10)
        datas = self.test_net.get(s).get()
        numpy.testing.assert_allclose(datas['v'], [[1., 1., 1.], [2., 2., 2.],
                                                   [1., 1., 1.], [1., 1., 1.],
                                                   [1., 1., 1.], [1., 1., 1.],
                                                   [2., 2., 2.], [1., 1., 1.],
                                                   [1., 1., 1.], [1., 1., 1.]])

    def test_spike_ref(self):
        """
        Tests if the time steps of *spikes* of a *Population* of spiking
        neurons with a defined *refractory* period are correctly recorded.
        """
        self.test_net.simulate(10)
        datas = self.test_net.get(s).get('spike')
        self.assertEqual(datas[1], [2, 7])
