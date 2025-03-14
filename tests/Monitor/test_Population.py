"""

    test_Population.py

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

from ANNarchy import Neuron, Network, Synapse

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

class test_PopulationMonitor(unittest.TestCase):
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
        cls._network = Network()
        
        pop1 = cls._network.create(geometry=3, neuron=neuron)
        pop2 = cls._network.create(geometry=5, neuron=neuron)
        pop3 = cls._network.create(geometry=3, neuron=neuron2)
        pop4 = cls._network.create(geometry=3, neuron=neuron3)

        cls._m = cls._network.monitor(pop1, 'r')
        cls._n = cls._network.monitor(pop1[:2], 'r')
        cls._o = cls._network.monitor(pop1, 'r', period=10.0)
        cls._p = cls._network.monitor(pop1, 'r', start=False)
        cls._q = cls._network.monitor(pop1[0] + pop1[2], 'r')
        cls._r = cls._network.monitor(pop2[:2] + pop2.neuron(4), 'r')
        cls._s = cls._network.monitor(pop3, ['v', 'spike'])
        cls._t = cls._network.monitor(pop4, ['v', 'spike'])
        
        proj = cls._network.connect(
            pre = pop1,
            post = pop2,
            target = "exc",
            synapse = Oja
        )
        proj.all_to_all(weights = 1.0)

        cls._network.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we call *reset()* to reset the network.
        """
        self._network.reset()

    def tearDown(self):
        """
        Since all tests are independent, after every test we use the *get()*
        method for every monotor to clear all recordings.
        """
        self._m.get()
        self._n.get()
        self._o.get()
        self._p.get()
        self._q.get()
        self._r.get()
        self._s.get()
        self._t.get()

    def test_r_sim_10(self):
        """
        Tests the recording of the variable *r* of a *Population* of 3 neurons
        for 10 time steps.
        """
        self._network.simulate(10)
        data_m = self._m.get()
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
        self._network.simulate(10)
        datan = self._n.get()
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
        self._network.simulate(100)
        datao = self._o.get()
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
        self._network.simulate(10)
        self._p.start()
        self._network.simulate(10)
        datap = self._p.get()
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
        self._m.pause()
        self._network.simulate(10)
        self._m.resume()
        self._network.simulate(10)
        datam = self._m.get()
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
        self._network.simulate(10)
        datam = self._m.get()
        numpy.testing.assert_allclose(datam['r'][5, :], [5.0, 5.0, 5.0])

    def test_r_from_rank(self):
        """
        Tests the access to the recording of the variable *r* belonging to a
        neuron, which is specified by rank.
        """
        self._network.simulate(10)
        datam = self._m.get()
        numpy.testing.assert_allclose(datam['r'][:, 1], [0.0, 1.0, 2.0, 3.0,
                                                         4.0, 5.0, 6.0, 7.0,
                                                         8.0, 9.0])

    def test_popview(self):
        """
        One can also record variables of a *PopulationView* object. This is
        tested here.
        """
        self._network.simulate(10)
        dataq = self._q.get()
        numpy.testing.assert_allclose(dataq['r'], [[0.0, 0.0], [1.0, 1.0],
                                                   [2.0, 2.0], [3.0, 3.0],
                                                   [4.0, 4.0], [5.0, 5.0],
                                                   [6.0, 6.0], [7.0, 7.0],
                                                   [8.0, 8.0], [9.0, 9.0]])

    def test_popview2(self):
        """
        One can also record variables of a *PopulationView* object. This is
        tested here. The PopulationView comprise of pop2[:2] and pop2[4]
        """
        self._network.simulate(10)
        datar = self._r.get()
        numpy.testing.assert_allclose(datar['r'], [[0.0, 0.0, 0.0],
                                                   [1.0, 1.0, 1.0],
                                                   [2.0, 2.0, 2.0],
                                                   [3.0, 3.0, 3.0],
                                                   [4.0, 4.0, 4.0],
                                                   [5.0, 5.0, 5.0],
                                                   [6.0, 6.0, 6.0],
                                                   [7.0, 7.0, 7.0],
                                                   [8.0, 8.0, 8.0],
                                                   [9.0, 9.0, 9.0]])

    def test_spike(self):
        """
        Tests if the time steps of *spikes* of a *Population* of spiking
        neurons are correctly recorded.
        """
        self._network.simulate(10)
        datas = self._s.get('spike')
        self.assertEqual(datas[0], [4, 6, 8])

    def test_r_ref(self):
        """
        Tests if the variable *v* of a *Population* consisting of neurons with
        a defined *refractory* period is correctly recorded.
        """
        self._network.simulate(10)
        data_s = self._t.get()
        numpy.testing.assert_allclose(data_s['v'], [[1., 1., 1.], [2., 2., 2.],
                                                    [1., 1., 1.], [1., 1., 1.],
                                                    [1., 1., 1.], [1., 1., 1.],
                                                    [2., 2., 2.], [1., 1., 1.],
                                                    [1., 1., 1.], [1., 1., 1.]])

    def test_spike_ref(self):
        """
        Tests if the time steps of *spikes* of a *Population* of spiking
        neurons with a defined *refractory* period are correctly recorded.
        """
        self._network.simulate(10)
        data_t = self._t.get('spike')
        self.assertEqual(data_t[1], [2, 7])
