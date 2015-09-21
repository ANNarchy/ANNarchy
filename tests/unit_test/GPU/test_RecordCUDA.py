"""

    test_RecordCUDA.py

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

from ANNarchy import *
setup(paradigm="cuda")

neuron = Neuron(
    equations="r = t"
)

neuron2 = Neuron(
    equations="""
        r = r + 1.0
    """,
    spike = "r == 5.0",
    reset = "r = 3.0"
)

neuron3 = Neuron(
    equations="""
        r = r + 1.0
    """,
    spike = "r == 3.0",
    reset = "r = 1.0 ",
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

compile(clean=True)

m = Monitor(pop1, 'r')
n = Monitor(pop1[:1], 'r')
o = Monitor(pop1, 'r', period=10.0)
p = Monitor(pop1, 'r', start=False) 
q = Monitor(pop1[0] + pop1[2], 'r')
r = Monitor(proj.dendrite(2), 'w')
s = Monitor(pop3, ['r', 'spike'])
t = Monitor(pop4, ['r', 'spike'])


class test_Record(unittest.TestCase):
    
    def setUp(self):
        """
        setUp method to reset network state and time to test different recordings with the same time period
        """
        reset()

    def tearDown(self):
        """
        clears recordings so that next test can be run properly
        """
        m.get()
        n.get()
        o.get()
        p.get()
        q.get()
        r.get()
        s.get()
        t.get()

    def test_r_sim_10(self):
        """
        tests if r is recorded correctly with simulate(10)
        """
        simulate(10)
        datam = m.get()
        self.assertTrue(numpy.allclose(datam['r'], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])) 

    def test_r_first_neurons(self):
        """
        tests if r of the first 2 neurons is recorded correctly
        """
        simulate(10)
        datan = n.get()
        self.assertTrue(numpy.allclose(datan['r'], [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [9.0, 9.0]])) 



    def test_r_sim_100_p_10(self):
        """
        tests if r is recorded correctly with simulate(100) and period = 10.0
        """
        simulate(100)
        datao = o.get()
        self.assertTrue(numpy.allclose(datao['r'], [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0], [40.0, 40.0, 40.0], [50.0, 50.0, 50.0], [60.0, 60.0, 60.0], [70.0, 70.0, 70.0], [80.0, 80.0, 80.0], [90.0, 90.0, 90.0]])) 

    def test_startrec(self):
        """
        tests the manual start function of a monitor
        """
        simulate(10)
        p.start()
        simulate(10)
        datap = p.get()
        self.assertTrue(numpy.allclose(datap['r'], [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0], [12.0, 12.0, 12.0], [13.0, 13.0, 13.0], [14.0, 14.0, 14.0], [15.0, 15.0, 15.0], [16.0, 16.0, 16.0], [17.0, 17.0, 17.0], [18.0, 18.0, 18.0], [19.0, 19.0, 19.0]])) 

    def test_a_pauserec(self):
        """
        tests the manual pause function of a monitor
        """

        m.pause()
        simulate(10)
        m.resume()
        simulate(10)

        datam = m.get()
        #print m.times()
        self.assertTrue(numpy.allclose(datam['r'], [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0], [12.0, 12.0, 12.0], [13.0, 13.0, 13.0], [14.0, 14.0, 14.0], [15.0, 15.0, 15.0], [16.0, 16.0, 16.0], [17.0, 17.0, 17.0], [18.0, 18.0, 18.0], [19.0, 19.0, 19.0]])) 

    def test_r_after_5(self):
        """
        tests access to recordings at specific moments
        """
        simulate(10)
        datam = m.get()
        self.assertTrue(numpy.allclose(datam['r'][5, :], [5.0, 5.0, 5.0])) 


    def test_r_from_rank(self):
        """
        tests access to recordings of specific neurons (specified by rank)
        """
        simulate(10)
        datam = m.get()
        self.assertTrue(numpy.allclose(datam['r'][:, 1], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])) 


    def test_popview(self):
        """
        tests if r of the chosen population view is correctly recorded
        """
        simulate(10)
        dataq = q.get()
        self.assertTrue(numpy.allclose(dataq['r'], [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0], [9.0, 9.0]])) 

    def test_dendrite(self):
        """
        tests if dendrites can be recorded
        """
        simulate(10)
        datar = r.get()
        self.assertTrue(numpy.allclose(datar['w'], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]))



    def test_spike(self):
        """
        tests if spikes are correctly recorded
        """
        simulate(10)
        datas = s.get('spike')
        self.assertEqual(datas[0], [4, 6, 8]) 


    def test_r_ref(self):
        """
        tests if r is correctly recorded (neuron with refractory)
        """
        
        simulate(10)
        datat = t.get()
        self.assertTrue(numpy.allclose(datat['r'], [[ 1.,  1.,  1.], [ 2.,  2.,  2.], [ 1.,  1.,  1.], [ 1.,  1.,  1.], [ 1.,  1.,  1.], [ 1.,  1.,  1.], [ 2.,  2.,  2.], [ 1.,  1.,  1.], [ 1.,  1.,  1.], [ 1.,  1.,  1.]])) 

    def test_spike_ref(self):
        """
        tests if spikes are correctly recorded (neuron with refractory)
        """
        simulate(10)
        datat = t.get('spike')
        self.assertEqual(datat[1], [2, 7]) 

