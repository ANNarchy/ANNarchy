import unittest
import numpy
from ANNarchy import *

class test_ProjectionMonitor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        simple_pre_neuron = Neuron(
            equations="r = t"
        )

        simple_post_neuron = Neuron(
            equations="r = 10 - t"
        )

        simple_synapse = Synapse(
            equations="y = pre.r * post.r * w"
        )

        p0 = Population(2, simple_pre_neuron)
        p1 = Population(1, simple_post_neuron)

        proj = Projection(p0, p1, "exc", simple_synapse)
        proj.connect_all_to_all(0.5)

        m = Monitor(proj, ['y'])
        
        cls.test_m = m
        cls.test_net = Network()
        cls.test_net.add([p0, p1, proj, m])
        cls.test_net.compile(silent=True)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls.test_net
        clear()

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
        self.test_net.get(self.test_m).get()
    
    def test_y_sim_10(self):
        """
        We compute a variable *y*, which changes over time dependent on the
        pre- and post-synaptic state variables. This test shows if the recorded
        values of *y* are as expected.
        """
        self.test_net.simulate(10)

        data_m = self.test_net.get(self.test_m).get()['y']
        target_m = np.array([[[0.0, 0.0]],
                             [[4.5, 4.5]],
                             [[8.0, 8.0]],
                             [[10.5, 10.5]],
                             [[12.0, 12.0]],
                             [[12.5, 12.5]],
                             [[12.0, 12.0]],
                             [[10.5, 10.5]],
                             [[8.0, 8.0]],
                             [[4.5, 4.5]]], dtype=object)
        
        equal = True
        for t_idx in range(len(data_m)):
            for neur_idx in range(len(data_m[t_idx])):
                tmp_x = np.array(target_m[t_idx][neur_idx], dtype=float)
                tmp_y = np.array(data_m[t_idx][neur_idx], dtype=float)
                if not numpy.allclose(tmp_x, tmp_y):
                    equal = False

        numpy.testing.assert_equal(equal, True)