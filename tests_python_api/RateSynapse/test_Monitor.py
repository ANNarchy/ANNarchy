"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import numpy
import unittest

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron, Synapse

class test_MonitorRatePSP(unittest.TestCase):
    """
    This test covers the recording of the post-synaptic potential in rate-coded
    networks.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        in_neuron = Neuron(
            equations="""
                r = t
            """
        )        
        out_neuron = Neuron(
            equations="""
                r = sum(exc)
            """
        )

        cls._network = Network()

        pre = cls._network.create(geometry=3, neuron=in_neuron)
        post = cls._network.create(geometry=1, neuron=out_neuron)
        proj = cls._network.connect(pre, post, "exc")
        proj.all_to_all(
            weights=1.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        cls._mon_sum_exc = cls._network.monitor(post, "sum(exc)", start=False)

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """ Delete class instance. """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset(populations=True, projections=True, monitors=True, synapses=False)

    def test_record_sum_exc(self):
        """
        Tests if the post-synaptic potential, i.e., sum(target) in rate-coded models,
        can be recored.
        """
        # Simulate 10 ms without recording
        self._network.simulate(10.0)

        # Record 5 ms
        self._mon_sum_exc.start()
        self._network.simulate(5.0)
        self._mon_sum_exc.pause()

        # Record another 10 ms
        self._network.simulate(10.0)

        # Retrieve the data
        rec_sum_exc = self._mon_sum_exc.get('sum(exc)')

        # Compare to expected result: t = 9 to t = 13 for 3 neurons
        numpy.testing.assert_allclose(rec_sum_exc, [ [27.0], [30.0],[33.0], [36.0], [39.0] ])


class test_MonitorLocalVariable(unittest.TestCase):
    """
    This test covers the recording of local variable of a rate-coded
    synapse.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        simple_pre_neuron = Neuron(
            equations="r = t"
        )

        simple_post_neuron = Neuron(
            equations="r = 10 - t"
        )

        simple_synapse = Synapse(
            equations="y = pre.r * post.r * w"
        )

        cls._network = Network()

        p0 = cls._network.create(geometry=2, neuron=simple_pre_neuron)
        p1 = cls._network.create(geometry=1, neuron=simple_post_neuron)

        proj = cls._network.connect(p0, p1, "exc", simple_synapse)
        proj.all_to_all(0.5)

        cls._mon_m = cls._network.monitor(proj, ['y'])
        
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

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
        self._mon_m.get()
    
    def test_y_sim_10(self):
        """
        We compute a variable *y*, which changes over time dependent on the
        pre- and post-synaptic state variables. This test shows if the recorded
        values of *y* are as expected.
        """
        self._network.simulate(10)

        data_m = self._mon_m.get()['y']
        target_m = numpy.array(
            [[[0.0, 0.0]],
            [[4.5, 4.5]],
            [[8.0, 8.0]],
            [[10.5, 10.5]],
            [[12.0, 12.0]],
            [[12.5, 12.5]],
            [[12.0, 12.0]],
            [[10.5, 10.5]],
            [[8.0, 8.0]],
            [[4.5, 4.5]]], dtype=object
        )
        
        equal = True
        for t_idx in range(len(data_m)):
            for neur_idx in range(len(data_m[t_idx])):
                tmp_x = numpy.array(target_m[t_idx][neur_idx], dtype=float)
                tmp_y = numpy.array(data_m[t_idx][neur_idx], dtype=float)
                if not numpy.allclose(tmp_x, tmp_y):
                    equal = False

        numpy.testing.assert_equal(equal, True)