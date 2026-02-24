"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron, TimedArray


class test_TimedArray(unittest.TestCase):
    """
    Test the correct evaluation of builtin functions
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test. Adapted the example
        from documentation ( section 3.7.3 Class TimedArray )
        """
        inputs = numpy.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        SimpleNeuron = Neuron(
            equations="""
                r = sum(exc)
                r2 = sum(exc2)
                r3 = sum(exc3)
                r4 = sum(exc4)
            """
        )

        cls._network = Network()

        inp = cls._network.create(population=TimedArray(rates=inputs))
        inp2 = cls._network.create(population=TimedArray(rates=inputs, schedule=10.0))
        inp3 = cls._network.create(population=TimedArray(rates=inputs, period=10.0))
        inp4 = cls._network.create(
            population=TimedArray(rates=inputs, schedule=2.0, period=20.0)
        )

        cls.output = cls._network.create(10, neuron=SimpleNeuron)

        proj = cls._network.connect(inp, cls.output, "exc")
        proj.one_to_one(1.0)
        proj2 = cls._network.connect(inp2, cls.output, "exc2")
        proj2.one_to_one(1.0)
        proj3 = cls._network.connect(inp3, cls.output, "exc3")
        proj3.one_to_one(1.0)
        proj4 = cls._network.connect(inp4, cls.output, "exc4")
        proj4.one_to_one(1.0)

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self._network.reset()

    def test_compile(self):
        """
        Enforce compilation of the network.
        """
        pass

    def test_run_one_loop(self):
        """
        We provide 10 ms input data, after 11ms the last element will be set.
        """
        self._network.simulate(11)
        numpy.testing.assert_allclose(self.output.r, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_more_steps(self):
        """
        We provide 10 ms input data, after 11ms the last element will be remain in the buffer.
        """
        self._network.simulate(12)
        numpy.testing.assert_allclose(self.output.r, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_run_one_schedule(self):
        """
        We provide 10 ms input data the 1st entry, after 11ms the 2nd element will be set. So the next 10
        iterations should return the 2nd element of the input array.
        """
        self._network.simulate(11)  # 1st entry

        self._network.simulate(10)  # 2nd entry
        numpy.testing.assert_allclose(self.output.r2, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period(self):
        """
        Period is set to 10, this means after 11 ms we have one cycle complete. This means the 12 step
        should return the 1st element in the output population.
        """
        self._network.simulate(12)  # 1st entry

        numpy.testing.assert_allclose(self.output.r3, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period_one_schedule(self):
        """
        Schedule == 2 doubles all entries. Period of 20 is one complete iteration. This means
        after 25 steps we are again on the 2nd entry.
        """
        self._network.simulate(25)

        numpy.testing.assert_allclose(self.output.r4, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])


class test_TimedArrayLateInit(unittest.TestCase):
    """
    Test the correct implementation of late-initialized timed arrays.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test. Adapted the example
        from documentation ( section 3.7.3 Class TimedArray )
        """
        inputs = numpy.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        SimpleNeuron = Neuron(
            equations="""
                r = sum(exc)
                r2 = sum(exc2)
                r3 = sum(exc3)
                r4 = sum(exc4)
            """
        )

        cls._network = Network()

        # Create TimedArrays without a pre-defined *rates* array
        inp = cls._network.create(population=TimedArray(geometry=(10)))
        inp2 = cls._network.create(population=TimedArray(geometry=(10)))
        inp3 = cls._network.create(population=TimedArray(geometry=(10)))
        inp4 = cls._network.create(population=TimedArray(geometry=(10)))

        # Test later initialization
        inp.update(rates=inputs)
        inp2.update(rates=inputs, schedule=10.0)
        inp3.update(rates=inputs, period=10.0)
        inp4.update(rates=inputs, schedule=2.0, period=20.0)

        # The output signal will be tested
        cls.output = cls._network.create(10, neuron=SimpleNeuron)

        proj = cls._network.connect(inp, cls.output, "exc")
        proj.one_to_one(1.0)
        proj2 = cls._network.connect(inp2, cls.output, "exc2")
        proj2.one_to_one(1.0)
        proj3 = cls._network.connect(inp3, cls.output, "exc3")
        proj3.one_to_one(1.0)
        proj4 = cls._network.connect(inp4, cls.output, "exc4")
        proj4.one_to_one(1.0)

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self._network.reset()

    def test_compile(self):
        """
        Enforce compilation of the network.
        """
        pass

    def test_run_one_loop(self):
        """
        We provide 10 ms input data, after 11ms the last element will be set.
        """
        self._network.simulate(11)
        numpy.testing.assert_allclose(self.output.r, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_more_steps(self):
        """
        We provide 10 ms input data, after 11ms the last element will be remain in the buffer.
        """
        self._network.simulate(12)
        numpy.testing.assert_allclose(self.output.r, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_run_one_schedule(self):
        """
        We provide 10 ms input data the 1st entry, after 11ms the 2nd element will be set. So the next 10
        iterations should return the 2nd element of the input array.
        """
        self._network.simulate(11)  # 1st entry

        self._network.simulate(10)  # 2nd entry
        numpy.testing.assert_allclose(self.output.r2, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period(self):
        """
        Period is set to 10, this means after 11 ms we have one cycle complete. This means the 12 step
        should return the 1st element in the output population.
        """
        self._network.simulate(12)  # 1st entry

        numpy.testing.assert_allclose(self.output.r3, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period_one_schedule(self):
        """
        Schedule == 2 doubles all entries. Period of 20 is one complete iteration. This means
        after 25 steps we are again on the 2nd entry.
        """
        self._network.simulate(25)

        numpy.testing.assert_allclose(self.output.r4, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])


class test_TimedArrayUpdate(unittest.TestCase):
    """
    This class tests the change of the input array after some simulation steps.
    """

    @classmethod
    def setUpClass(cls):
        cls.initial_inputs = numpy.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        cls.overwrite_inputs = numpy.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                [0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
                [0, 0, 1, 0, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 1, 0, 0, 2, 0, 0, 0],
                [0, 0, 0, 0, 1, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 2, 0, 0, 1, 0, 0, 0],
                [0, 0, 2, 0, 0, 0, 0, 1, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0, 1, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        SimpleNeuron = Neuron(
            equations="""
                r = sum(exc_1) + sum(exc_2)
            """
        )

        cls._network = Network()

        cls.input = cls._network.create(population=TimedArray(rates=cls.initial_inputs))
        cls.input2 = cls._network.create(
            population=TimedArray(rates=cls.initial_inputs, schedule=2.0)
        )

        cls.output = cls._network.create(geometry=10, neuron=SimpleNeuron)

        proj1 = cls._network.connect(cls.input, cls.output, "exc_1")
        proj1.one_to_one(1.0)

        proj2 = cls._network.connect(cls.input2, cls.output, "exc_2")
        proj2.one_to_one(1.0)

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self._network.reset()

    def test_compile(self):
        """
        Enforce compilation of the network.
        """
        pass

    def test_run_one_loop(self):
        """
        We provide 5 ms input data, then we set another input sequence and simulate 6 ms more. The
        expected result is the last value in the input buffer, as cycling is disabled
        and also the last input in the output population (from the 10th simulation step).
        """
        self._network.simulate(5)
        self.input.update(self.overwrite_inputs)
        self._network.simulate(6)
        numpy.testing.assert_allclose(
            self.output.sum("exc_1"), [2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        )

    def test_run_one_loop_and_schedule(self):
        """
        We provide 5 ms input data, then we set another input sequence and simulate until
        the end of the input sequence. With schedule=2 each input value is held for 2 ms.

        What should be the input values at different times:
        idx : timesteps
        0 : 1-2
        1 : 3-4
        2 : 5-6  --> after simulate 5 ms there should be self.initial_inputs[2]
        3 : 7-8
        4 : 9-10
        5 : 11-12 --> after simulate 11 ms there should be self.overwrite_inputs[5]
        6 : 13-14
        7 : 15-16
        8 : 17-18
        9 : 19-20 --> after simulate 20 ms there should be self.overwrite_inputs[9]

        In the output population we should always have the sum of the inputs from the
        previous timestep.
        """

        # first input block
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.initial_inputs[0])
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.initial_inputs[0])
        numpy.testing.assert_allclose(self.output.sum("exc_2"), self.initial_inputs[0])
        # second input block
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.initial_inputs[1])
        numpy.testing.assert_allclose(self.output.sum("exc_2"), self.initial_inputs[0])
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.initial_inputs[1])
        numpy.testing.assert_allclose(self.output.sum("exc_2"), self.initial_inputs[1])
        # third input block - during block change input
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.initial_inputs[2])
        numpy.testing.assert_allclose(self.output.sum("exc_2"), self.initial_inputs[1])
        self.input2.update(self.overwrite_inputs, schedule=2.0)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[2])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.initial_inputs[1]
        )  # in output pop still old input
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[2])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[2]
        )
        # fourth input block
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[3])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[2]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[3])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[3]
        )
        # fifth input block
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[4])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[3]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[4])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[4]
        )
        # sixth input block
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[5])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[4]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[5])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[5]
        )
        # 7th, 8th, 9th, 10th input blocks
        self._network.simulate(8)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[9])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[9]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[9])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[9]
        )

    def test_run_one_loop_reset(self):
        """
        We provide 5 ms input data, then we set another input sequence for the next 10 ms
        (by resetting the internal timers) and simulate 6 ms more. The expected result is
        the 6-th value in the new input buffer instead of the last one (as without reset)
        and the 5-th input in the output population.
        """
        self._network.simulate(5)
        self.input.update(self.overwrite_inputs, reset=True)
        self._network.simulate(6)
        numpy.testing.assert_allclose(self.input.r, self.overwrite_inputs[5])
        numpy.testing.assert_allclose(
            self.output.sum("exc_1"), self.overwrite_inputs[4]
        )

    def test_run_one_loop_and_schedule_reset(self):
        """
        We provide 5 ms input data, then we set another input sequence for the next 20 ms
        (by resetting the internal timers) and simulate 6 ms more. The expected result is:

        idx : timesteps
        0 : 1-2
        1 : 3-4
        2 : 5-  --> after simulate 5 ms we reset
        0 : 6-7
        1 : 8-9
        2 : 10-11 --> after simulate 11 ms there should be self.overwrite_inputs[2]
        """
        self._network.simulate(5)
        self.input2.update(self.overwrite_inputs, reset=True)  # schedule is still 2.
        self._network.simulate(6)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[2])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[2]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[3])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[2]
        )

    def test_run_one_loop_and_change_schedule_and_period_reset(self):
        """
        We provide 5 ms input data, then we set another input sequence for the next 40 ms
        (by resetting the internal timers and setting schedule to 4) and simulate 6 ms more.
        The expected result is:

        idx : timesteps
        0 : 1-2
        1 : 3-4
        2 : 5-  --> after simulate 5 ms we reset
        0 : 6-9
        1 : 10-13
        2 : 14-17 --> after simulate 17 ms there should be self.overwrite_inputs[2]

        Furthermore period is set to 40, so after 40 ms the input buffer should cycle:

        idx : timesteps
        9 : 37-40
        0 : 41-44
        """
        self._network.simulate(5)
        self.input2.update(self.overwrite_inputs, reset=True, schedule=4, period=40)
        self._network.simulate(12)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[2])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[2]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[3])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[2]
        )
        self._network.simulate(
            27
        )  # after simulate total 40 ms we should be at the end of the input buffer and in following steps it should cycle
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[9])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[9]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[0])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[9]
        )
        self._network.simulate(1)
        numpy.testing.assert_allclose(self.input2.r, self.overwrite_inputs[0])
        numpy.testing.assert_allclose(
            self.output.sum("exc_2"), self.overwrite_inputs[0]
        )
