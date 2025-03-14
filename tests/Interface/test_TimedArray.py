"""

    test_TimedArray.py

    This file is part of ANNarchy.

    Copyright (C) 2018-2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
    Copyright (C) 2019 Alex Schwarz <alex.schwarz@informatik.tu-chemnitz.de>

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
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
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

        inp = cls._network.create(geometry=10, population=TimedArray(rates=inputs))
        inp2 = cls._network.create(geometry=10, population=TimedArray(rates=inputs, schedule=10.))
        inp3 = cls._network.create(geometry=10, population=TimedArray(rates=inputs, period=10.))
        inp4 = cls._network.create(geometry=10, population=TimedArray(rates=inputs, schedule=2., period=20.))

        cls.output = cls._network.create(10, neuron=SimpleNeuron)

        proj = cls._network.connect(inp, cls.output, 'exc')
        proj.one_to_one(1.0)
        proj2 = cls._network.connect(inp2, cls.output, 'exc2')
        proj2.one_to_one(1.0)
        proj3 = cls._network.connect(inp3, cls.output, 'exc3')
        proj3.one_to_one(1.0)
        proj4 = cls._network.connect(inp4, cls.output, 'exc4')
        proj4.one_to_one(1.0)

        cls._network.compile(silent=True)

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
        self._network.simulate(11) # 1st entry

        self._network.simulate(10) # 2nd entry
        numpy.testing.assert_allclose(self.output.r2, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period(self):
        """
        Period is set to 10, this means after 11 ms we have one cycle complete. This means the 12 step
        should return the 1st element in the output population.
        """
        self._network.simulate(12) # 1st entry

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
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
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
        inp = cls._network.create(geometry=10, population=TimedArray(geometry=(10)))
        inp2 = cls._network.create(geometry=10, population=TimedArray(geometry=(10)))
        inp3 = cls._network.create(geometry=10, population=TimedArray(geometry=(10)))
        inp4 = cls._network.create(geometry=10, population=TimedArray(geometry=(10)))

        # Test later initialization
        inp.update(rates=inputs)
        inp2.update(rates=inputs, schedule=10.)
        inp3.update(rates=inputs, period=10.)
        inp4.update(rates=inputs, schedule=2., period=20.)

        # The output signal will be tested
        cls.output = cls._network.create(10, neuron=SimpleNeuron)

        proj = cls._network.connect(inp, cls.output, 'exc')
        proj.one_to_one(1.0)
        proj2 = cls._network.connect(inp2, cls.output, 'exc2')
        proj2.one_to_one(1.0)
        proj3 = cls._network.connect(inp3, cls.output, 'exc3')
        proj3.one_to_one(1.0)
        proj4 = cls._network.connect(inp4, cls.output, 'exc4')
        proj4.one_to_one(1.0)

        cls._network.compile(silent=True)

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
        self._network.simulate(11) # 1st entry

        self._network.simulate(10) # 2nd entry
        numpy.testing.assert_allclose(self.output.r2, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period(self):
        """
        Period is set to 10, this means after 11 ms we have one cycle complete. This means the 12 step
        should return the 1st element in the output population.
        """
        self._network.simulate(12) # 1st entry

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
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
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
                [2, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ]
        )

        SimpleNeuron = Neuron(
            equations="""
                r = sum(exc_1) + sum(exc_2)
            """
        )

        cls._network = Network()

        cls.input = cls._network.create(geometry=10, population=TimedArray(rates=cls.initial_inputs))
        cls.input2 = cls._network.create(geometry=10, population=TimedArray(rates=cls.initial_inputs, schedule=2.))

        cls.output = cls._network.create(geometry=10, neuron=SimpleNeuron)

        proj1 = cls._network.connect(cls.input, cls.output, 'exc_1')
        proj1.one_to_one(1.0)

        proj2 = cls._network.connect(cls.input2, cls.output, 'exc_2')
        proj2.one_to_one(1.0)

        cls._network.compile(silent=True)

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
        expected result is the last value in the input buffer, as cycling is disabled.
        """
        self._network.simulate(5)
        self.input.update(self.overwrite_inputs)
        self._network.simulate(6)
        numpy.testing.assert_allclose(self.output.sum("exc_1"), [2, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_run_one_loop_and_period(self):
        """
        We provide 5 ms input data, then we set another input sequence and simulate 6 ms more. As we
        have a schedule of 2 ms, the 5-th position should be read out.
        """
        self._network.simulate(5)
        self.input2.update(self.overwrite_inputs, schedule=2.0)
        self._network.simulate(6)
        numpy.testing.assert_allclose(self.output.sum("exc_2"), [0, 0, 0, 0, 1, 2, 0, 0, 0, 0])

