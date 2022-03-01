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

from ANNarchy import *

class test_TimedArray(object):
    """
    Test the correct evaluation of builtin functions
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test. Adapted the example
        from documentation ( section 3.7.3 Class TimedArray )
        """
        inputs = np.array(
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
        inp = TimedArray(rates=inputs)
        inp2 = TimedArray(rates=inputs, schedule=10.)
        inp3 = TimedArray(rates=inputs, period=10.)
        inp4 = TimedArray(rates=inputs, schedule=2., period=20.)

        pop = Population(10, neuron=SimpleNeuron)

        proj = Projection(inp, pop, 'exc')
        proj.connect_one_to_one(1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)
        proj2 = Projection(inp2, pop, 'exc2')
        proj2.connect_one_to_one(1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)
        proj3 = Projection(inp3, pop, 'exc3')
        proj3.connect_one_to_one(1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)
        proj4 = Projection(inp4, pop, 'exc4')
        proj4.connect_one_to_one(1.0, storage_format=cls.storage_format,
                                storage_order=cls.storage_order)

        cls.test_net = Network()
        cls.test_net.add([inp, inp2, inp3, inp4, pop, proj, proj2, proj3, proj4])
        cls.test_net.compile(silent=True)

        cls.output = cls.test_net.get(pop)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the network after every test.
        """
        self.test_net.reset()

    def test_compile(self):
        """
        Enforce compilation of the network.
        """
        pass

    def test_run_one_loop(self):
        """
        We provide 10 ms input data, after 11ms the last element will be set.
        """
        self.test_net.simulate(11)
        np.testing.assert_allclose(self.output.r, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_more_steps(self):
        """
        We provide 10 ms input data, after 11ms the last element will be remain in the buffer.
        """
        self.test_net.simulate(12)
        np.testing.assert_allclose(self.output.r, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_run_one_schedule(self):
        """
        We provide 10 ms input data the 1st entry, after 11ms the 2nd element will be set. So the next 10
        iterations should return the 2nd element of the input array.
        """
        self.test_net.simulate(11) # 1st entry

        self.test_net.simulate(10) # 2nd entry
        np.testing.assert_allclose(self.output.r2, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period(self):
        """
        Period is set to 10, this means after 11 ms we have one cycle complete. This means the 12 step
        should return the 1st element in the output population.
        """
        self.test_net.simulate(12) # 1st entry

        np.testing.assert_allclose(self.output.r3, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_run_one_period_one_schedule(self):
        """
        Schedule == 2 doubles all entries. Period of 20 is one complete iteration. This means
        after 25 steps we are again on the 2nd entry.
        """
        self.test_net.simulate(25)

        np.testing.assert_allclose(self.output.r4, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

