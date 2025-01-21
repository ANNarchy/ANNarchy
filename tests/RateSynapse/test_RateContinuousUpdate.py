"""

    test_ContinuousUpdate.py

    This file is part of ANNarchy.

    Copyright (C) 2016-2018 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
from ANNarchy import Neuron, Synapse, Population, Projection, Network

class test_RateCodedContinuousUpdate(unittest.TestCase):
    """
    Test the correct evaluation of local equation updates in synapses.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        simple_neuron = Neuron(
            parameters="r=1.0"
        )

        eq_set = Synapse(
            equations="""
                glob_var = 0.1 : projection
                semi_glob_var = 0.2 : postsynaptic
                w = t + glob_var + semi_glob_var
            """
        )

        cls._network = Network()

        pop0 = cls._network.create(geometry=3, neuron=simple_neuron)
        pop1 = cls._network.create(geometry=1, neuron=simple_neuron)

        proj = cls._network.connect(pop0, pop1, "exc", eq_set)
        proj.connect_all_to_all(
            weights=0.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order
        )

        cls._network.compile(silent=True)

    def setUp(self):
        """
        Automatically called before each test method, basically to reset the
        network after every test.
        """
        self._network.reset() # network reset

    def test_invoke_compile(self):
        self._network.simulate(1)
