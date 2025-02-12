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
from ANNarchy import load_parameters, save_parameters


from .networks import RateCodedNetwork, SpikingNetwork

def p_from_attr(attributes, isparam):
    return [i for i, j in zip(attributes, isparam) if j]

def assert_allclose_named(actual, expected, names, isparam=None, opt="Loading"):
    """
    Assert allclose and get the names of wrong parameters.
    """
    if isparam is not None:
        expected = p_from_attr(expected, isparam)
        names = p_from_attr(names, isparam)

    try:
        numpy.testing.assert_allclose(expected, actual)
    except AssertionError as exc:
        trace = ""
        for i, name in enumerate(names):
            if not numpy.allclose(expected[i], actual[i]):
                trace += f"\n  {name}: {actual[i]} should be {expected[i]}"
        exc.args = (f"\n {opt} Parameters failed: " + trace,)
        raise


class test_IO_Rate(unittest.TestCase):
    """
    Test basic input output operations for a rate based network.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        cls.network = RateCodedNetwork()
        cls.network.compile(silent=True)

        cls.attribute_names = ["Population Parameter", "Population Rate",
                               "Default Projection Parameter",
                               "Projection Weight", "Projection Parameter"]
        
        cls.init_attr = [1.0, 0.0, 0.01, 0.0, 10.0]
        cls.new_attr = [0.5, 5, 0.02, 3, 20]
        cls.isparam = [True, False, True, False, True]
        cls.savefolder = '_networksave/'
        os.mkdir(cls.savefolder)
        cls.save_extensions = ['.data', '.npz', '.txt.gz']

    @classmethod
    def tearDownClass(cls):
        """ Delete the save folder after all tests were run. """
        rmtree(cls.savefolder)

    def setUp(self):
        """ Clear the network before every test. """
        self.network.reset(projections=True, synapses=True)

    def set_attributes(self, attributes):
        """ Set the attributes of the network. """
        self.network.pop2.set({'baseline': attributes[0], 'r': attributes[1]})
        self.network.proj2.set({'eta': attributes[2], 'w': attributes[3]})
        self.network.proj3.set({'tau': attributes[4]})

    def get_attributes(self):
        """ Check if the attributes of the network are as expected. """
        return [
                self.network.pop2.get('baseline'), 
                self.network.pop2.get('r')[0],
                self.network.proj2.get('eta'), 
                self.network.proj2.get('w')[0][0],
                self.network.proj3.get('tau')
            ]

    def set_parameters(self, parameters):
        """ Set only the parameters of the network. """
        self.network.pop2.set({'baseline': parameters[0]})
        self.network.proj2.set({'eta': parameters[1]})
        self.network.proj3.set({'tau': parameters[2]})

    def get_parameters(self):
        """ Check if only the parameters of the network are as expected. """
        return [self.network.pop2.get('baseline'), self.network.proj2.get('eta'),
                self.network.proj3.get('tau')]

    def test_save_and_load(self):
        """
        Save the network in every loadable format and check if it can be loaded
        """
        for ext in self.save_extensions:

            with self.subTest(extension=ext):
            
                self.set_attributes(self.new_attr)
            
                with patch('sys.stdout', new=io.StringIO()): # suppress print
                    self.network.save(self.savefolder + "ratenet" + ext)
            
                self.network.reset(projections=True, synapses=True)
            
                self.network.load(self.savefolder + "ratenet" + ext)
            
                assert_allclose_named(self.get_attributes(), self.new_attr,
                                      self.attribute_names)

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
        
        self.set_parameters(p_from_attr(self.new_attr, self.isparam))
        
        with patch('sys.stdout', new=io.StringIO()): # suppress print
            save_parameters(self.savefolder + "ratenet.json", net_id=ID)
        
        self.network.reset(projections=True, synapses=True)
        
        load_parameters(self.savefolder + "ratenet.json", net_id=ID)
        
        assert_allclose_named(self.get_parameters(), self.init_attr,
                              self.attribute_names, self.isparam)

    def test_projection_save_and_load(self):
        """
        Save one projection from the network in every loadable format and check
        if it can be loaded
        """
        for ext in self.save_extensions:
            
            with self.subTest(extension=ext):
            
                self.network.proj2.save(self.savefolder + "pr2rate" + ext)
            
                self.network.proj2.load(self.savefolder + "pr2rate" + ext)


class test_IO_Spiking(unittest.TestCase):
    """
    Test basic input output operations for a rate based network.
    """
    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        cls.network = SpikingNetwork()
        cls.network.compile(silent=True)

        cls.attribute_names = ["Population Parameter", "Population V",
                               "STDP Projection Parameter",
                               "Projection Parameter",
                               "Projection Equation Value"]
        cls.init_attr = [8.0, None, 20, 0.5, None]
        cls.new_attr = [0.5, -60, 10, 5, 10]
        cls.isparam = [True, False, True, True, False]
        cls.savefolder = '_networksave/'
        os.mkdir(cls.savefolder)
        cls.save_extensions = ['.data', '.npz', '.txt.gz']

    def setUp(self):
        """ Clear the network before every test. """
        self.network.reset(projections=True, synapses=True)

    @classmethod
    def tearDownClass(cls):
        """ Delete the save folder after all tests were run. """
        rmtree(cls.savefolder)

    def set_attributes(self, attributes):
        """ Set the attributes of the network. """
        self.network.pop2.set({'d': attributes[0], 'v': attributes[1]})
        self.network.proj1.set({'tau_plus': attributes[2]})
        self.network.proj2.set({'U': attributes[3], 'x': attributes[4]})

    def get_attributes(self):
        """ Return the attributes of the network. """
        return [self.network.pop2.get('d')[0], self.network.pop2.get('v')[0],
                self.network.proj1.get('tau_plus'), self.network.proj2.get('U')[0][0],
                self.network.proj2.get('x')[0][0]]

    def set_parameters(self, parameters):
        """ Set the parameters of the network. """
        self.network.pop2.set({'d': parameters[0]})
        self.network.proj1.set({'tau_plus': parameters[1]})
        self.network.proj2.set({'U': parameters[2]})

    def get_parameters(self):
        """ Return the parameters of the network. """
        return [self.network.pop2.get('d')[0], self.network.proj1.get('tau_plus'),
                self.network.proj2.get('U')[0][0]]

    def test_save_and_load(self):
        """
        Save the network in every loadable format and check if it can be loaded
        """
        for ext in self.save_extensions:
            with self.subTest(extension=ext):
                self.set_attributes(self.new_attr)
                with patch('sys.stdout', new=io.StringIO()): # suppress print
                    self.network.save(self.savefolder + "spikenet" + ext)
                self.network.reset(projections=True, synapses=True)
                self.network.load(self.savefolder + "spikenet" + ext)
                assert_allclose_named(self.get_attributes(), self.new_attr,
                                      self.attribute_names)

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
        self.set_parameters(p_from_attr(self.new_attr, self.isparam))
        with patch('sys.stdout', new=io.StringIO()): # suppress print
            save_parameters(self.savefolder + "ratenet.json", net_id=ID)
        self.network.reset(projections=True, synapses=True)
        load_parameters(self.savefolder + "ratenet.json", net_id=ID)
        assert_allclose_named(self.get_parameters(), self.init_attr,
                              self.attribute_names, self.isparam)

    def test_projection_save_and_load(self):
        """
        Save one projection from the network in every loadable format and check
        if it can be loaded
        """
        for ext in self.save_extensions:
            with self.subTest(extension=ext):
                with patch('sys.stdout', new=io.StringIO()): # suppress print
                    self.network.proj2.save(self.savefolder + "pr2spike" + ext)
                self.network.proj2.load(self.savefolder + "pr2spike" + ext)


if __name__ == "__main__":
    unittest.main()
