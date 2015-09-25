import unittest
import numpy

from ANNarchy import *
setup(paradigm="cuda")

neuron = Neuron(
    parameters = "tau = 10",
    equations="r = t"
)

neuron2 = Neuron(
    parameters = "tau = 10: population",
    equations="r = sum(exc)"
)

Oja = Synapse(
    parameters="""
        tau = 5000.0
        alpha = 8.0
    """,
    equations = """
        r = t
    """
)

pop1 = Population((3, 3), neuron)
pop2 = Population((3, 3), neuron2)

proj = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)

proj2 = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)


proj3 = Projection(
     pre = pop1,
     post = pop2,
     target = "exc",
     synapse = Oja
)

proj.connect_one_to_one(weights = 1.0)
proj2.connect_all_to_all(weights = 1.0)
proj3.connect_one_to_one(weights = 1.0, delays = 10.0)

compile(clean=True)


class test_Connectivity(unittest.TestCase):

    def setUp(self):
        """
        basic setUp() method to reset the network after every test
        """
        reset()

    def test_one_to_one(self):
        """
        tests functionality of the one_to_one connectivity pattern
        """
        self.assertEqual(proj.dendrite(3).rank, [3])
        self.assertTrue(numpy.allclose(proj.dendrite(3).w, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_all_to_all(self):
        """
        tests functionality of the all_to_all connectivity pattern
        """
        self.assertEqual(proj2.dendrite(3).rank, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(numpy.allclose(proj2.dendrite(3).w, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    def test_delay(self):
        """
        tests the delay functionality
        """
        simulate(5)
        self.assertTrue(numpy.allclose(proj3.post.r, 30.0))











