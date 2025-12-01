"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""
import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron

class test_GlobalOps_1D(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects. Currently the following methods
    are supported:

        * mean()
        * max()
        * min()
        * norm1()
        * norm2()

    They are used in the equations of our neuron definition. This particular
    test focuses on a one-dimensional *Population*.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r=0
            """,
            equations="""
                mean_r = mean(r)
                max_r = max(r)
                min_r = min(r)
                l1 = norm1(r)
                l2 = norm2(r)
            """
        )

        cls._network = Network()
        cls._population = cls._network.create(geometry=6, neuron=neuron)
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we set the variable *r*.
        We also call *simulate()* to calculate mean/max/min.
        """
        # reset() set all variables to init value (default 0), which is
        # unfortunately meaningless for mean/max/min. So we set here some
        # better values
        self._population.r = [2.0, 1.0, 0.0, -5.0, -3.0, -1.0]

        # 1st step: calculate mean/max/min and store in intermediate
        #           variables
        # 2nd step: write intermediate variables to accessible variables.
        self._network.simulate(2)

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_get_mean_r(self):
        """
        Tests the result of *mean(r)* for *pop*.
        """
        numpy.testing.assert_allclose(self._population.mean_r, -1.0)

    def test_get_max_r(self):
        """
        Tests the result of *max(r)* for *pop*.
        """
        numpy.testing.assert_allclose(self._population.max_r, 2.0)

    def test_get_min_r(self):
        """
        Tests the result of *min(r)* for *pop*.
        """
        numpy.testing.assert_allclose(self._population.min_r, -5.0)

    def test_get_l1_norm(self):
        """
        Tests the result of *norm1(r)* (L1 norm) for *pop*.
        """
        numpy.testing.assert_allclose(self._population.l1, 12.0)

    def test_get_l2_norm(self):
        """
        Tests the result of *norm2(r)* (L2 norm) for *pop*.
        """
        # compute control value
        l2norm = numpy.linalg.norm(self._population.r, ord=2)

        # test
        numpy.testing.assert_allclose(self._population.l2, l2norm)

class test_GlobalOps_1D_Large(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects. Currently the following methods
    are supported:

        * mean()
        * max()
        * min()
        * norm1()
        * norm2()

    They are used in the equations of our neuron definition. This particular
    test focuses on a large one-dimensional *Population* instances. Contrary
    to *test_GlobalOps_1D* this object contains enough elements that a 
    parallelization will be applied.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r=0
            """,
            equations="""
                mean_r = mean(r)
                max_r = max(r)
                min_r = min(r)
                l1 = norm1(r)
                l2 = norm2(r)
            """
        )

        cls._network = Network()
        cls._population = cls._network.create(geometry=500, neuron=neuron)
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_mean_r(self):
        """
        """
        rand_val = numpy.random.random(500)
        self._population.r = rand_val
        self._network.simulate(2)

        numpy.testing.assert_allclose(self._population.mean_r, numpy.mean(rand_val))

    def test_min_r(self):
        """
        """
        rand_val = numpy.random.random(500)
        self._population.r = rand_val
        self._network.simulate(2)

        numpy.testing.assert_allclose(self._population.min_r, numpy.amin(rand_val))

    def test_max_r(self):
        """
        """
        rand_val = numpy.random.random(500)
        self._population.r = rand_val
        self._network.simulate(2)

        numpy.testing.assert_allclose(self._population.max_r, numpy.amax(rand_val))

class test_GlobalOps_2D(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects. Currently the following methods
    are supported:

        * mean()
        * max()
        * min()
        * norm1()
        * norm2()

    They are used in the equations of our neuron definition.
    This particular test focuses on a two-dimensional *Population*.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r=0
            """,
            equations="""
                mean_r = mean(r)
                max_r = max(r)
                min_r = min(r)
                l1 = norm1(r)
                l2 = norm2(r)
            """
        )

        cls._network = Network()
        cls._population = cls._network.create(geometry=(2, 3), neuron=neuron)

        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we set the variable *r*.
        We also call *simulate()* to calculate mean/max/min.
        """
        # reset() set all variables to init value (default 0), which is
        # unfortunately meaningless for mean/max/min. So we set here some
        # better values
        self._population.r = numpy.array(
            [[ 2.0,  1.0,  0.0],
             [-5.0, -3.0, -1.0]]
        )

        # 1st step: calculate mean/max/min and store in intermediate
        #           variables
        # 2nd step: write intermediate variables to accessible variables.
        self._network.simulate(2)

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_get_mean_r(self):
        """
        Tests the result of *mean(r)* for *pop*.
        """
        numpy.testing.assert_allclose(self._population.mean_r, -1.0)

    def test_get_max_r(self):
        """
        Tests the result of *max(r)* for *pop*.
        """
        numpy.testing.assert_allclose(self._population.max_r, 2.0)

    def test_get_min_r(self):
        """
        Tests the result of *min(r)* for *pop*.
        """
        numpy.testing.assert_allclose(self._population.min_r, -5.0)

    def test_get_l1_norm(self):
        """
        Tests the result of *norm1(r)* for *pop*.
        """
        numpy.testing.assert_allclose(self._population.l1, 12.0)

class test_GlobalOps_MultiUse(unittest.TestCase):
    """
    ANNarchy support several global operations, there are always applied on
    variables of *Population* objects. Currently the following methods
    are supported:

    Hint:

    Contrary to *test_GlobalOps_1D*, *test_GlobalOps_2D* and *test_GlobalOps_1D_Large*
    this class tests only compilation not correctness.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test.
        """
        neuron = Neuron(
            parameters="""
                r=0
            """,
            equations="""
                tmp = max(r)
                tmp2 = max(r)
                tmp3 = mean(r) + mean(r)
            """
        )

        cls._network = Network()
        cls._population = cls._network.create(geometry=6, neuron=neuron)
        cls._network.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        All tests of this class are done. We can destroy the network.
        """
        del cls._network

    def setUp(self):
        """
        In our *setUp()* function we set the variable *r*.
        We also call *simulate()* to calculate mean/max/min.
        """
        # reset() set all variables to init value (default 0), which is
        # unfortunately meaningless for mean/max/min. So we set here some
        # better values
        self._population.r = [2.0, 1.0, 0.0, -5.0, -3.0, -1.0]

        # 1st step: calculate mean/max/min and store in intermediate
        #           variables
        # 2nd step: write intermediate variables to accessible variables.
        self._network.simulate(2)

    def tearDown(self):
        """
        After each test we call *reset()* to reset the network.
        """
        self._network.reset()

    def test_compile(self):
        # fake test, the real test if the network can be compiled succesfully ...
        numpy.testing.assert_allclose(self._population.r, [2.0, 1.0, 0.0, -5.0, -3.0, -1.0])