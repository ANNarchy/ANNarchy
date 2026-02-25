"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import unittest

from conftest import TARGET_FOLDER
from ANNarchy import Neuron, Synapse, Network


class test_Locality(unittest.TestCase):
    """
    ANNarchy support several three different localities for
    variables/parameters: synaptic, postsynaptic, projection. This test should
    verify, that these keywords does not lead to compiler errors.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            parameters="""
                r = 0
            """
        )

        syn = Synapse(
            parameters="""
                a = 0.1
                b = 0.1 : postsynaptic
                c = 0.1 : projection
            """
        )

        cls._network = Network()

        pre = cls._network.create(geometry=3, neuron=neuron)
        post = cls._network.create(geometry=1, neuron=neuron)
        proj = cls._network.connect(pre, post, "exc", synapse=syn)
        proj.all_to_all(
            weights=1.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
        )

    def test_compile(self):
        """
        Tests if the network description is compilable.
        """
        self._network.compile(silent=True, directory=TARGET_FOLDER)


class test_AccessPSP(unittest.TestCase):
    """
    In this setup we test, if the access to post-synaptic potential, more
    detailed the statements pre.sum(exc) or post.sum(exc), is correctly
    implemented.

    Other statements like mean(pre.r) are covered by test_GlobalOperation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations="""
                r = sum(exc)
            """
        )

        syn = Synapse(
            equations="""
                w = pre.sum(exc) + post.sum(exc)
            """
        )

        cls._network = Network()

        pre = cls._network.create(geometry=1, neuron=neuron)
        post = cls._network.create(geometry=1, neuron=neuron)

        # to have an "exc" target in pre, we need to create forward and
        # backward connection
        proj1 = cls._network.connect(pre, post, "exc", synapse=syn)
        proj1.all_to_all(
            weights=1.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
            force_multiple_weights=True,
        )
        proj2 = cls._network.connect(post, pre, "exc", synapse=syn)
        proj2.all_to_all(
            weights=1.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
            force_multiple_weights=True,
        )

    def test_compile(self):
        """
        Tests if the network description is compilable.
        """
        self._network.compile(silent=True, directory=TARGET_FOLDER)


class test_ModifiedPSP(unittest.TestCase):
    """
    Test modified psp statements
    """

    @classmethod
    def setUpClass(cls):
        """
        Compile the network for this test
        """
        neuron = Neuron(
            equations="""
                r = sum(exc)
            """
        )

        ReversedSynapse = Synapse(
            parameters="""
                reversal = 1.0
            """,
            psp="""
                w*pos(reversal-pre.r)
            """,
        )

        cls._network = Network()

        pre = cls._network.create(geometry=1, neuron=neuron)
        post = cls._network.create(geometry=1, neuron=neuron)

        # to have an "exc" target in pre, we need to create forward and
        # backward connection
        proj = cls._network.connect(pre, post, "exc", synapse=ReversedSynapse)
        proj.all_to_all(
            weights=1.0,
            storage_format=cls.storage_format,
            storage_order=cls.storage_order,
            force_multiple_weights=True,
        )

    def test_compile(self):
        """
        Tests if the network description is compilable.
        """
        self._network.compile(silent=True, directory=TARGET_FOLDER)
