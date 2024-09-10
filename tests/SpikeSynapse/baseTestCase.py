import unittest
from ANNarchy import clear

class Base:
    class baseTestCase(unittest.TestCase):
        def parametrize(self):
            print("parametrize called")
            suite = []
            # return run_with(type(self), ['lil', 'csr', 'ell'], ['pre_to_post', 'post_to_pre'])
            for s_frmt in ['lil', 'csr', 'ell']:
                for s_ord in ['pre_to_post', 'post_to_pre']:
                    test_name = f"{self.__name__}_{s_ord}_{s_frmt}"
                    new_test = type(test_name, (self, Base.TestCase), {})
                    new_test.storage_order = s_ord
                    new_test.storage_format = s_frmt
                    suite.append(new_test())
            return suite
            # return unittest.TextTestRunner().run(suite)

    class TestCase(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            raise NotImplementedError("baseTestCase function setUpClass must be overloaded")

        @classmethod
        def tearDownClass(cls):
            """
            All tests of this class are done. We can destroy the network.
            """
            del cls.test_net
            clear()

        def setUp(self):
            """
            Automatically called before each test method, basically to reset the
            network after every test.
            In our *setUp()* function we call *reset()* to reset the network before every test.
            """
            self.test_net.reset()


def run_with(c, formats, orders):
    """
    Run the tests with all given storage formats and orders. This is achieved
    by copying the classes for every data format.
    """
    classes = {}
    for s_format in formats:
        for s_order in orders:
            if s_order == "pre_to_post" and s_format not in ["lil", "csr"]:
                continue
            cls_name = c.__name__ + "_" + str(s_format) + "_" + str(s_order)
            glob = {"storage_format":s_format, "storage_order":s_order}
            classes[cls_name] = type(cls_name, (c, unittest.TestCase), glob)
    return classes

