"""

    test_openmp.py

    This file is part of ANNarchy.

    Copyright (C) 2020 Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>,
    Julien Vitay <julien.vitay@informatik.tu-chemnitz.de>

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

from Unittests import *
from ANNarchy import setup

def run_suite(values, runner):
    suite = unittest.TestSuite()
    for mod in d_glob:
        suite.addTest(unittest.makeSuite(mod))
    return runner.run(suite)


if __name__ == '__main__':
    glob = dict(globals())
    d_glob = [glob[k] for k in glob if isinstance(glob[k], type) and
              issubclass(glob[k], (unittest.case.TestCase, unittest.TestSuite))]
    # pr = list(it.product(("lil", "csr", "ell"), ("pre_to_post", "post_to_pre")))

    runner = unittest.TextTestRunner(verbosity=2)
    print("Testing single thread run")
    result0 = run_suite(d_glob, runner)
    print("Testing multi threaded run")
    setup(num_threads=3)
    result1 = run_suite(d_glob, runner)
    print("Testing CUDA run")
    setup(paradigm="cuda")
    result2 = run_suite(d_glob, runner)

