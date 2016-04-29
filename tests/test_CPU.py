"""

    test_CPU.py

    This file is part of ANNarchy.

    Copyright (C) 2013-2016 Joseph Gussev <joseph.gussev@s2012.tu-chemnitz.de>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

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
from __future__ import print_function
import os
from subprocess import call

nb_errors = 0
nb_tests = 0

for f in os.listdir('tests/CPU'):
    if f.startswith('test_') and f.endswith('.py'):
        print('Testing', f, '...')
        ret = call(['python', '-m', 'unittest', f.replace('.py', '')], cwd = 'tests/CPU')
        if ret != 0: # Test failed
            nb_errors += 1
        nb_tests += 1

if nb_errors != 0:
    print('Some tests failed:', nb_errors, '/', nb_tests)
else:
    print('Everything is fine.')
