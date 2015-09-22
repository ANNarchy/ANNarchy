"""

    RUN_ALL_GPU.py

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
from subprocess import call

#
# This file simply runs all CUDA-tests through a single commandfrom command line.
#
# prompt> python RUN_ALL_CPU.py
#
print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_SpikingNeuronCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_SpikingNeuron'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_PopulationCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_Population'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_Population2CUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_Population2'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_PopulationViewCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_PopulationView'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_ProjectionCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_Projection'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_DendriteCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_Dendrite'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_StructuralPlasticityCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_StructuralPlasticity'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_RecordCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_Record'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRUNNING TEST >>test_ConnectivityCUDA<<\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
call(['python', '-m', 'unittest', 'test_Connectivity'], cwd = 'GPU')

print '\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nFINISHED TESTING! HAVE SOME CAKE!\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
