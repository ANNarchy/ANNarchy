"""

    CUDAProfile.py

    This file is part of ANNarchy.

    Copyright (C) 2016-2018  Julien Vitay <julien.vitay@gmail.com>,
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
from ANNarchy.core import Global

from ProfileGenerator import ProfileGenerator
from ProfileTemplate import profile_template

class CUDAProfile(ProfileGenerator):
    
    def __init__(self, annarchy_dir, net_id):
        ProfileGenerator.__init__(self, annarchy_dir, net_id)

    def generate(self):
        """
        Generate Profiling class code, called from Generator instance.
        """
        # Generate header for profiling
        with open(self.annarchy_dir+'/generate/net'+str(self._net_id)+'/Profiling.h', 'w') as ofile:
            ofile.write(self._generate_header())

    def generate_init_population(self, pop):
        return "", ""

    def generate_init_projection(self, proj):
        return "", ""

    def _generate_header(self):
        """
        generate Profiling.h
        """
        from .ProfileTemplate import profile_header

        config_xml = """
        _out_file << "  <config>" << std::endl;
        _out_file << "    <paradigm>%(paradigm)s</paradigm>" << std::endl;
        _out_file << "  </config>" << std::endl;
        """ % {'paradigm': Global.config["paradigm"]}
        
        config = Global.config["paradigm"]
        return profile_header % { 'config': config, 'config_xml': config_xml }
