""" 

    RateProjectionGenerator.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
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

from ProjectionGenerator import ProjectionGenerator
from Templates import *

class RateProjectionGenerator(ProjectionGenerator):
    """ Class for generating C++ code from a rate population description. """
    def __init__(self, name, desc):
        ProjectionGenerator.__init__(self, name, desc)
            
    def generate_header(self):
        " Generates the C++ header file."
        
        # Access method for attributes
        access_header = self.generate_dendrite_access_declaration()
        
        # Custom function
        functions = self.generate_functions()
        
        # Generate the code
        template = rate_projection_header
        dictionary = { 
            'class': self.name, 
            'dend_class': self.name.replace('Projection', 'Dendrite'),
            'pre_name': self.desc['pre_class'],
            'post_name': self.desc['post_class'],
            'access': access_header,
            'functions': functions }
        return template % dictionary

    def generate_body(self):

        # Access method for attributes
        access_body = self.generate_dendrite_access_definition()
        
        # Generate the code
        template = rate_projection_body
        dictionary = {         
            'class': self.name,
            'add_include': self.generate_add_proj_include(),
            'dend_class': self.name.replace('Projection', 'Dendrite'), 
            'access': access_body,
            'destructor': '' ,
            'pre_type': self.desc['pre_class'],
            'post_type': self.desc['post_class']
        } 
        return template % dictionary
    
    def generate_pyx(self):
        """
        Generate complete cython wrapper class
            
        Notes:
            * dependent on coding.
        """
        # Get the C++ methods
        cwrappers = self.generate_cwrappers()
        # Get the python functions
        pyfunctions = self.generate_pyfunctions()
        # Generate the code

        template = rate_projection_pyx
        dictionary = { 
            'name': self.name, 
            'cFunction': cwrappers, 
            'pyFunction': pyfunctions
        }
        return template % dictionary    