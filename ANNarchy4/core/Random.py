"""

    Random.py
    
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
import numpy as np

class RandomDistribution(object):
    """ Class returning values from a random distributoin.
    
    Distributions available:
    
    * constant
    
    * uniform
    
    * normal
    """

    def __init__(self, distribution='uniform', parameters=[0,1], seed=-1):
        self.distribution = distribution
        self.parameters = parameters
        self.seed = seed

    def get_values(self, shape):
      
        if self.distribution == 'uniform':
            return np.random.uniform(self.parameters[0], self.parameters[1], shape)
        if self.distribution == 'normal':
            return np.random.normal(self.parameters[0], self.parameters[1], shape)
        if self.distribution == 'constant':
            return self.parameters[0] * np.ones(shape)
        else:
            print 'Unknown distribution', self.distribution
            return None

    def get_value(self):
        return self.get_values((1))

    def genCPP(self):
        code = ''

        if self.distribution == 'uniform':
            # (min, max)
            if len(self.parameters)==2:
                if(self.seed == -1):
                    code = 'UniformDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+')'
                else:
                    code = 'UniformDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+','+ str(self.seed)+')'
            else:
                print 'wrong parameters for uniform distribution, expected [min, max]'
        elif self.distribution == 'normal':
            # (mean, sigma)
            if len(self.parameters)==2:
                if(self.seed == -1):
                    code = 'NormalDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+')'
                else:
                    code = 'NormalDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+','+ str(self.seed)+')'
            else:
                print 'wrong parameters for normal distribution, expected [mean, sigma].'
        elif self.distribution == 'constant':
            # (constant)
            if len(self.parameters)==1:
                code = 'Constant<DATA_TYPE>('+str(self.parameters[0])+')'
            else:
                print 'wrong parameters for constant, expected [constant].'
        else:
            print 'unknown distribution type.'

        return code
