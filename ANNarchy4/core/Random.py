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
    """ 
    BaseClass for random distributions.
    """

    def __init__(self):
        pass

    def get_values(self, shape):
        print 'ERROR: instantiated base class RandomDistribution is not allowed.'
        return 0.0

    def get_value(self):
        print 'ERROR: instantiated base class RandomDistribution is not allowed.'
        return 0.0

    def genCPP(self):
        print 'ERROR: instantiated base class RandomDistribution is not allowed.'
        return ''

class Constant(RandomDistribution):
    def __init__(self, value):
        """
        Constructor.
        
        Parameter:
        * *value*: the constant value
        """
        self._value = value
        
    def get_values(self, shape):
        return self._value * np.ones(shape)
    
    def get_value(self):
        return self.get_values((1))
    
    def _gen_cpp(self):
        return 'Constant<DATA_TYPE>('+str(self._value)+')'
        
class Uniform(RandomDistribution):
    def __init__(self, min, max, cpp_seed=-1):
        """
        Constructor.
        
        Parameters:
        * *min*: min
        * *min*: min
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._min = min
        self._max = max
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        return np.random.uniform(self._min, self._max, shape)
    
    def get_value(self):
        return self.get_values((1))
    
    def _gen_cpp(self):
        if(self._cpp_seed == -1):
            return 'UniformDistribution<DATA_TYPE>('+str(self._min)+','+ str(self._max)+')'
        else:
            return 'UniformDistribution<DATA_TYPE>('+str(self._min)+','+ str(self._max)+','+ str(self._cpp_seed)+')'

class Normal(RandomDistribution):
    def __init__(self, mu, sigma, cpp_seed=-1):
        """
        Constructor.
        
        Parameters:
        * *mu*: mean
        * *sigma*: standard deviation
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._mu = mu
        self._sigma = sigma
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        return np.random.normal(self._mu, self._sigma, shape)
    
    def get_value(self):
        return self.get_values((1))
    
    def _gen_cpp(self):
        if(self._cpp_seed == -1):
            return 'NormalDistribution<DATA_TYPE>('+str(self._mu)+','+ str(self._sigma)+')'
        else:
            return 'NormalDistribution<DATA_TYPE>('+str(self._mu)+','+ str(self._sigma)+','+ str(self._cpp_seed)+')'
        