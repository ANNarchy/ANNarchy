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
from ANNarchy.core import Global

distributions_arguments = {
    'Constant' : 0,
    'Uniform' : 2,
    'DiscreteUniform': 2,
    'Normal' : 2,
    'LogNormal': 2,
    'Exponential': 1,
    'Gamma': 2
}
distributions_templates = {
    'Constant' : '<DATA_TYPE>',
    'Uniform' : '<DATA_TYPE>',
    'DiscreteUniform': '<>',
    'Normal' : '<DATA_TYPE>',
    'LogNormal': '',
    'Exponential': '',
    'Gamma': ''
}

# List of available distributions
available_distributions = distributions_arguments.keys()

class RandomDistribution(object):
    """ 
    BaseClass for random distributions.
    """

    def __init__(self):
        pass

    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        Global._error('instantiated base class RandomDistribution is not allowed.')
        return 0.0

    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        Global._error('instantiated base class RandomDistribution is not allowed.')
        return []

    def get_value(self):
        """
        Returns a single float value.
        """
        Global._error('instantiated base class RandomDistribution is not allowed.')
        return 0.0

    def keywords(self):
        return available_distributions

class Constant(RandomDistribution):
    """
    Random distribution instance returning a constant value.
    """
    def __init__(self, value):
        """        
        Parameter:
        
        * *value*: the constant value
        """
        self._value = value
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return self._value * np.ones(shape)
        
    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return [self._value for i in range(size) ]
    
    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]
    
    def max(self):
        return self._value

    def min(self):
        return self._value

    def _gen_cpp(self):
        return 'Constant<DATA_TYPE>('+str(self._value)+')'

class Uniform(RandomDistribution):
    """
    Random distribution instance returning a random value based on uniform distribution for floating point values.
    """   
    def __init__(self, min, max, cpp_seed=-1):
        """        
        Parameters:
        
        * *min*: minimum value
        
        * *max*: maximum value
        
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._min = min
        self._max = max
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return np.random.uniform(self._min, self._max, shape)
    
    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(np.random.uniform(self._min, self._max, size))

    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]

    def max(self):
        return self._max

    def min(self):
        return self._min

    def _gen_cpp(self):
        return 'UniformDistribution<DATA_TYPE>('+str(self._min)+','+str(self._max)+', '+str(self._cpp_seed)+')'

class DiscreteUniform(RandomDistribution):
    """
    Random distribution instance returning a random value based on uniform distribution for integer values.
    """   
    def __init__(self, min, max, cpp_seed=-1):
        """        
        Parameters:
        
        * *min*: minimum value
        
        * *max*: maximum value
        
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._min = min
        self._max = max
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return np.random.uniform(self._min, self._max, shape)
    
    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(np.random.uniform(self._min, self._max, size))

    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]

    def max(self):
        return self._max

    def min(self):
        return self._min
    
    def _gen_cpp(self):
        return 'UniformDistribution<int>('+str(self._min)+','+str(self._max)+', '+str(self._cpp_seed)+')'
    
class Normal(RandomDistribution):
    """
    Random distribution instance returning a random value based on uniform distribution.
    """   
    def __init__(self, mu, sigma, cpp_seed=-1):
        """        
        Parameters:
        
        * *mu*: mean of the distribution
        
        * *sigma*: standard deviation of the distribution
        
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._mu = mu
        self._sigma = sigma
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return np.random.normal(self._mu, self._sigma, shape)
    
    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(np.random.normal(self._mu, self._sigma, size))
    
    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]
        
    def mu(self):
        return self._mu

    def sigma(self):
        return self._sigma
    
    def _gen_cpp(self):
        return 'NormalDistribution<DATA_TYPE>('+str(self._mu)+','+str(self._sigma)+', '+str(self._cpp_seed)+')'

class LogNormal(RandomDistribution):
    """
    Random distribution instance returning a random value based on lognormal distribution.
    """   
    def __init__(self, mu, sigma, cpp_seed=-1):
        """        
        Parameters:
        
        * *mu*: mean of the distribution
        
        * *sigma*: standard deviation of the distribution
        
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._mu = mu
        self._sigma = sigma
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return np.random.lognormal(self._mu, self._sigma, shape)
    
    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(np.random.lognormal(self._mu, self._sigma, size))
    
    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]
        
    def mu(self):
        return self._mu

    def sigma(self):
        return self._sigma

    def _gen_cpp(self):
        return 'LogNormalDistribution('+str(self._mu)+','+str(self._sigma)+', '+str(self._cpp_seed)+')'

class Exponential(RandomDistribution):
    """
    Random distribution instance returning a random value based on exponential distribution, according the density function:
    
    .. math ::
    
        P(x | \lambda) = \lambda e^{(-\lambda x )}

    """   
    def __init__(self, Lambda, cpp_seed=-1):
        """        
        Parameters:
        
        * *Lambda*: rate parameter
        
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._lambda = Lambda
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return np.random.exponential(self._lambda, shape)
    
    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(np.random.exponential(self._lambda, size))

    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]
    
    def Lambda(self):
        return self._lambda

    def _gen_cpp(self):
        return 'ExponentialDistribution('+str(self._lambda)+')'

class Gamma(RandomDistribution):
    """
    Random distribution instance returning a random value based on gamma distribution.
    """   
    def __init__(self, alpha, beta=1.0, cpp_seed=-1):
        """        
        Parameters:
        
        * *alpha*: shape of the gamma distribution
        
        * *beta*: scale of the gamma distribution
        
        * *cpp_seed*: seed value for cpp. If cpp_seed == -1, the cpp seed will be initialized without a special.
        """
        self._alpha = alpha
        self._beta = beta
        self._cpp_seed = cpp_seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return np.random.gamma(self._alpha, self._beta, shape)
    
    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(np.random.gamma(self._alpha, self._beta, size))

    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]
    
    def alpha(self):
        return self._alpha

    def beta(self):
        return self._beta

    def _gen_cpp(self):
        return 'GammaDistribution('+str(self._alpha)+','+str(self._beta)+', '+str(self._cpp_seed)+')'
