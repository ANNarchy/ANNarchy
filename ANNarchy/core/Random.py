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
    'Uniform' : 2,
    'DiscreteUniform': 2,
    'Normal' : 2,
    'LogNormal': 2,
    'Exponential': 1,
    'Gamma': 2
}
distributions_templates = {
    'Uniform' : '<double>',
    'DiscreteUniform': '<>',
    'Normal' : '<double>',
    'LogNormal': '',
    'Exponential': '',
    'Gamma': ''
}
distributions_equivalents = {
    'Uniform' : 'std::uniform_real_distribution<double>',
    'DiscreteUniform': 'TODO',
    'Normal' : 'std::normal_distribution<double>',
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
        return list(self.get_values(size))

    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values((1))[0]

    def keywords(self):
        return available_distributions

class Uniform(RandomDistribution):
    """
    Random distribution object using the uniform distribution between ``min`` and ``max``.

    The returned values are floats in the range [min, max].
    """   
    def __init__(self, min, max, seed=-1):
        """        
        *Parameters*:
        
        * **min**: minimum value.
        
        * **max**: maximum value.
        
        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.
        """
        self.min = min
        self.max = max
        if seed == -1:
            seed = Global.config['seed']
        self._cpp_seed = seed
        
    def get_values(self, shape):
        """
        Returns a Numpy array with the given shape.
        """
        if self._cpp_seed != -1:
            np.random.seed(self._cpp_seed)
        return np.random.uniform(self.min, self.max, shape)

    def _gen_cpp(self):
        return 'UniformDistribution<double>('+str(self.min)+','+str(self.max)+', '+str(self._cpp_seed)+')'

class DiscreteUniform(RandomDistribution):
    """
    Random distribution object using the discrete uniform distribution between ``min`` and ``max``.

    The returned values are integers in the range [min, max].
    """   
    def __init__(self, min, max, seed=-1):
        """        
        *Parameters*:
        
        * **min**: minimum value
        
        * **max**: maximum value
        
        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.
        """
        self.min = min
        self.max = max
        if seed == -1:
            seed = Global.config['seed']
        self._cpp_seed = seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        if self._cpp_seed != -1:
            np.random.seed(self._cpp_seed)
        return np.random.random_integers(self.min, self.max, shape)
    
    def _gen_cpp(self):
        return 'UniformDistribution<int>('+str(self.min)+','+str(self.max)+', '+str(self._cpp_seed)+')'
    
class Normal(RandomDistribution):
    """
    Random distribution instance returning a random value based on a normal (Gaussian) distribution.
    """   
    def __init__(self, mu, sigma, seed=-1):
        """        
        *Parameters*:
        
        * **mu**: mean of the distribution
        
        * **sigma**: standard deviation of the distribution
        
        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.
        """
        self.mu = mu
        self.sigma = sigma
        if seed == -1:
            seed = Global.config['seed']
        self._cpp_seed = seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        if self._cpp_seed != -1:
            np.random.seed(self._cpp_seed)
        return np.random.normal(self.mu, self.sigma, shape)
    
    def _gen_cpp(self):
        return 'NormalDistribution<double>('+str(self.mu)+','+str(self.sigma)+', '+str(self._cpp_seed)+')'

class LogNormal(RandomDistribution):
    """
    Random distribution instance returning a random value based on lognormal distribution.
    """   
    def __init__(self, mu, sigma, seed=-1):
        """        
        *Parameters*:
        
        * **mu**: mean of the distribution
        
        * **sigma**: standard deviation of the distribution
        
        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.
        """
        self.mu = mu
        self.sigma = sigma
        if seed == -1:
            seed = Global.config['seed']
        self._cpp_seed = seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        if self._cpp_seed != -1:
            np.random.seed(self._cpp_seed)
        return np.random.lognormal(self.mu, self.sigma, shape)

    def _gen_cpp(self):
        return 'LogNormalDistribution('+str(self.mu)+','+str(self.sigma)+', '+str(self._cpp_seed)+')'

class Exponential(RandomDistribution):
    """
    Random distribution instance returning a random value based on exponential distribution, according the density function:
    
    .. math ::
    
        P(x | \lambda) = \lambda e^{(-\lambda x )}

    """   
    def __init__(self, Lambda, seed=-1):
        """        
        *Parameters*:
        
        * **Lambda**: rate parameter.
        
        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.
        
        .. note::

            ``Lambda`` is capitalized, otherwise it would be a reserved Python keyword.

        """
        self.Lambda = Lambda
        if seed == -1:
            seed = Global.config['seed']
        self._cpp_seed = seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape.
        """
        if self._cpp_seed != -1:
            np.random.seed(self._cpp_seed)
        return np.random.exponential(self.Lambda, shape)

    def _gen_cpp(self):
        return 'ExponentialDistribution('+str(self.Lambda)+')'

class Gamma(RandomDistribution):
    """
    Random distribution instance returning a random value based on gamma distribution.
    """   
    def __init__(self, alpha, beta=1.0, seed=-1):
        """        
        *Parameters*:
        
        * **alpha**: shape of the gamma distribution
        
        * **beta**: scale of the gamma distribution
        
        * **seed**: seed for the random number generator. By default, the seed takes the value defined in ``ANNarchy.setup()``.
        """
        self.alpha = alpha
        self.beta = beta
        if seed == -1:
            seed = Global.config['seed']
        self._cpp_seed = seed
        
    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        if self._cpp_seed != -1:
            np.random.seed(self._cpp_seed)
        return np.random.gamma(self.alpha, self.beta, shape)

    def _gen_cpp(self):
        return 'GammaDistribution('+str(self.alpha)+','+str(self.beta)+', '+str(self._cpp_seed)+')'
