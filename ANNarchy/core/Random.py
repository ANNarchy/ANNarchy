#===============================================================================
#
#     Random.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
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

distributions_equivalents = {
    'Uniform' : 'std::uniform_real_distribution< %(float_prec)s >',
    'DiscreteUniform': 'std::uniform_int_distribution<int>',
    'Normal' : 'std::normal_distribution< %(float_prec)s >',
    'LogNormal': 'std::lognormal_distribution< %(float_prec)s >',
    'Exponential': 'std::exponential_distribution< %(float_prec)s >',
    'Gamma': 'std::gamma_distribution< %(float_prec)s >'
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
        return np.array([0.0])

    def get_list_values(self, size):
        """
        Returns a list of the given size.
        """
        return list(self.get_values(size))

    def get_value(self):
        """
        Returns a single float value.
        """
        return self.get_values(1)[0]

    def keywords(self):
        return available_distributions

    def latex(self):
        return '?'

class Uniform(RandomDistribution):
    """
    Random distribution object using the uniform distribution between ``min`` and ``max``.

    The returned values are floats in the range [min, max].
    """
    def __init__(self, min, max):
        """
        :param min: minimum value.
        :param max: maximum value.
        """
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a Numpy array with the given shape.
        """
        return np.random.uniform(self.min, self.max, shape)

    def latex(self):
        return "$\\mathcal{U}$(" + str(self.min) + ', ' + str(self.max) + ')'


class DiscreteUniform(RandomDistribution):
    """
    Random distribution object using the discrete uniform distribution between ``min`` and ``max``.

    The returned values are integers in the range [min, max].
    """
    def __init__(self, min, max):
        """
        :param min: minimum value.
        :param max: maximum value.
        """
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        return np.random.random_integers(self.min, self.max, shape)

    def latex(self):
        return "$\\mathcal{U}$(" + str(self.min) + ', ' + str(self.max) + ')'


class Normal(RandomDistribution):
    """
    Random distribution instance returning a random value based on a normal (Gaussian) distribution.
    """
    def __init__(self, mu, sigma, min=None, max=None):
        """
        :param mu: mean of the distribution.
        :param sigma: standard deviation of the distribution.
        :param min: minimum value (default: unlimited).
        :param max: maximum value (default: unlimited).
        """
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        data = np.random.normal(self.mu, self.sigma, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\mathcal{N}$(" + str(self.mu) + ', ' + str(self.sigma) + ')'

class LogNormal(RandomDistribution):
    """
    Random distribution instance returning a random value based on lognormal distribution.
    """
    def __init__(self, mu, sigma, min=None, max=None):
        """
        :param mu: mean of the distribution.
        :param sigma: standard deviation of the distribution.
        :param min: minimum value (default: unlimited).
        :param max: maximum value (default: unlimited).
        """
        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        data = np.random.lognormal(self.mu, self.sigma, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\ln\\mathcal{N}$(" + str(self.mu) + ', ' + str(self.sigma) + ')'

class Exponential(RandomDistribution):
    """
    Random distribution instance returning a random value based on exponential distribution, according the density function:

    $$P(x | \\lambda) = \\lambda e^{(-\\lambda x )}$$

    """
    def __init__(self, Lambda, min=None, max=None):
        """
        **Note:** ``Lambda`` is capitalized, otherwise it would be a reserved Python keyword.

        :param Lambda: rate parameter.
        :param min: minimum value (default: unlimited).
        :param max: maximum value (default: unlimited).

        """
        self.Lambda = Lambda
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape.
        """
        data = np.random.exponential(self.Lambda, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\exp$(" + str(self.Lambda) + ')'

class Gamma(RandomDistribution):
    """
    Random distribution instance returning a random value based on gamma distribution.
    """
    def __init__(self, alpha, beta=1.0, seed=-1, min=None, max=None):
        """
        :param alpha: shape of the gamma distribution
        :param beta: scale of the gamma distribution
        :param min: minimum value returned (default: unlimited).
        :param max: maximum value returned (default: unlimited).
        """
        self.alpha = alpha
        self.beta = beta
        self.min = min
        self.max = max

    def get_values(self, shape):
        """
        Returns a np.ndarray with the given shape
        """
        data = np.random.gamma(self.alpha, self.beta, shape)
        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\Gamma$(" + str(self.alpha) + ', ' + str(self.beta) + ')'
