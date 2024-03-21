"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np
from ANNarchy.core import Global

distributions_arguments = {
    'Uniform' : 2,
    'DiscreteUniform': 2,
    'Normal' : 2,
    'LogNormal': 2,
    'Exponential': 1,
    'Gamma': 2,
    'Binomial' : 2
}

distributions_equivalents = {
    'Uniform' : 'std::uniform_real_distribution< %(float_prec)s >',
    'DiscreteUniform': 'std::uniform_int_distribution<int>',
    'Normal' : 'std::normal_distribution< %(float_prec)s >',
    'LogNormal': 'std::lognormal_distribution< %(float_prec)s >',
    'Exponential': 'std::exponential_distribution< %(float_prec)s >',
    'Gamma': 'std::gamma_distribution< %(float_prec)s >',
    'Binomial': 'std::binomial_distribution<int>'
}

# List of available distributions
available_distributions = distributions_arguments.keys()

class RandomDistribution :
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
    
    def get_cpp_args(self):
        raise NotImplementedError

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

    def get_cpp_args(self):
        return self.min, self.max

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
        Returns a np.ndarray with the given shape.
        """
        # randint draws from half-open interval [min, max)
        return np.random.randint(self.min, self.max+1, shape)

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
        if sigma < 0.0:
            Global._error("Normal: the standard deviation sigma should be positive.")
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

    def get_cpp_args(self):
        return self.mu, self.sigma

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
        if sigma < 0.0:
            Global._error("LogNormal: the standard deviation sigma should be positive.")
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

    def get_cpp_args(self):
        return self.mu, self.sigma

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
        if Lambda < 0.0:
            Global._error("Exponential: the rate parameter Lambda should be positive.")
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
        
class Binomial(RandomDistribution):
    """
    Random distribution object using the binomial distribution with specified parameters, n trials and p probability of success where n an integer >= 0 and p is in the interval [0,1].

    The returned values are number of successes over the n trials.
    """

    def __init__(self, n, p):
        """
        :param n: trials.
        :param p: probability of success.
        """
        self.n = n
        self.p = p

    def get_values(self, shape):
        """
        Returns a Numpy array with the given shape.
        """
        return np.random.binomial(self.n, self.p, size=shape)

    def latex(self):
        return "$\\mathcal{B}$(" + str(self.n) + ", " + str(self.p) + ")"

    def get_cpp_args(self):
        return self.n, self.p
