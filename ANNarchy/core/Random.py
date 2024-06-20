"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import numpy as np

from ANNarchy.intern import Messages
from ANNarchy.intern.ConfigManagement import get_global_config

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
        Messages._error('instantiated base class RandomDistribution is not allowed.')
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
    Uniform distribution between ``min`` and ``max``.

    The returned values are floats in the range [min, max].

    :param min: minimum value.
    :param max: maximum value.
    :param seed: (optional) seed for the random generator. If left None, the value set in `ann.setup()` is used.
    """
    def __init__(self, min: float, max:float, seed:int=None):

        self.rng = np.random.default_rng(seed=seed if seed is not None else get_global_config('seed'))

        self.min = min
        self.max = max

    def get_values(self, shape:tuple) -> np.ndarray:
        """
        Returns a Numpy array with the given shape.

        :param shape: Shape of the array.
        :returns: Array.
        """
        return self.rng.uniform(self.min, self.max, shape)

    def latex(self):
        return "$\\mathcal{U}$(" + str(self.min) + ', ' + str(self.max) + ')'

    def get_cpp_args(self):
        return self.min, self.max

class DiscreteUniform(RandomDistribution):
    """
    Discrete uniform distribution between ``min`` and ``max``.

    The returned values are integers in the range [min, max].

    :param min: minimum value.
    :param max: maximum value.
    :param seed: (optional) seed for the random generator. If left None, the value set in `ann.setup()` is used.
    """
    def __init__(self, min: int, max:int, seed:int=None):
        self.rng = np.random.default_rng(seed=seed if seed is not None else get_global_config('seed'))

        self.min = min
        self.max = max

    def get_values(self, shape:tuple) -> np.ndarray:
        """
        Returns a Numpy array with the given shape.

        :param shape: Shape of the array.
        :returns: Array.
        """
        # randint draws from half-open interval [min, max)
        return self.rng.integers(self.min, self.max+1, shape)

    def latex(self):
        return "$\\mathcal{U}$(" + str(self.min) + ', ' + str(self.max) + ')'


class Normal(RandomDistribution):
    """
    Normal distribution.

    :param mu: Mean of the distribution.
    :param sigma: Standard deviation of the distribution.
    :param min: Minimum value (default: unlimited).
    :param max: Maximum value (default: unlimited).
    :param seed: (optional) seed for the random generator. If left None, the value set in `ann.setup()` is used.
    """
    def __init__(self, mu:float, sigma:float, min:float=None, max:float=None, seed:int=None) -> None:

        if sigma < 0.0:
            Messages._error("Normal: the standard deviation sigma should be positive.")
        
         
        self.rng = np.random.default_rng(seed=seed if seed is not None else get_global_config('seed'))

        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def get_values(self, shape:tuple) -> np.ndarray:
        """
        Returns a Numpy array with the given shape.

        :param shape: Shape of the array.
        :returns: Array.
        """
        data = self.rng.normal(self.mu, self.sigma, shape)

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
    Log-normal distribution.

    :param mu: Mean of the distribution.
    :param sigma: Standard deviation of the distribution.
    :param min: Minimum value (default: unlimited).
    :param max: Maximum value (default: unlimited).
    :param seed: (optional) seed for the random generator. If left None, the value set in `ann.setup()` is used.
    """
    def __init__(self, mu:float, sigma:float, min:float=None, max:float=None, seed:int=None):

        if sigma < 0.0:
            Messages._error("LogNormal: the standard deviation sigma should be positive.")

        self.rng = np.random.default_rng(seed=seed if seed is not None else get_global_config('seed'))

        self.mu = mu
        self.sigma = sigma
        self.min = min
        self.max = max

    def get_values(self, shape:tuple) -> np.ndarray:
        """
        Returns a Numpy array with the given shape.

        :param shape: Shape of the array.
        :returns: Array.
        """
        data = self.rng.lognormal(self.mu, self.sigma, shape)

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
    Exponential distribution, according to the density function:

    $$P(x | \\lambda) = \\lambda e^{(-\\lambda x )}$$

    **Note:** ``Lambda`` is capitalized, otherwise it would be a reserved Python keyword.

    :param Lambda: rate parameter.
    :param min: minimum value (default: unlimited).
    :param max: maximum value (default: unlimited).
    :param seed: (optional) seed for the random generator. If left None, the value set in `ann.setup()` is used.
    """
    def __init__(self, Lambda:float, min:float=None, max:float=None, seed:int=None):

        if Lambda < 0.0:
            Messages._error("Exponential: the rate parameter Lambda should be positive.")

        self.rng = np.random.default_rng(seed=seed if seed is not None else get_global_config('seed'))
        
        self.Lambda = Lambda
        self.min = min
        self.max = max

    def get_values(self, shape:tuple) -> np.ndarray:
        """
        Returns a Numpy array with the given shape.

        :param shape: Shape of the array.
        :returns: Array.
        """
        data = self.rng.exponential(self.Lambda, shape)

        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\exp$(" + str(self.Lambda) + ')'

class Gamma(RandomDistribution):
    """
    Gamma distribution.

    :param alpha: Shape of the gamma distribution.
    :param beta: Scale of the gamma distribution.
    :param min: Minimum value returned (default: unlimited).
    :param max: Maximum value returned (default: unlimited).
    :param seed: (optional) seed for the random generator. If left None, the value set in `ann.setup()` is used.
    """
    def __init__(self, alpha:float, beta:float=1.0, min:float=None, max:float=None, seed:int=None):

        self.rng = np.random.default_rng(seed=seed if seed is not None else get_global_config('seed'))

        self.alpha = alpha
        self.beta = beta
        self.min = min
        self.max = max

    def get_values(self, shape:tuple) -> np.ndarray:
        """
        Returns a Numpy array with the given shape.

        :param shape: Shape of the array.
        :returns: Array.
        """
        data = self.rng.gamma(self.alpha, self.beta, shape)

        if self.min != None:
            data[data<self.min] = self.min
        if self.max != None:
            data[data>self.max] = self.max
        return data

    def latex(self):
        return "$\\Gamma$(" + str(self.alpha) + ', ' + str(self.beta) + ')'
        
class Binomial(RandomDistribution):
    """
    Binomial distribution.
    
    Parameters: n trials and p probability of success where n an integer >= 0 and p is in the interval [0,1].

    The returned values are the number of successes over the n trials.

    :param n: Number of trials.
    :param p: Probability of success.
    :param seed: (optional) seed for the random generator. If left None, the value set in `ann.setup()` is used.
    """

    def __init__(self, n:int, p:float, seed:int=None):

        self.rng = np.random.default_rng(seed=seed if seed is not None else get_global_config('seed'))

        self.n = n
        self.p = p

    def get_values(self, shape:tuple) -> np.ndarray:
        """
        Returns a Numpy array with the given shape.

        :param shape: Shape of the array.
        :returns: Array.
        """
        return self.rng.binomial(self.n, self.p, size=shape)

    def latex(self):
        return "$\\mathcal{B}$(" + str(self.n) + ", " + str(self.p) + ")"

    def get_cpp_args(self):
        return self.n, self.p
