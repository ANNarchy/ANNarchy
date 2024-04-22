# SpecificPopulation inheritances
from .InputArray import InputArray
from .PoissonPopulation import PoissonPopulation
from .SpikeSourceArray import SpikeSourceArray
from .SpikeTrains import HomogeneousCorrelatedSpikeTrains
from .TimedArray import TimedArray, TimedPoissonPopulation

# SpecificProjecion inheritances
from .CurrentInjection import CurrentInjection
from .DecodingProjection import DecodingProjection

__all__ = [
    'InputArray',
    'PoissonPopulation', 'SpikeSourceArray', 'TimedArray',
    'HomogeneousCorrelatedSpikeTrains', 'TimedPoissonPopulation',
    'DecodingProjection', 'CurrentInjection'
]