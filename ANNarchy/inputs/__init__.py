# SpecificPopulation inheritances
from .InputArray import InputArray
from .PoissonPopulation import PoissonPopulation
from .SpikeSourceArray import SpikeSourceArray
from .SpikeTrains import HomogeneousCorrelatedSpikeTrains
from .TimedArray import TimedArray
from .TimedPoissonPopulation import TimedPoissonPopulation

# SpecificProjecion inheritances
from .CurrentInjection import CurrentInjection
from .DecodingProjection import DecodingProjection

# For reporting
input_type_list = (
    InputArray,
    PoissonPopulation, 
    SpikeSourceArray, 
    TimedArray,
    HomogeneousCorrelatedSpikeTrains, 
    TimedPoissonPopulation,
    DecodingProjection, 
    CurrentInjection
)
input_name_list = [
    "Input Array", 
    "Poisson distribution", 
    "Spike source", 
    "Timed Array",
    "Homogeneous correlated spike trains",
    "Scheduled Poisson distribution", 
]

# Classes exported by the ANNarchy.inputs sub-module.
__all__ = [
    'InputArray',
    'PoissonPopulation', 
    'SpikeSourceArray', 
    'TimedArray',
    'HomogeneousCorrelatedSpikeTrains', 
    'TimedPoissonPopulation',
    'DecodingProjection', 
    'CurrentInjection'
]