***********************************
Refractoriness
***********************************

ANNarchy allows to model the absolute refractory period of a neuron in a flexible way. 

Defining the refractory period
-----------------------------------

The refractory period is specified by the ''refractory'' parameter of ''SpikeNeuron''. 
As any other variable it can be modified population wise.

    .. code-block :: python

        RefractoryNeuron = SpikeNeuron (
            parameters = """ ... """,
            equations = """ ... """,
            spike = """ ... """,
            reset = """ ... """,
            refractory = 5.0
        )