***********************************
Hybrid networks
***********************************

ANNarchy has the possibility to simulate either rate-coded or spiking networks. It is therefore possible to define hybrid networks mixing rate-coded and spiking populations.

A typical application would be to define a rate-coded network to process visual inputs, which is used to feed a spiking network for action selection. A dummy exmaple is provided in ``examples/hybrid``.

Rate-coded to Spike
====================

Converting a rate-coded population to a spiking network is straightforward. The ``PoissonPopulation`` (see :doc:`../API/SpecificPopulation`) defines a population od spiking neurons emitting spikes following a Poisson distribution::

    pop = PoissonPopulation(1000, rates=50.)

In this case, the 1000 neurons emit spikes at a rate of 50 Hz (the rate of individual neurons can be later modified). It is possible to use a weighted sum of rate-coded synapses in order to determine the firing rate of each Poisson neuron. It requires to connect a rate-coded population to the ``PoissonPopulation`` with a given target::

    pop1 = Population(4, Neuron(parameters="r=0.0"))
    pop2 = PoissonPopulation(1000, target='exc')
    proj = Projection(pop1, pop2, 'exc')
    proj.connect_fixed_number_pre(weights=10.0, number=1)

In this example, each of the 4 pre-synaptic neurons "controls" the firing rate of one fourth (on average) of the post-synaptic ones. If ``target`` is used in the Poisson population, ``rates`` will be ignored.

The weights determine the scaling of the transmission: a presynaptic rate ``r`` of 1.0 generates a firing rate of ``w`` Hz in the post-synaptic neurons. Here setting ``pop1.r = 1.0`` will make the post-synaptic neurons fire at 10 Hz.

Spike to Rate-coded
====================

Decoding a spiking population is a harder process, because of the stochastic nature of spike trains. One can take advantage of the fact here that a rate-coded neuron usually represents an ensemble of spiking neurons, so the average firing rate in that ensemble can be more precisely decoded.

In order to do so, one needs to connect the spiking population to a rate-coded one with a many-to-one pattern using a ``DecodingProjection``. A ``DecodingProjection`` heritates all methods of ``Projection`` (including the connection methods) but performs the necessary conversion from spike trains to a instantaneous rate::

    pop1 = PoissonPopulation(1000, rates=50.0)
    pop2 = Population(1, Neuron(equations="r=sum(exc)"))
    proj = DecodingProjection(pop1, pop2, 'exc', window=10.0)
    proj.connect_all_to_all(weights=1.0)

In this example, the spiking population fires at 50 Hz. The single rate-coded neuron decoding that population will count how many spikes arrived in the last $T$ milliseconds and divide it by the total number of synapses in order to estimate the population firing rate in ``pop1``. This would be accessed in ``sum(exc)`` (or whatever target is used in the projection). Because of its simple definition, it will therefore have its rate ``r`` at 50.0 (with some variance due to the stochastic nature of spike trains).

The ``window`` argument defines the duration in milliseconds of the sliding temporal window used to estimate the firing rate. By default, it is equal to ``dt``, which means spikes are counted in a very narrow period of time, what could lead to very big variations of the decoded firing rate. If the window is too big, it would introduce a noticeable lag for the decoded firing rate if the input varies too quickly. ``window = 10.0`` is usually a good compromise, but this depends on the input firing rate.

The weights of the projection define the scaling of the decoded firing rate. If one wants a firing rate of 100 Hz to be represented by ``r=1.0``, the weights should be set to 0.01.

No ``Synapse`` model can be used in a ``DecodingProjection``.
