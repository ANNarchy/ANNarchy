***********************************
Vogels and Abbott benchmark
***********************************

The scripts ``COBA.py`` and ``CUBA.py``  in ``examples/vogels_abbott`` reproduces the two first benchmarks used in:

    **Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al.** (2007), Simulation of networks of spiking neurons: a review of tools and strategies., *J. Comput. Neurosci., 23, 3, 349–98*

Both are based on the balanced network proposed by: 

    **Vogels, T. P. and Abbott, L. F.** (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., *J. Neurosci., 25, 46, 10786–95*

It is composed of 4000 neurons (3200 excitatory and 800 inhibitory), reciprocally connected with a probability of 0.02 (sparse connection).

The CUBA network uses a current-based integrate-and-fire neuron model:

.. math::

    \tau \cdot \frac{dv (t)}{dt} = E_l - v(t) + g_\text{exc} (t) - g_\text{inh} (t)

while the COBA model uses conductance-based IF neurons:

.. math::

    \tau \cdot \frac{dv (t)}{dt} = E_l - v(t) + g_\text{exc} (t) * (E_\text{exc}) - v(t)) + g_\text{inh} (t) * (E_\text{inh}) - v(t)) + I(t)
    
Apart from the neuron model and synaptic weights, both networks are equal, so we'll focus on the COBA network here.

First, ANNarchy is imported and the discretization step `dt` is set to 0.1 ms::

    from ANNarchy import * 
    dt = 0.1
    setup(dt=dt) 

The COBA neuron model is then defined::

    COBA = Neuron(
        parameters="""
            El = -60.0          : population
            Vr = -60.0          : population
            Erev_exc = 0.0      : population
            Erev_inh = -80.0    : population
            Vt = -50.0          : population
            tau = 20.0          : population
            tau_exc = 5.0       : population
            tau_inh = 10.0      : population
            I = 20.0            : population
        """,
        equations="""
            tau * dv/dt = (El - v) + g_exc * (Erev_exc - v) + g_inh * (Erev_inh - v ) + I

            tau_exc * dg_exc/dt = - g_exc 
            tau_inh * dg_inh/dt = - g_inh 
        """,
        spike = "v > Vt",
        reset = "v = Vr",
        refractory = 5.0
    )

A population of 4000 neurons is then created and split into excitatory and inhibitory subsets::

    P = Population(geometry=4000, neuron=COBA)
    Pe = P[:3200]
    Pi = P[3200:]

The initial value of the neural variables can be set::

    P.v = Normal(-55.0, 5.0)
    P.g_exc = Normal(4.0, 1.5)
    P.g_inh = Normal(20.0, 12.0)

The sparse connection between the neurons is created using ``connect_fixed_probability``::

    Ce = Projection(pre=Pe, post=P, target='exc')
    Ce.connect_fixed_probability(weights=0.6, probability=0.02)
    Ci = Projection(pre=Pi, post=P, target='inh')
    Ci.connect_fixed_probability(weights=6.7, probability=0.02)

After compilation, the network can be simulated for one second, while the spikes are recorded::

    m = Monitor(P, ['spike'])
    simulate(1000.0, measure_time=True)
    data = m.get()

The raster plot is then easily plotted::

    t, n = m.raster_plot(data['spike'])
    print('Mean firing rate in the population: ' + str(len(t) / 4000.) + 'Hz')

    from pylab import *
    plot(t, n, '.', markersize=0.5)
    xlabel('Time (ms)')
    ylabel('# neuron')
    show()