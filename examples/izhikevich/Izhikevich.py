from ANNarchy import *

# Define the neurons
Izhikevich = SpikeNeuron(
    parameters="""
        noise_scale = 5.0 : population
        a = 0.02
        b = 0.2
        c = -65.0
        d = 2.0 
        tau = 5.0: population
    """,
    equations="""
        noise = Normal(0.0, 1.0) * noise_scale
        I = g_exc - g_inh + noise : init = 0.0
        v += 0.04 * v * v + 5.0*v + 140.0 -u + I : init=-65.0
        u += a * (b*v - u) : init = -13.0
        g_exc = 0.0
        g_inh = 0.0 
        s = 0.0
    """,
    spike = """
        v >= 30.0
    """,
    reset = """
        v = c
        u += d
        s = 1.0
    """
)

Excitatory = Population(name='Excitatory', geometry=800, neuron=Izhikevich)
re = np.random.random(800)
Excitatory.c = -65.0 + 15.0*re**2
Excitatory.d = 8.0 - 6.0*re**2

Inhibitory = Population(name='Inhibitory', geometry=200, neuron=Izhikevich)
ri = np.random.random(200)
Inhibitory.noise_scale = 2.0
Inhibitory.b = 0.25 - 0.05*ri
Inhibitory.a = 0.02 + 0.08*ri
Inhibitory.u = (0.25 - 0.05*ri) * (-65.0) # b * v

exc_exc = Projection(
    pre=Excitatory, 
    post=Excitatory, 
    target='exc'
).connect_all_to_all(weights=Uniform(0.0, 0.5))
   
exc_inh = Projection(
    pre=Excitatory, 
    post=Inhibitory, 
    target='exc',
).connect_all_to_all(weights=Uniform(0.0, 0.5))
  
inh_exc = Projection(
    pre=Inhibitory, 
    post=Excitatory, 
    target='inh'
).connect_all_to_all(weights=Uniform(0.0, 1.0))
  
inh_inh = Projection(
    pre=Inhibitory, 
    post=Inhibitory, 
    target='inh'
).connect_all_to_all(weights=Uniform(0.0, 1.0))

                   
if __name__ == '__main__':
    
    # Compile
    compile()

    # Simulate 1s   
    print 'Starting simulation' 
    simulate(1000.0)
    print 'Done'

    # Retrieve the spike timings
    spikes_exc = Excitatory.raster_plot()
    spikes_inh = Inhibitory.raster_plot()
    spikes = np.concatenate((spikes_exc, spikes_inh + [0, 800]), axis=0)

    #Plot the results
    try:
        import pyqtgraph as pg
    except:
        print 'PyQtGraph is not installed, can not visualize the simulation.'
        exit(0)
    else:
        pg.plot(spikes[:, 0], spikes[:, 1], pen=None, symbol='o', symbolSize=3)
        raw_input()
