from ANNarchy import *

setup(dt=1.0)

# Define the Izhikevich neuron
Izhikevich = SpikeNeuron(
    parameters="""
        noise_scale = 5.0 : population
        a = 0.02
        b = 0.2
        c = -65.0
        d = 2.0 
    """,
    equations="""
        noise = Normal(0.0, 1.0) * noise_scale
        I = g_exc - g_inh + noise : init = 0.0
        v += 0.04 * v * v + 5.0*v + 140.0 -u + I : init=-65.0
        u += a * (b*v - u) : init = -13.0
        g_exc = 0.0
        g_inh = 0.0 
    """,
    spike = """
        v >= 30.0
    """,
    reset = """
        v = c
        u += d
    """
)

# Create the excitatory population
Excitatory = Population(name='Excitatory', geometry=800, neuron=Izhikevich)
re = np.random.random(800)
Excitatory.c = -65.0 + 15.0*re**2
Excitatory.d = 8.0 - 6.0*re**2

# Create the Inhibitory population
Inhibitory = Population(name='Inhibitory', geometry=200, neuron=Izhikevich)
ri = np.random.random(200)
Inhibitory.noise_scale = 2.0
Inhibitory.b = 0.25 - 0.05*ri
Inhibitory.a = 0.02 + 0.08*ri
Inhibitory.u = (0.25 - 0.05*ri) * (-65.0) # b * v

# Create the projections
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

# Main loop                   
if __name__ == '__main__':
    
    # Compile
    compile()

    # Start recording the spikes in the network to produce the raster plot
    Excitatory.start_record('spike')
    Inhibitory.start_record('spike')

    # Simulate 1s   
    print 'Starting simulation' 
    from time import time
    t_start = time()
    simulate(1000.0)
    print 'Done in :', time() - t_start

    # Retrieve the spike timings
    spikes_exc = raster_plot(Excitatory.get_record('spike'))
    spikes_inh = raster_plot(Inhibitory.get_record('spike'))
    spikes = np.concatenate((spikes_exc, spikes_inh + [0, 800]), axis=0)

    # Plot the results
    import pylab as plt
    plt.plot(spikes[:, 0], spikes[:, 1], '.')
    plt.show()
