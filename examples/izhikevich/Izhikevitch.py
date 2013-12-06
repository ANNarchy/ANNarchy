from ANNarchy4 import *
from pylab import show, figure, subplot, legend, close

#
# experiment setup
setup(dt=1)
nb_steps = 50
nb_neurons = 1

# Define the neurons
Izhikevitch = Neuron(
    I_in = Variable(init=0.0),
    I = Variable(init=0.0, eq = "I = I_in + noise"),
    noise = Variable(eq=Uniform(0.0,5.0)),
    #noise = Variable(init=0),
    a = 0.02,
    b = 0.2,
    c = -65.0,
    d = 2.0,
    u = Variable(init=-65.*0.2, eq="du/dt = a * (b*v - u)"), # init should be b*baseline
    v = SpikeVariable(eq="dv/dt = 0.04 * v * v + 5*v + 140 -u + I", threshold=-30.0, init=-65.0, reset=['v = c', 'u = u+d']),
    order = ['I', 'v','u']
)

Pop = Population(name='Pop', geometry=(nb_neurons), neuron=Izhikevitch)
Pop2 = Population(name='Pop2', geometry=(nb_neurons), neuron=Izhikevitch)

# Compile
compile()

if __name__ == '__main__':
    # Run the simulation
    to_record = [
        { 'pop': Pop, 'var': 'u' }, 
        { 'pop': Pop, 'var': 'v' },
        { 'pop': Pop, 'var': 'I' }
    ]
    
    # 20-100ms is an input
    I = np.zeros((nb_steps,nb_neurons))
    if nb_steps > 20:
        I[50:nb_steps,:] = 10. # 10mV as input
    
    record( to_record )
    for i in xrange(nb_steps):
        
        # first 20 ms no input
        Pop.I_in = I[i,:]    
        simulate(1)
        
    data = get_record( to_record )
    
    input =  data['Pop']['I']['data']
    neur_v =  data['Pop']['v']['data']
    neur_u =  data['Pop']['u']['data']    
    
    
    X = np.arange(nb_steps)
    spikes = np.zeros((nb_neurons, nb_steps))

    #
    # reformat data for plot    
    for i in xrange(nb_neurons):
        timing = Pop.cyInstance.get_spike_timings()[i]
        
        neur_v[ i, timing ] = 30
        spikes[ i, timing ] = 1
        
    rest_pot = np.ones((nb_steps,1)) * -65.0
    threshold = np.ones((nb_steps,1)) * -30.0
        
    close('all')
    
    X = np.arange(nb_steps)
    for i in range(1):
        figure()
        ax = subplot(3,1,1)
        ax.set_title("Izhikevich, 2003")
        ax.plot( neur_v[i,:], label = "membrane potential")
        ax.plot( neur_u[i,:], label = "membrane recovery")
        ax.plot( rest_pot, label = "resting potential")
        ax.plot( threshold, label = "threshold")
        ax.legend(loc=2)
        ax.set_ylim([-70,40])
        
        ax = subplot(3,1,2)
        ax.plot( input[i,:] )
        ax.set_ylim([-70,40])
        
        ax = subplot(3,1,3)
        ax.bar( X, spikes[i,:] )
        ax.set_xlim([0,nb_steps])

    figure()
    ax = subplot(2,1,1)
    ax.imshow( spikes, cmap='hot' )
    ax.set_xlim([0,nb_steps])

    ax = subplot(2,1,2)
    ax.plot( I[:,0] )
    ax.set_xlim([0,nb_steps])
    
    show()