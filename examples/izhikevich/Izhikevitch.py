from ANNarchy4 import *
from pylab import plot, bar, show, figure, subplot

setup(dt=1)

# Define the neurons
Izhikevitch = Neuron(
    I = Variable(init=0.0),
    a = 0.02,
    b = 0.2,
    c = -65.0,
    d = 2,
    u = Variable(init=-65.*0.2, eq="u = a * (b*v - u)"), # init should be b*baseline
    v = SpikeVariable(eq="dv/dt +v = (0.04 * v * v + 5*v + 140 -u +I)", threshold=30.0, init=-65, reset=['v = c', 'u = u+d']),
    order = ['v', 'u']
)

Pop = Population(name='Pop', geometry=(1,1), neuron=Izhikevitch)

# Compile
compile()

if __name__ == '__main__':
    # Run the simulation
    to_record = [
        { 'pop': Pop, 'var': 'u' }, 
        { 'pop': Pop, 'var': 'v' }
    ]
    
    nb_steps = 10
    
    # 20-100ms is an input
    I = np.zeros((nb_steps,1))
    if nb_steps > 20:
        I[20:nb_steps,:] = 1.
    
    record( to_record )
    for i in xrange(nb_steps):
        
        # first 20 ms no input
        Pop.I = I[i,:]    
        simulate(1)
        
    data = get_record( to_record )
    
    neur_v_1 =  data['Pop']['v']['data'][0,0,:]
    neur_u_1 =  data['Pop']['u']['data'][0,0,:]    
    input_1 = I[:,0]
    spike_time_1 = np.zeros((nb_steps,1))
    spike_time_1[Pop.cyInstance.get_spike_timings()[0,:]] = 1
    X = np.arange(nb_steps)
    
    figure()
    print neur_v_1
    print neur_u_1
    subplot(3,1,1)
    plot( neur_v_1 )
    plot( neur_u_1 )
    
    subplot(3,1,2)
    plot( input_1 )

    subplot(3,1,3)
    bar( X, spike_time_1 )
    
    show()