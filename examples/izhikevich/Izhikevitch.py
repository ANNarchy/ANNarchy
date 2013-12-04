from ANNarchy4 import *

# Define the neurons
Izhikevitch = Neuron(
    I = Variable(init=0.0),
    
    u = Variable(init=-65.*0.2, eq="u = a * (b*mp - u)"), # init should be b*baseline
    v = SpikeVariable(eq="dv/dt = 0.04 * v * v + 5*v + 140 -u +I", threshold=30.0, init=-65, reset=['mp = c', 'u = u+d']),
    order = ['v', 'u']
)

Pop = Population(name='Pop', geometry=(1,1), neuron=Izhikevitch)

# Compile
compile()

if __name__ == '__main__':
    # Run the simulation
    to_record = [
        { 'pop': Excitatory, 'var': 'u' }, 
        { 'pop': Excitatory, 'var': 'mp' }
    ]
    
    record( to_record )
    for i in xrange(200):
        simulate(1)
        
    data = get_record( to_record )
    
    neur_1 = data['Excitatory']['mp']['data'][1,1,:]
    
    plot( neur_1 )
    
    show()
    
    raw_input('Press a key to continue ...')