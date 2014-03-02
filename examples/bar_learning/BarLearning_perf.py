#
#   BarLearning example for ANNarchy4
#
#   authors: Julien Vitay, Helge Uelo Dinkelbach
#
from ANNarchy4 import *

setup()

# Defining the neurons
InputNeuron = RateNeuron(
    parameters=""" 
        tau = 10.0 : population
        baseline = 0.0 
    """,
    equations="""
        tau * drate/dt + rate = baseline : min=0.0
    """
)

LeakyNeuron = RateNeuron(
    parameters=""" 
        tau = 10.0 : population
        baseline = 0.0 
    """,
    equations="""
        tau * drate/dt + rate = sum(exc) - sum(inh) : min=0.0
    """
)

# Defining the synapses
Oja = RateSynapse(
    parameters=""" 
        tau = 2000 : postsynaptic
        alpha = 8.0 : postsynaptic
    """,
    equations="""
        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value
    """
)  

AntiHebb = RateSynapse(
    parameters=""" 
        tau = 2000 : postsynaptic
        alpha = 0.3 : postsynaptic
    """,
    equations="""
        tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value : min = 0.0
    """
)  

# Creating the populations
nb_neurons = 32  
input_pop = Population(geometry=(nb_neurons, nb_neurons), neuron=InputNeuron)
feature_pop = Population(geometry=(nb_neurons, 4), neuron=LeakyNeuron)

# Creating the projections
input_feature = Projection(
    pre=input_pop, 
    post=feature_pop, 
    target='exc', 
    synapse = Oja
).connect_all_to_all( weights = Uniform(-0.5, 0.5) )
                     
feature_feature = Projection(
    pre = feature_pop, 
    post = feature_pop, 
    target = 'inh',
    synapse = AntiHebb
).connect_all_to_all( weights = Uniform(0, 1.0) )

# Definition of the environment
def set_input():
    # Choose which bars will be used as inputs
    values = np.zeros((nb_neurons, nb_neurons))
    for w in range(nb_neurons):
        if np.random.random() < 1./ float(nb_neurons):
            values[:, w] = 1.
    for h in range(nb_neurons):
        if np.random.random() < 1./ float(nb_neurons):
            values[h, :] = 1.

    # Set the input
    input_pop.baseline = values.reshape(nb_neurons**2)

# visualization meanwhile yes/no
vis_during_sim=True

if __name__=='__main__':

    # Compiling the network
    compile(debug_build=False, cpp_stand_alone=False, profile_enabled=True)

    # Collect visualizing information
    plot1 = {'pop': input_pop, 'var': 'rate'}
    plot2 = {'pop': feature_pop, 'var': 'rate'}
    plot3 = {'proj': input_feature, 'var': 'value', 
         'max': 0.1, 'title': 'Receptive fields'}

    # Setup visualizer
    vis = Visualization( [plot1, plot2, plot3 ] )

    save('init.mat')
    
    #
    # setup the test
    num_trials = 5
    #thread_count = [ x+1 for x in range(6) ]
    thread_count = [ 4-x for x in range(4) ]
    trial_dur = 50
    
    # profile instance
    profiler = Profile(thread_count, num_trials)
    profiler.add_to_profile("network")
    profiler.add_to_profile("Population0")
    profiler.add_to_profile("Population1")
    
    #
    # pre setup
    diff_runs = []
    index = [ x for x in xrange(num_trials) ]

    for test in range(len(thread_count)):
        diff_runs.append( [0 for x in range(num_trials) ] )

        profiler.set_num_threads(thread_count[test])   

        # Run the simulation        
        for trial in range(num_trials):
            set_input()
            simulate(trial_dur) 

            if vis_during_sim:
               vis.render()
    
            profiler.measure(thread_count[test], trial, trial*trial_dur, (trial+1)*trial_dur)

        profiler.reset_timer()

        # Visualize the result of learning
        vis.render()  

        print 'simulation finished.'

    print 'all simulation finished.'

    profiler.analyse_data()
        
    raw_input()
