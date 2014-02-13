from ANNarchy4 import *

from pylab import show, figure, subplot, legend, close

#setup(verbose=True)

Izhikevitch = SpikeNeuron(
parameters="""
    I_in = 0.0
    noise_scale = 0.0 : population
    a = 0.02 : population
    b = 0.2 : population
    c = -65.0 : population
    d = 2.0 : population
    tau = 10.0 : population
""",
equations="""
    noise = Normal(0.0,1.0)
    I = g_exc + noise * noise_scale : init = 0.0
    tau * dg_exc / dt = -g_exc
    dv/dt = 0.04 * v * v + 5*v + 140 -u + I : init = -65.0
    du/dt = a * (b*v - u) : init = 0.2
""",
spike = """
    v >= 30.0
""",
reset = """
    v = c
    u = u+d
"""
)

Simple =SpikeSynapse(
pre_spike="""
    g_target += 1.0
"""              
)
 
SimpleLearn=SpikeSynapse(
parameters="""
    tau_pre = 10 : population
    tau_post = 1 : population
    cApre = 1 : population
    cApost = 0 : population
""",
equations = """
    tau_pre * dApre/dt = -Apre
    tau_post * dApost/dt = -Apost
""",
pre_spike="""
    Apre += cApre
    g_target += value
    value += Apost
"""#,              
#post_spike="""
#    Apost += cApost
#    value += Apre
#"""      
)

Small = Population(5, Izhikevitch)
Small.noise_scale = 5.0
Middle = Population(1, Izhikevitch)
Middle.noise_scale = 0.0

testAll2AllSpike = Projection( 
    pre = Small, 
    post = Middle, 
    target = 'exc',
    synapse = Simple,
    connector = all2all(pre = Small, post = Middle, weights=Uniform(0,1) )
)

compile()

to_record = [{'pop': Small, 'var': 'v'}, 
             {'pop': Small, 'var': 'g_exc'},
             {'pop': Middle, 'var': 'v'}, 
             {'pop': Middle, 'var': 'g_exc'} ]

record ( to_record )

#testAll2AllSpike.dendrite(0).start_record('Apre')

for i in range(1000):
    simulate(1)

data = get_record( to_record )

close('all')

#===============================================================================
# #
# #plot pre neurons
# for i in range(Small.size):
#     fig = figure()
#     fig.suptitle(Small.name+', neuron '+str(i))
#      
#     ax = subplot(211)
#      
#     ax.plot( data['Population0']['v']['data'][i,:], label = "membrane potential")
#     ax.legend(loc=2)
#      
#     ax = subplot(212)
#      
#     ax.plot( data['Population0']['g_exc']['data'][i,:], label = "g_exc")
#     ax.legend(loc=2)
#===============================================================================
 
#
# plot post neurons
for i in range(Middle.size):
    fig = figure()
    fig.suptitle(Middle.name+', neuron '+str(i))
     
    ax = subplot(211)
     
    ax.plot( data['Population1']['v']['data'][i,:], label = "membrane potential")
    ax.legend(loc=2)
     
    ax = subplot(212)
     
    ax.plot( data['Population1']['g_exc']['data'][i,:], label = "g_exc")
    ax.legend(loc=2)
 
show()

print 'done'
raw_input()