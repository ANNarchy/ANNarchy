from ANNarchy4 import *

Izhikevitch = SpikeNeuron(
parameters="""
    I_in = 0.0
    noise_scale = 5.0 : population
    a = 0.02 : population
    b = 0.2 : population
    c = -65.0 : population
    d = 2.0 : population
""",
equations="""
    noise = Normal(0.0,1.0)
    I = sum(exc) + sum(inh) + noise * noise_scale : init = 0.0
    dv/dt = 0.04 * v * v + 5*v + 140 -u + I
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

Small = Population((4, 4), Izhikevitch)
Middle = Population((2,2), Izhikevitch)

testAll2AllSpike = Projection( 
    pre = Small, 
    post = Middle, 
    target = 'exc', 
    connector = all2all(pre = Small, post = Middle, weights=1.0 )
)

compile()

simulate(10)

print 'done'
raw_input()