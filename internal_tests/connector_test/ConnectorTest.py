from ANNarchy4 import *

Simple = RateNeuron(   
parameters="""
    tau = 1.0
    baseline = 0.0
""",
equations="""
    noise = Uniform(0,1)
    tau * dmp / dt + mp = baseline + noise
    rate = pos(mp)
""" 
)

Simple2 = RateSynapse(
psp = """ value * (1.0-pre.rate) """                      
)

In = Population((20, 20), Simple)
Middle = Population((20, 20), Simple)
Out = Population((4, 4), Simple)

testOne2One = Projection( 
    pre = In, 
    post = Middle, 
    target = 'exc',
).connect_with_func(method=one2one, weights=1.0)

testAll2All = Projection( 
    pre = In, 
    post = Out, 
    target = 'exc',
    method = all2all,
    synapse = Simple2,
    weights = 1.0,
    delays = 0.0
)

testGaussian = Projection(
    pre = In, 
    post = Out, 
    target = 'inh',
    method = gaussian,
    sigma=0.3,
    amp=0.1 
)
 
testDog = Projection(
    pre = In, 
    post = Out, 
    target = 'inh', 
    method = dog,
    amp_pos=0.2, 
    sigma_pos=0.2, 
    amp_neg=0.1, 
    sigma_neg=0.3
)

compile()

print testOne2One
for dendrite in testOne2One:
    print dendrite.cy_instance.post_rank
    print dendrite.cy_instance.rank

#visOne2One = Visualization( [ { 'proj': testOne2One, 'var': 'value', 'min': 0.0, 'max': 1.0, 'title': 'weights one2one'} ] )
#visOne2One.render()
 
#visAll2All = Visualization( [ { 'proj': testAll2All, 'var': 'value', 'min': 0.0, 'max': 1.0, 'title': 'weights all2all'} ] )
#visAll2All.render()
 
#visGaussian = Visualization( [ { 'proj': testGaussian, 'var': 'value', 'min': 0.0, 'max': 0.1, 'title': 'weights gaussian'} ] )
#visGaussian.render()
 
#visDog = Visualization( [ { 'proj': testDog, 'var': 'value', 'min': 0.0, 'max': 0.1, 'title': 'weights difference of gaussian'} ] )
#visDog.render()

raw_input()
