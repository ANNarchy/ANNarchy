from ANNarchy4 import *

Input = RateNeuron(   
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

Small1 = Population((10, 10), Input)
Small2 = Population((10, 10), Input)
Middle = Population((20, 20), Input)

testOne2One = Projection( 
    pre = Small1, 
    post = Small2, 
    target = 'exc', 
    connector = one2one(pre = Small1, post = Small2, weights=1.0 )
)

testAll2All = Projection( 
    pre = Small2, 
    post = Small2, 
    target = 'exc', 
    connector = all2all(pre = Small2, post = Small2, weights=1.0 )
)

testGaussian = Projection(
    pre = Small2, 
    post = Small2, 
    target = 'inh', 
    connector = gaussian(
                   pre = Small2,
                   post = Small2, 
                   sigma=0.3,
                   amp=0.1, 
                ) 
)

testDog = Projection(
    pre = Small2, 
    post = Small2, 
    target = 'inh', 
    connector = dog(
                   pre = Small2,
                   post = Small2, 
                   amp_pos=0.2, 
                   sigma_pos=0.2, 
                   amp_neg=0.1, 
                   sigma_neg=0.3
                ) 
)

compile()

visOne2One = Visualization( [ { 'proj': testOne2One, 'var': 'value', 'min': 0.0, 'max': 1.0, 'title': 'weights one2one'} ] )
visOne2One.render()

visAll2All = Visualization( [ { 'proj': testAll2All, 'var': 'value', 'min': 0.0, 'max': 1.0, 'title': 'weights all2all'} ] )
visAll2All.render()

visGaussian = Visualization( [ { 'proj': testGaussian, 'var': 'value', 'min': 0.0, 'max': 0.1, 'title': 'weights gaussian'} ] )
visGaussian.render()

visDog = Visualization( [ { 'proj': testDog, 'var': 'value', 'min': 0.0, 'max': 0.1, 'title': 'weights difference of gaussian'} ] )
visDog.render()

raw_input()