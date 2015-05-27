from ANNarchy import *
import time

p1 = Population( 20000, LeakyIntegrator(noise="Normal(0.0, 1.0)") )

compile()

simulate(1)

acc_time = 0.0
NTrials = 100

print 'all element access'

tmp = []
for i in range(NTrials):
    t1 = time.clock()
    tmp = p1.r
    t2 = time.clock()
    acc_time += t2 - t1

#print tmp, type(tmp)
print 'old:',(acc_time / NTrials) * 1000.0, 'ms'

acc_time = 0.0
acc_time2 = 0.0
for i in range(NTrials):
    t1 = time.clock()
    tmp = p1.cyInstance.get_r2()
    t2 = time.clock()
    acc_time += t2 - t1
    t2 = time.clock()
    tmp = np.array(tmp)
    t3 = time.clock()
    acc_time2 += t3 - t2

#print tmp, type(tmp)
print 'new:', (acc_time / NTrials) * 1000.0, 'ms'
print 'conv:', (acc_time2 / NTrials) * 1000.0, 'ms'

print 'single element access'
acc_time = 0.0
tmp = []
for i in range(NTrials):
    t1 = time.clock()
    tmp = p1.cyInstance.get_single_r(10)
    t2 = time.clock()
    acc_time += t2 - t1

print tmp, type(tmp)
print 'old:', (acc_time / NTrials) * 1000.0 * 1000.0, 'us'

acc_time = 0.0
for i in range(NTrials):
    t1 = time.clock()
    tmp = p1.cyInstance.get_single_r2(10)
    t2 = time.clock()

print tmp, type(tmp)
print 'new:', (acc_time / NTrials) * 1000.0 * 1000.0, 'us'

