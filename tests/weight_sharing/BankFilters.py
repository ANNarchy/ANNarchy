from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws): min=0.0")

# Populations
depth = 3
In = Population((100, 100, depth), Input)
Out = Population((50, 50, 4), Output)
Pool = Population((50, 50), Output)


# Filters
bank_filter = [
    [
        [ [-1.0]*depth, [0.0]*depth, [1.0]*depth],
        [ [-1.0]*depth, [0.0]*depth, [1.0]*depth],
        [ [-1.0]*depth, [0.0]*depth, [1.0]*depth]
    ],
    [
        [ [1.0]*depth, [0.0]*depth, [-1.0]*depth],
        [ [1.0]*depth, [0.0]*depth, [-1.0]*depth],
        [ [1.0]*depth, [0.0]*depth, [-1.0]*depth]
    ],
    [
        [ [-1.0]*depth, [-1.0]*depth, [-1.0]*depth],
        [ [ 0.0]*depth, [ 0.0]*depth, [ 0.0]*depth],
        [ [ 1.0]*depth, [ 1.0]*depth, [ 1.0]*depth]
    ],
    [
        [ [ 1.0]*depth, [ 1.0]*depth, [ 1.0]*depth],
        [ [ 0.0]*depth, [ 0.0]*depth, [ 0.0]*depth],
        [ [-1.0]*depth, [-1.0]*depth, [-1.0]*depth]
    ]
  ]

# Full connection
proj = SharedProjection(
    pre = In, 
    post = Out, 
    target = 'ws',
).convolve( weights = bank_filter, multiple=True, method='filter', padding='border')

pool = SharedProjection(
    pre = Out, 
    post = Pool, 
    target = 'ws',
    operation='max'
).pooling(extent=(1,1,4))

# Compile
compile()

# Set input
In[30:70, 30:70, :].r = 1.0

# Simulate()
simulate(10000.0, measure_time=True)

# Plot
from pylab import *
subplot(3,4,1)
imshow(In.r[:,:,0], cmap = cm.gray, interpolation='nearest')
subplot(3,4,5)
imshow(Out.r[:,:, 0], cmap = cm.gray, interpolation='nearest')
subplot(3,4,6)
imshow(Out.r[:,:, 1], cmap = cm.gray, interpolation='nearest')
subplot(3,4,7)
imshow(Out.r[:,:, 2], cmap = cm.gray, interpolation='nearest')
subplot(3,4,8)
imshow(Out.r[:,:, 3], cmap = cm.gray, interpolation='nearest')
subplot(3,4,9)
imshow(Pool.r, cmap = cm.gray, interpolation='nearest')
show()