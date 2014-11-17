from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws): min=0.0")

# Populations
depth = 3
In = Population((10, 10, 3), Input)
Out = Population((5, 5, 2), Output)


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
    ]
  ]

# Full connection
proj = SharedProjection(
    pre = In, 
    post = Out, 
    target = 'ws',
).convolve( weights = bank_filter, filter_or_kernel=False, padding='border')

# Compile
compile()

# Set input
In[:, 3:, :].r = 1.0
In[:, 7:, :].r = 0.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(1,3,1)
imshow(In.r[:,:,0], cmap = cm.gray, interpolation='nearest')
subplot(1,3,2)
imshow(Out.r[:,:, 0], cmap = cm.gray, interpolation='nearest')
subplot(1,3,3)
imshow(Out.r[:,:, 1], cmap = cm.gray, interpolation='nearest')
show()