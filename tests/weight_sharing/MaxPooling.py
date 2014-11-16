from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws)")

# Populations
In = Population((10, 10, 2), Input)
Out = Population((2, 2, 2), Output)


# Max-pooling
proj = SharedProjection(
    pre = In, 
    post = Out, 
    target = 'ws',
    operation = 'max'
).pooling()

# Compile
compile()

# Set input
In[2, 2, 0].r = 1.0
In[7, 7, 1].r = 1.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(2,2,1)
imshow(In.r[:, :, 0], cmap = cm.gray, interpolation='nearest')
subplot(2,2,2)
imshow(Out.r[:, :, 0], cmap = cm.gray, interpolation='nearest')
subplot(2,2,3)
imshow(In.r[:, :, 1], cmap = cm.gray, interpolation='nearest')
subplot(2,2,4)
imshow(Out.r[:, :, 1], cmap = cm.gray, interpolation='nearest')
show()