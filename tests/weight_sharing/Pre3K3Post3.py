from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws)")

# Populations
depth = 3
size_3D = (10, 10, 10)
In = Population(size_3D, Input)
Out = Population(size_3D, Output)
smallOut = Population((5, 5, 5), Output)


# Filters
vertical_filter = np.array(
    [
        [ [-1.0]*depth, [0.0]*depth, [1.0]*depth],
        [ [-1.0]*depth, [0.0]*depth, [1.0]*depth],
        [ [-1.0]*depth, [0.0]*depth, [1.0]*depth]
    ]
)

# Full connection
proj = SharedProjection(
    pre = In, 
    post = Out, 
    target = 'ws',
).convolve( weights = vertical_filter, method='filter', padding='border')

proj = SharedProjection(
    pre = In, 
    post = smallOut, 
    target = 'ws',
).convolve( weights = vertical_filter, method='filter', padding='border')

# Compile
compile()

# Set input
In[5:, 5:, :].r = 1.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(1,3,1)
imshow(In.r[:,:,1], cmap = cm.gray, interpolation='nearest')
subplot(1,3,2)
imshow(Out.r[:,:,1], cmap = cm.gray, interpolation='nearest')
subplot(1,3,3)
imshow(smallOut.r[:,:,1], cmap = cm.gray, interpolation='nearest')
show()