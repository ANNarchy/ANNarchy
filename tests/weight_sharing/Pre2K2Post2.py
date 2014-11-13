from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws)")

# Populations
size_2D = (10, 10)
In = Population(size_2D, Input)
Out = Population(size_2D, Output)
smallOut = Population((5, 5), Output)


# Filters
vertical_filter = np.array(
    [
        [ -1.0, 0.0, 1.0],
        [ -1.0, 0.0, 1.0],
        [ -1.0, 0.0, 1.0]
    ]
)

# Full connection
proj = SharedProjection(
    pre = In, 
    post = Out, 
    target = 'ws',
).convolve( weights = vertical_filter, filter_or_kernel=False, padding='border')

proj = SharedProjection(
    pre = In, 
    post = smallOut, 
    target = 'ws',
).convolve( weights = vertical_filter, filter_or_kernel=False, padding='border')

# Compile
compile()

# Set input
In[:, 5:].r = 1.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(1,3,1)
imshow(In.r, cmap = cm.gray, interpolation='nearest')
subplot(1,3,2)
imshow(Out.r, cmap = cm.gray, interpolation='nearest')
subplot(1,3,3)
imshow(smallOut.r, cmap = cm.gray, interpolation='nearest')
show()