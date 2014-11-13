from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws)")

# Populations
In = Population((20, 10, 5), Input)
Out = Population((10, 5, 5), Output)


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

# Compile
compile()

# Set input
for i in range(5):
    In[:, 2*i:, i].r = 1.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(3,2,1)
imshow(In.r[:,:,1], cmap = cm.gray, interpolation='nearest')
subplot(3,2,2)
imshow(Out.r[:,:,1], cmap = cm.gray, interpolation='nearest')
subplot(3,2,3)
imshow(In.r[:,:,2], cmap = cm.gray, interpolation='nearest')
subplot(3,2,4)
imshow(Out.r[:,:,2], cmap = cm.gray, interpolation='nearest')
subplot(3,2,5)
imshow(In.r[:,:,3], cmap = cm.gray, interpolation='nearest')
subplot(3,2,6)
imshow(Out.r[:,:,3], cmap = cm.gray, interpolation='nearest')
show()