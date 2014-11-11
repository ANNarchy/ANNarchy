from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws)")

# Populations
size_2D = (10, 10)
In = Population(size_2D, Input)
Out = Population(size_2D, Output)


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
).connect( weights = vertical_filter)

# Compile
compile()

# Set input
In[:, 5:].r = 1.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(1,2,1)
imshow(In.r, cmap = cm.gray, interpolation='nearest')
subplot(1,2,2)
imshow(Out.r, cmap = cm.gray, interpolation='nearest')
show()