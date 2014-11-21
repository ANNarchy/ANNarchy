from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws)")

# Populations
In = Population((10, 10), Input)
Out = Population((5, 10), Output)


# Filters
vertical_filter = np.array(
    [ -1.0, 0.0, 1.0]
)

# Full connection
proj = SharedProjection(
    pre = In, 
    post = Out, 
    target = 'ws'
).convolve( weights = vertical_filter, keep_last_dimension=True, method='filter', padding='border')

# Compile
compile()

# Set input
for i in range(10):
    In[i:, i].r = 1.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(1,2,1)
imshow(In.r, cmap = cm.gray, interpolation='nearest')
subplot(1,2,2)
imshow(Out.r, cmap = cm.gray, interpolation='nearest')
show()