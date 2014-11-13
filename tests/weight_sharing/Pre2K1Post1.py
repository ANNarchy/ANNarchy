from ANNarchy import *
from ANNarchy.extensions.weightsharing import SharedProjection

# Dummy neurons
Input = Neuron(parameters="r = 0.0")
Output = Neuron(equations="r = sum(ws)")

# Populations
size_2D = (10, 10)
size_1D = (1, 5) # <- trick to make it work!
In = Population(size_2D, Input)
Out = Population(size_1D, Output)


# Filters
vertical_filter = np.array(
    [[-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
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
for i in range(10):
    In[i, i:].r = 1.0

# Simulate()
simulate(1.0)

# Plot
from pylab import *
subplot(1,2,1)
imshow(In.r, cmap = cm.gray, interpolation='nearest')
subplot(1,2,2)
plot(Out.r[0,:])
show()