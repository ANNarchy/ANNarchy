from ANNarchy import *
from ANNarchy.extensions.diagonal import *

# Dummy neurons
input_neuron = Neuron(parameters="r=0.0")
sum_neuron = Neuron(equations="r=sum(exc)")

# Populations
dim_x, dim_y = 20, 20
pop1 = Population(geometry = (dim_x, dim_y), neuron=input_neuron)
pop2 = Population(geometry = (dim_x, dim_y), neuron=sum_neuron)

# Create the diagonal projection
proj = DiagonalProjection(pop1, pop2, 'exc').connect(weights = np.array([0.1, 0.5, 1.0, 0.5, 0.1]), offset=dim_x/2, slope=-1)

# Compile
compile()

# Set input
cx, cy = 5, 10
x, y = np.meshgrid(range(dim_x), range(dim_y))
pop1.r = np.exp(-((x - cx)**2+(y - cy)**2)/3.0).T

# Simulate one step
step()

# Display
from pylab import *
subplot(121)
imshow(pop1.r.T, cmap='hot', origin='lower')
title('Input')
xlabel('First dimension')
ylabel('Second dimension')
subplot(122)
imshow(pop2.r.T, cmap='hot', origin='lower')
title('Output')
xlabel('First dimension')
ylabel('Second dimension')
show()