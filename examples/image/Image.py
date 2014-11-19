from ANNarchy import *
from ANNarchy.extensions.image import *
from ANNarchy.extensions.weightsharing import SharedProjection

Linear = Neuron(equations="r=sum(exc): min=0.0")

# Create the population    
pop = ImagePopulation(geometry=(480, 640, 3))
pooled = Population(geometry=(48, 64, 3), neuron = Linear)
filtered = Population(geometry=(48, 64, 3), neuron = Linear)

# Max pooling over the input image
pool_proj = SharedProjection(pre=pop, post=pooled, target='exc', operation='mean').pooling()

# Blue Filter
blue_filter = [ [[ [2.0, -1.0, -1.0] ]] , [[ [-1.0, 2.0, -1.0] ]] , [[ [-1.0, -1.0, 2.0] ]]  ]
blue_proj = SharedProjection(pre=pooled, post=filtered, target='exc').convolve(weights=blue_filter, method='filter', multiple=True, padding='border')

# Compile
compile()

# Set the image
pop.set_image('test.jpg')

# Simulate
simulate(3.0)

# Visualize with Matplotlib
import pylab as plt
import matplotlib.image as mpimg

fig = plt.figure()

ax = fig.add_subplot(431)
ax.imshow(mpimg.imread('test.jpg'))
ax.set_title('Original')
ax = fig.add_subplot(434)
ax.imshow(pop.r[:,:,0], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('pop.r R')
ax = fig.add_subplot(435)
ax.imshow(pop.r[:,:,1], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('pop.r G')
ax = fig.add_subplot(436)
ax.imshow(pop.r[:,:,2], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('pop.r B')
ax = fig.add_subplot(437)
ax.imshow(pooled.r[:,:,0], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('pooled.r R')
ax = fig.add_subplot(438)
ax.imshow(pooled.r[:,:,1], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('pooled.r G')
ax = fig.add_subplot(439)
ax.imshow(pooled.r[:,:,2], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('pooled.r B')
ax = fig.add_subplot(4, 3, 10)
ax.imshow(filtered.r[:,:,0], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('Red color filter')
ax = fig.add_subplot(4, 3, 11)
ax.imshow(filtered.r[:,:,1], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('Green color filter')
ax = fig.add_subplot(4, 3, 12)
ax.imshow(filtered.r[:,:,2], cmap='gray', vmin= 0.0, vmax=1.0)
ax.set_title('Blue color filter')

plt.show()
