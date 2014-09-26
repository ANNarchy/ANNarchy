from ANNarchy import *
from ANNarchy.extensions.image import *

# Create the population    
pop = ImagePopulation(geometry=(480, 640))

# Compile
compile()

# Set the image
pop.set_image('test.jpg')

# Visualize with Matplotlib
import pylab as plt
import matplotlib.image as mpimg

fig = plt.figure()
if pop.dimension == 3: # color
    ax = fig.add_subplot(221)
    ax.imshow(mpimg.imread('test.jpg'))
    ax.set_title('Original')
    ax = fig.add_subplot(222)
    ax.imshow(pop.r[:,:,0], cmap='gray', vmin= 0.0, vmax=1.0)
    ax.set_title('pop.r R')
    ax = fig.add_subplot(223)
    ax.imshow(pop.r[:,:,1], cmap='gray', vmin= 0.0, vmax=1.0)
    ax.set_title('pop.r G')
    ax = fig.add_subplot(224)
    ax.imshow(pop.r[:,:,2], cmap='gray', vmin= 0.0, vmax=1.0)
    ax.set_title('pop.r B')
else: # grayscale
    ax = fig.add_subplot(121)
    ax.imshow(mpimg.imread('test.jpg'))
    ax.set_title('Original')
    ax = fig.add_subplot(122)
    ax.imshow(pop.r, cmap='gray', vmin= 0.0, vmax=1.0)
    ax.set_title('pop.r')

plt.show()
