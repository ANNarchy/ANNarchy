#
#   ANNarchy - Refractoriness
#
#   A simple model showing the usage of refractoriness in ANNarchy
# 
#   See e.g. Mainen & Sejnowski (1995) for experimental results in vitro.
#
#   Code adapted from the Brian example: https://brian2.readthedocs.org/en/latest/examples/non_reliability.html
#
#   Note that the Brian example uses constant refractory periods of 5 ms, while they are uniformaly distributed between 1 and 5 ms here.
#
#   authors: Helge Uelo Dinkelbach, Julien Vitay
#
from ANNarchy import *

Neuron = Neuron(
parameters = """
    tau = 20.0 : population
    sigma = 0.015 : population
    target = 1.1 : population
""",
equations = """
    noise =  sigma * sqrt( 2.0 * tau ) * Normal(0.0, 1.0) 
    tau * dx/dt + x = target + noise
""",
spike = """
    x > 1
""",
reset = """
    x = 0
""",
refractory = Uniform(1.0, 5.0)
)

pop = Population( geometry=25, neuron = Neuron )

compile()

m = Monitor(pop, 'spike')
simulate ( 500.0 )

times, ranks = m.raster_plot()

# Plot the results
import matplotlib.pyplot as plt
plt.plot(times, ranks, '.')
plt.show()
