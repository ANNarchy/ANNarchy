from ANNarchy import *

# Compulsory to allow structural plasticity
setup(structural_plasticity=True)

# Simple neuron type
LeakyIntegratorNeuron = Neuron(
    parameters="""
        tau = 10.0 : population
        baseline = 0.0 
    """,
    equations = """
        tau * dr/dt + r = baseline + sum(exc) : min=0.0
    """
)

# Structurally plastic synapse
StructuralPlasticSynapse = Synapse(
    parameters = " T = 10000 : int, projection ",
    equations = """
        age = if pre.r * post.r > 1.0 :
                0
              else :
                age + 1 : init = 0, int""",
    pruning = "age > T : proba = 0.2",
    creating = "pre.r * post.r > 1.0 : proba = 0.1, w = 0.01",
)

# A single population
pop = Population(100, LeakyIntegratorNeuron)

# Lateral excitatory projection, initially sparse
proj = Projection(pop, pop, 'exc', StructuralPlasticSynapse)
proj.connect_fixed_probability(weights = 0.01, probability=0.1)


compile()

# Save the initial connectivity matrix
initial_weights = proj.connectivity_matrix()

# Start creating and pruning
proj.start_creating(period=100.0)
proj.start_pruning(period=100.0)

# Let structural plasticity over several trials
num_trials = 100
for trial in range(num_trials):
    # Activate the first subpopulation
    pop[:50].baseline = 1.0
    # Simulate for 1s
    simulate(1000.)
    # Reset the population
    pop.baseline = 0.0
    simulate(100.)
    # Activate the second subpopulation
    pop[50:].baseline = 1.0
    # Simulate for 1s
    simulate(1000.)
    # Reset the population
    pop.baseline = 0.0
    simulate(100.)

# Inspect the final connectivity matrix
final_weights = proj.connectivity_matrix()

# Visualize the two connectivity matrices
import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(initial_weights)
plt.title('Connectivity matrix before')
plt.subplot(122)
plt.imshow(final_weights)
plt.title('Connectivity matrix after')
plt.show()



