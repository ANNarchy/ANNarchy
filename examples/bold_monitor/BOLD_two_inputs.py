#   BOLD monitoring example to demonstrate the two-input model
#
#   please note: This example is just intented to demonstrate the coupling between the source and input variables.
#                The coupling used in this script are not based on biological constraints.
#
#   More details can be found in the recent article: TODO
#
#   author: Helge Uelo Dinkelbach, Oliver Maith
from ANNarchy import *
from ANNarchy.extensions.bold import *

import matplotlib.pyplot as plt

# Two populations of 100 izhikevich neurons
pop0 = Population(100, neuron=Izhikevich)
pop1 = Population(100, neuron=Izhikevich)

# Set noise to create some baseline activity
pop0.noise = 5.0; pop1.noise = 5.0

# Compute mean firing rate in Hz on 100ms window
pop0.compute_firing_rate(window=100.0)
pop1.compute_firing_rate(window=100.0)

# Create required monitors
mon_pop0 = Monitor(pop0, ["r"], start=False)
mon_pop1 = Monitor(pop1, ["r"], start=False)

m_bold = BoldMonitor(
    populations = [pop0, pop1], # recorded populations
    bold_model = balloon_two_inputs(), # BOLD model to use 
    # mean firing rate as source variable coupled to the input variable I_CBF
    # membrane potential as source variable coupled to the input variable I_CMRO2
    mapping={'I_CBF': 'r','I_CMRO2': 'v'},            
    normalize_input=2000, # time window to compute the baseline
    recorded_variables=["I_CBF", "I_CMRO2", "BOLD"]
)

# Compile and initialize the network
compile()

# Ramp up time
simulate(1000)

# Start recording
mon_pop0.start()
mon_pop1.start()
m_bold.start()

# we manipulate the noise for the half of the neurons
simulate(5000)      # 5s with low noise
pop0.noise = 7.5
simulate(5000)      # 5s with higher noise (one population)
pop0.noise = 5
simulate(10000)     # 10s with low noise

# retrieve the recordings
mean_fr1 = np.mean(mon_pop0.get("r"), axis=1)
mean_fr2 = np.mean(mon_pop1.get("r"), axis=1)

If_data = m_bold.get("I_CBF")
Ir_data = m_bold.get("I_CMRO2")
bold_data = m_bold.get("BOLD")

# An example evaluation, which consists of:
# A) the mean firing activity
# B) the recorded activity which serves as input to BOLD
# C) the resulting BOLD signal
plt.figure(figsize=(20,6))
grid = plt.GridSpec(1, 3, left=0.05, right=0.95)

# mean firing rate
ax1 = plt.subplot(grid[0, 0])
ax1.plot(mean_fr1, label="pop0")
ax1.plot(mean_fr2, label="pop1")
plt.legend()
ax1.set_ylabel("average mean firing rate [Hz]", fontweight="bold", fontsize=18)

# BOLD input signal
ax2 = plt.subplot(grid[0, 1])

ax2.plot(If_data, label='I_CBF')
ax2.plot(Ir_data, label='I_CMRO2')
ax2.set_ylabel("BOLD input variables", fontweight="bold", fontsize=18)
ax2.legend()

# BOLD input signal as percent
ax3 = plt.subplot(grid[0, 2])
ax3.plot(bold_data*100.0)
ax3.set_ylabel("BOLD [%]", fontweight="bold", fontsize=18)

# x-axis labels as seconds
for ax in [ax1, ax2, ax3]:
    ax.set_xticks(np.arange(0,21,2)*1000)
    ax.set_xticklabels(np.arange(0,21,2))
    ax.set_xlabel("time [s]", fontweight="bold", fontsize=18)

plt.show()
