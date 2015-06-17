import sys
import nest
import nest.raster_plot

# ###########################################
# Configuration
# ###########################################
nest.SetKernelStatus({"resolution": 0.1})
if len(sys.argv) > 1:
    nb_threads = sys.argv[1]
else:
    nb_threads = 1
nest.SetKernelStatus({"local_num_threads": int(nb_threads)})

# ###########################################
# Parameters 
# ###########################################
J_ex  = 0.1 # excitatory weight
J_in  = -0.5 # inhibitory weight
p_rate = 20.0 # external Poisson rate
simtime = 100. # simulation time
Nrec = 1000 # number of neurons to record from
 
neuron_params= {"C_m": 1.0, "tau_m": 20.0, "t_ref": 2.0,
                "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": 20.0}
 
# ###########################################
# Neuron models
# ###########################################
nest.SetDefaults("iaf_psc_delta", neuron_params)
nest.SetDefaults("poisson_generator",{"rate": p_rate})
nest.SetDefaults("spike_detector",{"withtime": True, "withgid": True})

# ########################################### 
# Populations
# ###########################################
nodes_ex = nest.Create("iaf_psc_delta", 10000) 
nodes_in = nest.Create("iaf_psc_delta", 2500)
parrots = nest.Create("parrot_neuron", 12500)
noise = nest.Create("poisson_generator")
espikes = nest.Create("spike_detector")

# ########################################### 
# Synapse models
# ###########################################
nest.CopyModel("static_synapse_hom_wd", "excitatory", {"weight":J_ex, "delay":1.5})
nest.CopyModel("static_synapse_hom_wd", "inhibitory", {"weight":J_in, "delay":1.5})
 
# ###########################################
# Projections
# ###########################################
nest.Connect(nodes_ex, nodes_ex+nodes_in, {"rule": 'fixed_indegree', "indegree": 1000}, "excitatory")
nest.Connect(nodes_in, nodes_ex+nodes_in, {"rule": 'fixed_indegree', "indegree": 250}, "inhibitory")
nest.Connect(noise, parrots, 'all_to_all')
nest.Connect(parrots, nodes_ex+nodes_in, {"rule": 'fixed_indegree', "indegree": 1000}, "excitatory")
nest.Connect(nodes_ex[:Nrec], espikes, 'all_to_all')

# ###########################################
# Simulation
# ###########################################
from time import time
ts = time()
nest.Simulate(simtime)
print 'Simulating', simtime, 'ms took', time() - ts, 'seconds.'

# ###########################################
# Data analysis
# ###########################################
events = nest.GetStatus(espikes,"n_events")
N_rec_local_E = sum(nest.GetStatus(nodes_ex[:Nrec], 'local'))
rate_ex= events[0]/simtime*1000.0/N_rec_local_E
print("Mean firing rate: %.2f Hz" % rate_ex)

# Plot results
nest.raster_plot.from_device(espikes, hist=True)
nest.raster_plot.show()
