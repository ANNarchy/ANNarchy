# -*- coding: utf-8 -*-
#
# brunel2000_rand_plastic.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import nest
import nest.raster_plot
import pylab
import numpy
from time import time
import sys

# ###########################################
# Configuration
# ###########################################
record = True
plot_all = True

N_vp      = 1     # number of virtual processes to use
base_seed = 10000  # increase in intervals of at least 2*n_vp+1
N_rec     = 50    # Number of neurons to record from
data2file = False # whether to record data to file

nest.ResetKernel()
nest.SetKernelStatus({"print_time": True,
                      "total_num_virtual_procs": N_vp,
                      "overwrite_files": True})
n_vp = nest.GetKernelStatus('total_num_virtual_procs')
bs = base_seed # abbreviation to make following code more compact
pyrngs = [numpy.random.RandomState(s) for s in range(bs, bs+n_vp)]
nest.SetKernelStatus({'grng_seed': bs+n_vp,
                      'rng_seeds': range(bs+n_vp+1, bs+1+2*n_vp)})

# ###########################################
# Parameters
# ###########################################
# Network parameters. These are given in Brunel (2000) J.Comp.Neuro.
g       = 5.0    # Ratio of IPSP to EPSP amplitude: J_I/J_E
eta     = 2.0    # rate of external population in multiples of threshold rate
delay   = 1.5    # synaptic delay in ms
tau_m   = 20.0   # Membrane time constant in mV
V_th    = 20.0   # Spike threshold in mV
simtime = 300.0 # how long shall we simulate [ms]

N_E = 8000
N_I = 2000
N_neurons = N_E+N_I

C_E    = int(N_E/10) # number of excitatory synapses per neuron
C_I    = int(N_I/10) # number of inhibitory synapses per neuron

J_E  = 0.1
J_I  = -g*J_E

# Synaptic parameters
STDP_alpha   = 2.02     # relative strength of STDP depression w.r.t potentiation 
STDP_Wmax = 3*J_E       #maximum weight of plastic synapse

# ###########################################
# Neuron model
# ###########################################
nest.SetDefaults("iaf_psc_delta", 
                 {"C_m": 1.0,
                  "tau_m": tau_m,
                  "t_ref": 2.0,
                  "E_L": 0.0,
                  "V_th": V_th,
                  "V_reset": 10.0})

# ###########################################
# Populations
# ###########################################
nodes   = nest.Create("iaf_psc_delta",N_neurons)
nodes_E = nodes[:N_E]
nodes_I = nodes[N_E:]

# randomize membrane potential
node_E_info = nest.GetStatus(nodes_E, ['local','global_id','vp'])
node_I_info = nest.GetStatus(nodes_I, ['local','global_id','vp'])
local_E_nodes = [(gid,vp) for islocal,gid,vp in node_E_info if islocal]
local_I_nodes = [(gid,vp) for islocal,gid,vp in node_I_info if islocal]
for gid,vp in local_E_nodes + local_I_nodes: 
  nest.SetStatus([gid], {'V_m': pyrngs[vp].uniform(-V_th,0.95*V_th)})

# Poisson input
#noise = nest.Create("poisson_generator", N_neurons,{"rate": 20.}) # very slow
noise = nest.Create("poisson_generator", 1,{"rate": 20000.}) 

# ###########################################
# Synapse models
# ###########################################
nest.CopyModel("stdp_synapse_hom",
               "excitatory-plastic",
               {"alpha":STDP_alpha,
                "Wmax":STDP_Wmax})

nest.CopyModel("static_synapse", "excitatory-static")

nest.CopyModel("static_synapse",
               "inhibitory",
               {"weight":J_I, 
                "delay":delay})

nest.CopyModel("static_synapse_hom_wd",
               "excitatory-input",
               {"weight":J_E, "delay":delay})

# ###########################################
# Projections
# ###########################################
nest.Connect(nodes_E, nodes_E,
             {"rule": 'fixed_indegree', "indegree": C_E},
             {"model": "excitatory-plastic", "delay": delay,
              "weight": {"distribution": "uniform",
                         "low": 0.5 * J_E, "high": 1.5 * J_E}})

nest.Connect(nodes_E, nodes_I,
             {"rule": 'fixed_indegree', "indegree": C_E},
             {"model": "excitatory-static", "delay": delay,
              "weight": {"distribution": "uniform",
                         "low": 0.5 * J_E, "high": 1.5 * J_E}})


nest.Connect(nodes_I, nodes,
             {"rule": 'fixed_indegree', "indegree": C_I},
             "inhibitory")

nest.Connect(noise, nodes, 
             'all_to_all', #{"rule": 'fixed_indegree', "indegree": 1000}, 
             "excitatory-input")

if record:
    spikes=nest.Create("spike_detector",1, 
                       [{"label": "brunel-py-ex", "to_file": data2file}])

    # connect using all_to_all: all recorded excitatory neurons to one detector
    nest.Connect(nodes_E[:N_rec], spikes, 'all_to_all')

# ###########################################
# Simulation
# ###########################################
ts = time()
nest.Simulate(simtime)
print 'Simulating', simtime, 'ms took', time() - ts, 'seconds.'

# ###########################################
# Data analysis
# ###########################################
if record:
    events = nest.GetStatus(spikes,"n_events")
    N_rec_local_E = sum(nest.GetStatus(nodes_E[:N_rec], 'local'))
    rate_ex= events[0]/simtime*1000.0/N_rec_local_E
    print("Excitatory rate   : %.2f Hz" % rate_ex)
    if plot_all:
        nest.raster_plot.from_device(spikes, hist=True)
        pylab.show()

# weights of excitatory connections
if plot_all:
  w = nest.GetStatus(nest.GetConnections(nodes_E[:N_rec],
    synapse_model='excitatory-plastic'),
    'weight')
  pylab.figure()
  pylab.hist(w, bins=100)
  pylab.xlabel('Synaptic weight [pA]')
  pylab.show()
