*******************************
********** Interface **********
*******************************
== Top-Level methods: ==
Basics
    * [ ] setup
    * [ ] compile
    * [ ] clear
    * [ ] simulate
    * [ ] simulate_until
    * [ ] step
    * [ ] reset
    * [ ] every
Callbacks
    * [ ] enable_callbacks
    * [ ] disable_callbacks
    * [ ] clear_all_callbacks
Access
    * [ ] get_population
    * [ ] get_projection
    * [ ] PopulationView
Learning
    * [ ] enable_learning
    * [ ] disable_learning
Time
    * [ ] get_time
    * [ ] set_time
    * [ ] get_current_step
    * [ ] set_current_step
    * [ ] dt

* == Network ==
    * [x] add
    * [x] compile
    * [ ] enable_learning / disable_learning
    * [x] get
    * [ ] get_current_step
    * [x] get_population
    * [ ] get_populations
    * [ ] get_projection
    * [ ] get_projections
    * [ ] get_time
    * [ ] load / save
    * [x] reset
    * [ ] set_current_step
    * [x] set_seed
    * [ ] set_time
    * [x] simulate
    * [ ] simulate_until
    * [ ] step
    * [ ] parallel_run

== Monitor ==
Basics
    * [x] period
    * [ ] period_offset
    * [ ] variables
    * [ ] times
    * [x] get
    * [ ] size_in_bytes
    * [x] start
    * [x] pause
    * [x] resume
    * [ ] stop
Analysis
    * [ ] histogram
    * [ ] mean_fr
    * [ ] population_rate
    * [ ] raster_plot
    * [ ] smoothed_rate

== Saving / Loading ==
    * [ ] Population.load / .save
    * [ ] Projection.save_connectivity / .connect_from_file
    * [ ] Projection.load / .save

== Tensorboard Logging ==
    * [ ] add_figure
    * [ ] add_histogram
    * [ ] add_image
    * [ ] add_images
    * [ ] add_parameters
    * [ ] add_scalar
    * [ ] add_scalars
    * [ ] close
    * [ ] flush

== BOLD Monitor ==
BoldMonitor
    * [ ] get
    * [ ] start
    * [ ] stop
BOLD models
    * [ ] BoldModel
    * [ ] balloon_RN
    * [ ] balloon_RL
    * [ ] balloon_CL
    * [ ] balloon_CN
    * [ ] balloon_maith2021
    * [ ] balloon_two_inputs

== Report ==
    * [ ] Generates a tex file

****************************
********** Neuron **********
****************************
== Neurons + built-ins ==
Rate Neuron
    * [x] parameters
    * [x] equations
    * [x] functions
Spike Neuron
    * [x] parameters
    * [x] equations
    * [ ] functions
    * [x] spike
    * [ ] axon_spike
    * [x] reset
    * [ ] axon_reset
    * [ ] refactory
Built-ins -> only compile tests (semantic tests might be hard)
    * [x] LeakyIntegrator
    * [ ] Izhikevich
    * [ ] IF_curr_exp
    * [ ] IF_cond_exp
    * [ ] IF_curr_alpha
    * [ ] IF_cond_alpha
    * [ ] HH_cond_exp
    * [ ] EIF_cond_exp_isfa_ista
    * [ ] EIF_cond_alpha_isfa_ista

== Populations / Specific Populations ==
Basics
    * [x] init
    * [ ] init with stop_condition
    * [?] init with storage_order ("post_to_pre", "pre_to_post")
    * [x] get / set
    * [ ] sum
    * [l] neuron
    * [ ] enable / disable
    * [ ] clear
    * [ ] reset
    * [ ] compute_firing_rate
    * [ ] size_in_bytes
    * [x] coordinates_from_rank
    * [ ] normalized_coordinates_from_rank
    * [x] rank_from_coordinates
Specific Populations
    * [x] TimedArray
    * [ ] SpikeSourceArray
    * [ ] HomogeneousCorrelatedSpikeTrains
    * [l] PoissonPopulation
    * [l] TimedPoissonPopulation
    * [l] ImagePopulation
    * [l] VideoPopulation

*****************************
********** Synapse **********
*****************************
== Synapse + built-ins ==
Rate Synapse
    * [x] parameters
    * [x] equations
    * [x] psp
    * [x] operation
    * [x] functions
Spiking Synapse
    * [x] parameters
    * [ ] equations
    * [x] psp
    * [x] pre_spike
    * [x] post_spike
    * [ ] pre_axon_spike
    * [ ] functions -> auch in spike_condition
Built-ins -> compile
    * [ ] Hebb
    * [ ] Oja
    * [ ] IBCM
    * [ ] STP
    * [ ] STDP

== Projections ==
Basics
    * [ ] init
    * [x] -> synapse
    * [ ] -> disable_omp -> test_OptimizationFlags
    * [ ] set / get
    * [ ] reset
    * [x] dendrite
    * [ ] dendrites
    * [ ] synapse
    * [x] size
    * [ ] size_in_bytes
    * [ ] nb_synapses
    * [ ] nb_efferent_synapses
    * [ ] nb_synapses_per_dendrite
    * [l] connectivity_matrix
    * [l] receptive_field
    * [ ] disable / enable_learning
Structural Plasticity
    * [ ] start_creating
    * [ ] start_pruning
    * [ ] stop_creating
    * [ ] stop_pruning
Connectors
    * [x] connect_all_to_all
    * [ ] connect_dog
    * [ ] connect_fixed_number_post
    * [x] connect_fixed_number_pre
    * [ ] connect_fixed_probability
    * [ ] connect_from_matrix
    * [l] connect_from_matrix_market
    * [x] connect_from_sparse
    * [ ] connect_gaussian
    * [x] connect_one_to_one
    * [x] connect_with_func

== Dendrite ==
    * [x] pre_ranks
    * [x] size
    * [ ] synapse
    * [ ] synapses
    * [x] create_synapse -> stuctural plasticity
    * [x] prune_synapse -> stuctural plasticity
    * [x] get / set
    * [ ] receptive_field

== Convolution / Pooling ==
Convolution
    * [x] connect_filter
    * [x] connect_filters
    * [ ] connectivity_matrix
    * [ ] load
    * [ ] receptive_fields
    * [ ] save
    * [ ] save_connectivity
Pooling
    * [x] connect_pooling
    * [ ] connectivity_matrix
    * [ ] load
    * [ ] receptive_fields
    * [ ] save
    * [ ] save_connectivity
Copy
    * [ ] connect_copy
    * [ ] connectivity_matrix
    * [ ] generate_omp
    * [ ] load
    * [ ] receptive_fields
    * [ ] save
    * [ ] save_connectivity

== Hybrid Networks ==
    * [x] CurrentInjection
    * [ ] DecodingProjection

****************************
********** Common **********
****************************
Functions / Constants
    * [x] add_function
    * [ ] functions
    * [ ] parser_flags
        * [ ] init
        * [ ] min
        * [ ] max
        * [ ] population
        * [ ] postsynaptic
        * [ ] synaptic
        * [ ] projection
        * [x] exponential, midpoint
        * [Neuron only] explicit, implicit, event-driven
    * [ ] Global Operations (min, max, mean, norm1, norm2)
    * [ ] Constant (init, new, set)

== Random Distributions ==
get_values()
    * [x] Uniform
    * [ ] DiscreteUniform
    * [x] Normal
    * [ ] LogNormal
    * [ ] Gamma
    * [ ] Exponential

