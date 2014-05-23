***********************************
Structural plasticity
***********************************

ANNarchy supports the dynamic change of dendritic structures through simulation. Two functions are available for this:

    * add_synapse

    * remove_synapse

provided by the Dendrite class.

Add synapses to a dendrite
===================================

As following we show a simple example how adding synapses based on a fixed probability could implemented. The code snippet assumes an existing Projection 
*proj* and could take place after simulate() call.

    .. code-block :: python

        ranks = proj.dendrite(0).rank

        for r in xrange(proj.pre.size):
            if r in ranks:
                continue    # The synapse to this neuron already exists

            if numpy.random.rand() < 0.1:
                proj.dendrite(0).add_synapse(r, 1.0)    # add a synapse to *r* with weight 1.0 and no delay
            
Remove synapses from a dendrite
===================================

As following we show a simple example how removing synapses based on their weights could implemented. The code snippet assumes an existing Projection 
*proj* and could take place after simulate() call.

    .. code-block :: python

        ranks = proj.dendrite(0).rank
        values = proj.dendrite(0).value

        
        for r in xrange(len(ranks)):
            if values[r] < 0.01:
                proj.dendrite(0).remove_synapse(r)
            
