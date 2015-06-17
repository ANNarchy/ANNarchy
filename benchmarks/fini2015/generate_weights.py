from ANNarchy import sparse_random_matrix

e = sparse_random_matrix(3200, 4000, 0.02, 1.0)
i = sparse_random_matrix(800, 4000, 0.02, 1.0)

import cPickle

with open('exc.data', 'w') as wfile:
    cPickle.dump(e, wfile)
with open('inh.data', 'w') as wfile:
    cPickle.dump(i, wfile)