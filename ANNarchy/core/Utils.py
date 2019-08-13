#===============================================================================
#
#     Utils.py
#
#     This file is part of ANNarchy.
#
#     Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
#     Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     ANNarchy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================
import numpy as np
import ANNarchy.core.Global as Global


######################
# Sparse matrices
######################

def sparse_random_matrix(pre, post, p, weight, delay=0):
    """
    Returns a sparse (lil) matrix to connect the pre and post populations with the probability p and the value weight.
    """
    try:
        from scipy.sparse import lil_matrix
    except:
        Global._warning("scipy is not installed, sparse matrices won't work")
        return None
    from random import sample
    W=lil_matrix((pre, post))
    for i in range(pre):
        k=np.random.binomial(post,p,1)[0]
        W.rows[i]=sample(range(post),k)
        W.data[i]=[weight]*k

    return W



