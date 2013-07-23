from ANNarchy4.core.Random import *

from libcpp.vector cimport vector 
from libc.stdlib cimport malloc
from libc.math cimport exp, abs

cdef class One2OneConnector:

    cdef weights
    cdef delays
    cdef postSize

    def __cinit__(self, weights=RandomDistribution('constant',[0.0]), delays=None):
        self.weights = weights
        self.delays = delays

    cdef genRanks(self):
        cdef int i
        cdef vector[int] tmp
        cdef vector[vector[int]] ranks

        ranks.clear()
        tmp.clear()
        tmp.push_back(0)

        for i in range(self.postSize):
            tmp[0] = i

            ranks.push_back(tmp)

        return ranks

    def connect(self, pre, post, distribution, target, parameters):
        self.postSize = post.getSize()

        if (pre.getSize() != self.postSize):
            return None

        r = self.genRanks()
        Proj = pyProjection(pre.id, post.id, self.postSize, target)

        for p in xrange(self.postSize):
            v = distribution.getValue()

            Proj.getLocal(p).init(r[p], v)            

        return Proj    

cdef class All2AllConnector:

    cdef weights
    cdef delays
    cdef allowSelfConnections
    cdef preID
    cdef postID
    cdef preSize
    cdef postSize

    def __cinit__(self, allow_self_connections = False, weights=RandomDistribution('constant',[0.0]), delays=None):
        self.weights = weights
        self.delays = delays

        self.allowSelfConnections = allow_self_connections

    cdef genRanks(self):
        cdef int i
        cdef int j
        cdef vector[int] tmp
        cdef vector[vector[int]] ranks
        
        ranks.clear()
        if (self.preID == self.postID) and not self.allowSelfConnections:
            tmp.clear()

            instance()
            for i in range(self.postSize):

                j = 0
                while j < self.preSize:
                    if( i != j):
                        tmp.push_back(j)
                    j = j+1

                ranks.push_back(tmp)
                i=i+1
        else:
             tmp.clear()
             for i in xrange(self.preSize):
                 tmp.push_back(i)

             for j in xrange(self.postSize):
                 ranks.push_back(tmp)

        return ranks

    def connect(self, pre, post, distribution, target, parameters):
        cdef int preID, postID

        self.preSize = pre.getSize()
        self.postSize = post.getSize()

        self.preID = pre.id
        self.postID = post.id

        r = self.genRanks()
        
        Proj = pyProjection(self.preID, self.postID, self.postSize, target)

        for p in xrange(self.postSize):
            if(self.allowSelfConnections and (pre==post)):
                v = distribution.getValues(self.preSize-1)
            else:
                v = distribution.getValues(self.preSize)

            Proj.getLocal(p).init(r[p], v)            

        return Proj

cdef class DoGConnector:

    cdef weights
    cdef delays
    cdef preSize
    cdef postSize
    cdef pre
    cdef post
    cdef amp_pos
    cdef amp_neg
    cdef sigma_pos
    cdef sigma_neg
    cdef limit

    def __cinit__(self, weights=RandomDistribution('constant',[0.0]), delays=None):
        self.weights = weights
        self.delays = delays
        self.amp_pos = 1.0
        self.amp_neg = 1.0
        self.sigma_pos = 1.0
        self.sigma_neg = 1.0
        self.limit = 0.1
      
    cdef compDist(self, pre, post):
        cdef float res = 0.0
        cdef int i = 0

        for i in range(len(pre)):
            res = res + (pre[i]-post[i])*(pre[i]-post[i]);

        return res

    cdef genRanksAndValues(self, postRank):
        cdef int j
        cdef float dist, value
        cdef vector[int] normPre, normPost
        cdef vector[int] ranks
        cdef vector[float] values
        selfConnection = False

        ranks.clear()
        values.clear()

        normPost = self.post.getNormalizedCoordinateFromRank(postRank)

        for j in range(self.preSize):
             if (not selfConnection or (self.pre != self.post)):
                 normPre = self.pre.getNormalizedCoordinateFromRank(j)
                 dist = self.compDist(normPre, normPost)
                 value = self.amp_pos*exp(-dist/2.0/self.sigma_pos/self.sigma_pos) - self.amp_neg*exp(-dist/2.0/self.sigma_neg/self.sigma_neg)
                 if (abs(value) > self.limit*abs(self.amp_pos-self.amp_neg)):
                     ranks.push_back(j)
                     values.push_back(value)
        
        return ranks, values

    def connect(self, pre, post, distribution, target, parameters):
        self.preSize = pre.getSize()
        self.postSize = post.getSize()
        self.pre = pre
        self.post = post

        self.sigma_pos = parameters['sigma_pos']
        self.sigma_neg = parameters['sigma_neg']
        self.amp_pos = parameters['amp_pos']
        self.amp_neg = parameters['amp_neg']

        if (pre.getSize() != self.postSize):
            return None

        Proj = pyProjection(pre.id, post.id, self.postSize, target)

        for p in xrange(self.postSize):
            r, v = self.genRanksAndValues(p)

            Proj.getLocal(p).init(r, v)

        return Proj

