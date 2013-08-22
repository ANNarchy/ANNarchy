from ANNarchy4.core.Random import *
import ANNarchy4.core.Projection as PyProjection

from libcpp.vector cimport vector 
from libc.stdlib cimport malloc
from libc.math cimport exp, abs

cdef class Connector:
    cdef proj_type
    
    def __cinit__(self, proj_type):
        print 'ProjClass:', proj_type
        self.proj_type = proj_type
        
cdef class One2OneConnector(Connector):

    cdef postSize

    def __cinit__(self, proj_type):
        Connector.__init__(proj_type)
        
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
        self.postSize = post.size

        if (pre.size != self.postSize):
            return None

        r = self.genRanks()

        Proj = []
        
        for p in xrange(self.postSize):
            local = LocalProjection(self.proj_type, pre.id, post.id, p, target)
            v = distribution.getValue()

            local.init(r[p], v)            
            
            Proj.append(local)

        return Proj    

cdef class All2AllConnector(Connector):

    cdef allowSelfConnections
    cdef preID
    cdef postID
    cdef preSize
    cdef postSize
    cdef vector[vector[int]] ranks

    def __cinit__(self, proj_type):
        Connector.__init__(proj_type)

    cdef genRanks(self):
        cdef int i
        cdef int j
        cdef vector[int] tmp
        
        self.ranks = vector[vector[int]]()
        
        if (self.preID == self.postID) and not self.allowSelfConnections:
            for i in range(self.postSize):
                tmp.clear()
                j = 0
    
                while j < self.preSize:
                    if( i != j):
                        tmp.push_back(j)
                    j = j+1

                self.ranks.push_back(tmp)
        else:
            tmp.clear()
            for i in xrange(self.preSize):
                tmp.push_back(i)

            for j in xrange(self.postSize):
                self.ranks.push_back(tmp)
    
    def connect(self, pre, post, distribution, target, parameters):

        self.preID = pre.id
        self.postID = post.id        
        self.preSize = pre.size
        self.postSize = post.size

        if 'allowSelfConnections' in parameters.keys():
            self.allowSelfConnections = parameters['allowSelfConnections']
        else:
            self.allowSelfConnections = False

        self.genRanks()
        
        Proj = []

        for p in xrange(self.postSize):
            local = LocalProjection(self.proj_type, pre.id, post.id, p, target)
            if not self.allowSelfConnections and (pre==post):
                v = distribution.getValues(self.preSize-1)
            else:
                v = distribution.getValues(self.preSize)
            
            local.init(self.ranks[p], v)
            Proj.append(local)            

        return Proj

cdef class DoGConnector(Connector):

    cdef preSize
    cdef postSize
    cdef pre
    cdef post
    cdef amp_pos
    cdef amp_neg
    cdef sigma_pos
    cdef sigma_neg
    cdef limit

    def __cinit__(self, proj_type):
        Connector.__init__(proj_type)
      
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

        normPost = self.post.normalized_coordinates_from_rank(postRank)

        for j in range(self.preSize):
             if (not selfConnection or (self.pre != self.post)):
                 normPre = self.pre.normalized_coordinates_from_rank(j)
                 dist = self.compDist(normPre, normPost)
                 value = self.amp_pos*exp(-dist/2.0/self.sigma_pos/self.sigma_pos) - self.amp_neg*exp(-dist/2.0/self.sigma_neg/self.sigma_neg)
                 if (abs(value) > self.limit*abs(self.amp_pos-self.amp_neg)):
                     ranks.push_back(j)
                     values.push_back(value)
        
        return ranks, values

    def connect(self, pre, post, distribution, target, parameters):
        self.preSize = pre.size
        self.postSize = post.size
        self.pre = pre
        self.post = post

        self.sigma_pos = parameters['sigma_pos']
        self.sigma_neg = parameters['sigma_neg']
        self.amp_pos = parameters['amp_pos']
        self.amp_neg = parameters['amp_neg']

        if 'limit' in parameters.keys():
            self.limit = parameters['limit']
        else:
            self.limit = 0.01

        if (self.preSize != self.postSize):
            return None

        Proj = []

        for p in xrange(self.postSize):
            local = LocalProjection(self.proj_type, pre.id, post.id, p, target)
            
            r, v = self.genRanksAndValues(p)

            local.init(r, v)
            Proj.append(local)

        return Proj

