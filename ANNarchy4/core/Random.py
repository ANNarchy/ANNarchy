import numpy as np

class RandomDistribution:
    def __init__(self, distribution='uniform', parameters=[0,1], seed=-1):
        self.distribution = distribution
        self.parameters = parameters
        self.seed = seed

    def getValues(self, shape):
      
        if self.distribution == 'uniform':
            return np.random.uniform(self.parameters[0], self.parameters[1], shape)
        if self.distribution == 'normal':
            return np.random.normal(self.parameters[0], self.parameters[1], shape)
        if self.distribution == 'constant':
            return np.array([self.parameters[0]])
        else:
            print 'Unknown distribution.'

    def getValue(self):
        if self.distribution == 'uniform':
            return np.random.uniform(self.parameters[0], self.parameters[1], (1))
        if self.distribution == 'normal':
            return np.random.normal(self.parameters[0], self.parameters[1], (1))
        if self.distribution == 'constant':
            return np.array([self.parameters[0]])
        else:
            print 'Unknown distribution.'

    def genCPP(self):
        code = ''

        if self.distribution == 'uniform':
            # (min, max)
            if len(self.parameters)==2:
                if(self.seed == -1):
                    code = 'UniformDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+')'
                else:
                    code = 'UniformDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+','+ str(self.seed)+')'
            else:
                print 'wrong parameters for uniform distribution, expected [min, max]'
        elif self.distribution == 'normal':
            # (mean, sigma)
            if len(self.parameters)==2:
                if(self.seed == -1):
                    code = 'NormalDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+')'
                else:
                    code = 'NormalDistribution<DATA_TYPE>('+str(self.parameters[0])+','+ str(self.parameters[1])+','+ str(self.seed)+')'
            else:
                print 'wrong parameters for normal distribution, expected [mean, sigma].'
        elif self.distribution == 'constant':
            # (constant)
            if len(self.parameters)==1:
                code = 'Constant<DATA_TYPE>('+str(self.parameters[0])+')'
            else:
                print 'wrong parameters for constant, expected [constant].'
        else:
            print 'unknown distribution type.'

        return code
