import Global
import os, sys
import Master

from ANNarchy4 import generator

class Projection:
    """
    Definition of a projection.
    """
    def __init__(self, pre, post, target, connector, synapse=None):
        """
        * pre: pre synaptic layer (either string or object)
        * post: post synaptic layer (either string or object)
        * target: connection type
        * connector: connection pattern
        * synapse: synapse object
        """
        
        # the user provide either a string or a population object
        # in case of string, we need to search for the corresponding object 
        if isinstance(pre, str):
            for p in Global.populations_:
                if p.name == pre:
                    self.pre = p
        else:
            self.pre = pre
                                 
        if isinstance(post, str):
            for p in Global.populations_:
                if p.name == post:
                    self.post = p
        else:
            self.post= post
            
        self.post.generator.add_target(target)
        self.target = target
        self.connector = connector
        self.synapse = synapse

        self.projClass = None

        self.generator = generator.Projection(synapse)
        Global.projections_.append(self)        

    def generate_cpp_add(self):
        if self.connector != None:
            return ('net_->connect('+
                str(self.pre.id)+', '+
                str(self.post.id)+', '+
                self.connector.cpp_call() +', '+ 
                str(self.projClass['ID'])+', '+ 
                str(self.post.generator.targets.index(self.target)) +
                ');\n')
        else:
            print '\tWARNING: no connector object provided.'
            return ''
        

