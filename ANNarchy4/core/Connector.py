"""
    
    Connector.py
    
    This file is part of ANNarchy.
    
    Copyright (C) 2013-2016  Julien Vitay <julien.vitay@gmail.com>,
    Helge Uelo Dinkelbach <helge.dinkelbach@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ANNarchy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
"""
from .Dendrite import Dendrite
from math import exp

def one2one(pre, post, weights, delays=0.0):
    synapses = {}
    
    for pre_neur in xrange(pre.size):
        try:
            w = weights.get_value()
        except:
            w = weights
            
        try:
            d = delays.get_value()
        except:
            d = delays
        synapses[(pre_neur, pre_neur)] = { 'w': w, 'd': d }
            
    return synapses

def all2all(pre, post, weights, delays=0.0, allow_self_connections=False):
    allow_self_connections = (pre!=post) and not allow_self_connections
    synapses = {}

    for post_neur in xrange(post.size):
        for pre_neur in xrange(pre.size):
            if (pre_neur == post_neur) and not allow_self_connections:
                continue
            
            try:
                w = weights.get_value()
            except:
                w = weights
                
            try:
                d = delays.get_value()
            except:
                d = delays

            synapses[(post_neur, pre_neur)] = { 'w': w, 'd': d }
            
    return synapses

def gaussian(pre, post, sigma, amp, delays=0.0, limit=0.01):
    def compDist(pre, post):
        res = 0.0

        for i in range(len(pre)):
            res = res + (pre[i]-post[i])*(pre[i]-post[i]);

        return res
    
    allow_self_connections = (pre!=post) and not allow_self_connections
    synapses = {}
    
    for post_neur in xrange(post.size):
        normPost = post.normalized_coordinates_from_rank(post_neur)
        
        for pre_neur in range(pre.size):
            if (pre_neur == post_neur) and not allow_self_connections:
                continue

            normPre = pre.normalized_coordinates_from_rank(pre_neur)

            dist = compDist(normPre, normPost)
            
            value = amp * exp(-dist/2.0/sigma/sigma)
            if (abs(value) > limit * abs(amp)):
                    
                try:
                    d = delays.get_value()
                except:
                    d = delays
                synapses[(post_neur, pre_neur)] = { 'w': value, 'd': d }
                    
    return synapses
    
def dog(pre, post, sigma_pos, sigma_neg, amp_pos, amp_neg, delays=0.0, limit=0.01):
    def compDist(pre, post):
        res = 0.0

        for i in range(len(pre)):
            res = res + (pre[i]-post[i])*(pre[i]-post[i]);

        return res
    
    allow_self_connections = (pre!=post) and not allow_self_connections
    synapses = {}
    
    for post_neur in xrange(post.size):
        normPost = post.normalized_coordinates_from_rank(post_neur)
        
        for pre_neur in range(pre.size):
            if (pre_neur == post_neur) and not allow_self_connections:
                continue

            normPre = pre.normalized_coordinates_from_rank(pre_neur)

            dist = compDist(normPre, normPost)

            value = amp_pos * exp(-dist/2.0/sigma_pos/sigma_pos) - amp_neg * exp(-dist/2.0/sigma_neg/sigma_neg)
            if ( abs(value) > limit * abs( amp_pos - amp_neg ) ):
                    
                try:
                    d = delays.get_value()
                except:
                    d = delays
                synapses[(post_neur, pre_neur)] = { 'w': value, 'd': d }
                    
    return synapses    