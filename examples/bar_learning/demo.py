from ANNarchy4 import *
import math

def sim():

    #Global._visualizer.set_color(0, 0, 0.0, 0.6, 0.0)
    #Global._visualizer.set_color(1, 0, 0.6, 0.0, 0.0)
    
    for t in range(10000):
        
        x = [ i for i in range(360)]
        y = [ math.cos( float(i+(t % 360)) * math.pi / 180.0) for i in range(360) ]
        y2 = [ math.sin( float(i+(t % 360)) * math.pi / 180.0) for i in range(360) ]
        
        Global._visualizer.set_data(0, 0, x, y)
        Global._visualizer.set_data(1, 0, x, y2)
        
        simulate(1)
        render()

ANNarchyEditor( sim )