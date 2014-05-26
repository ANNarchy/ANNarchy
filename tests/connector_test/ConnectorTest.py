from ANNarchy import *

try:
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
except:
    print 'PyQtGraph is not installed, can not visualize the network.'
    exit(0)

Simple = RateNeuron(   
parameters="""
    tau = 1.0
    baseline = 0.0
""",
equations="""
    noise = Uniform(0,1)
    tau * dmp / dt + mp = baseline + noise
    r = pos(mp)
""" 
)

Simple2 = RateSynapse(
psp = """ w * (1.0-pre.r) """                      
)

OneDimIn = Population((10,1), Simple)
OneDimOut = Population((1,3), Simple)

In = Population((20,20), Simple)
Middle = Population((20, 20), Simple)
Out = Population((5,5), Simple)
 
testOne2One = Projection( 
    pre = In, 
    post = Middle, 
    target = 'one',
)
testOne2One.connect_one_to_one( weights=1.0 )

testAll2All = Projection( 
    pre = Middle, 
    post = Out, 
    target = 'all',
).connect_all_to_all( weights = Uniform(0.0, 1.0), delays = 0.0 )

testGaussian = Projection(
    pre = In, 
    post = Out, 
    target = 'gauss'
).connect_gaussian( sigma=0.7, amp=0.1 )
 
testDog = Projection(
    pre = In, 
    post = Out, 
    target = 'dog', 
).connect_dog(
    amp_pos=0.2, 
    sigma_pos=0.1, 
    amp_neg=0.1, 
    sigma_neg=0.7
)

testFixedProbability = Projection(
    pre = In, 
    post = Out, 
    target = 'fixed_prob', 
).connect_fixed_probability(
    weights = Uniform(0.0, 1.0),
    probability = 0.1
)

testFixedNumberPre = Projection(
    pre = OneDimIn, 
    post = OneDimOut, 
    target = 'fixed_pre', 
).connect_fixed_number_pre(
    weights = Uniform(0.0, 1.0),
    number = 3
)

testFixedNumberPost = Projection(
    pre = OneDimIn, 
    post = OneDimOut, 
    target = 'fixed_post', 
).connect_fixed_number_post(
    weights = Uniform(0.0, 1.0),
    number = 2
)

#===============================================================================
# def stochastic_pattern(pre, post, weight, probability):
#     
#     synapse_dict = {}
#     
#     for post_h in xrange(post.height):
#         for post_w in xrange(post.width):
#             post_rank = post.rank_from_coordinates( (post_w, post_h) )
#             
#             for pre_h in xrange(pre.height):
#                 for pre_w in xrange(pre.width):
#                     pre_rank = pre.rank_from_coordinates( (pre_w, pre_h) )
#                     if np.random.random() < probability:
#                         synapse_dict[(pre_rank, post_rank)] = { 'w': weight, 'd': 2.0 }
#   
#     return synapse_dict
# 
# testUserPattern = Projection(
#     pre = Middle, 
#     post = Out, 
#     target = 'user' 
# ).connect_with_func(method=stochastic_pattern, weight=1.0, probability=0.3)   
# 
# def stochastic_pattern2(pre, post, weight, probability):
#     
#     synapse_dict = []
#     app = synapse_dict.append
#     
#     for post_h in xrange(post.height):
#         for post_w in xrange(post.width):
#             post_rank = post.rank_from_coordinates( (post_w, post_h) )
#             
#             for pre_h in xrange(pre.height):
#                 for pre_w in xrange(pre.width):
#                     pre_rank = pre.rank_from_coordinates( (pre_w, pre_h) )
#                     if np.random.random() < probability:
#                         app( (pre_rank, post_rank, weight, 2.0,) )
#   
#     return synapse_dict
#     
# testUserPattern2 = Projection(
#     pre = Middle, 
#     post = Out, 
#     target = 'user2' 
# ).connect_with_func(method=stochastic_pattern2, weight=1.0, probability=0.3)   
#===============================================================================

class Viewer(object):
    " Class to visualize the network activity using PyQtGraph."
    
    def __init__(self):
    
        app = pg.mkQApp()
        self.win = pg.GraphicsWindow(title="Bar learning")
        self.win.resize(800,800)
        
        self.win.addLabel("One2One")
        self.win.addLabel("All2All")
        self.win.addLabel("Gaussian")
        self.win.addLabel("DoG")
        self.win.nextRow()

        box = self.win.addViewBox(lockAspect=True)
        self.one2oneVis = pg.ImageItem()
        box.addItem(self.one2oneVis)
        
        box = self.win.addViewBox(lockAspect=True)
        self.all2allVis = pg.ImageItem()
        box.addItem(self.all2allVis)

        box = self.win.addViewBox(lockAspect=True)
        self.gaussianVis = pg.ImageItem()
        box.addItem(self.gaussianVis)

        box = self.win.addViewBox(lockAspect=True)
        self.dogVis = pg.ImageItem()
        box.addItem(self.dogVis)
        
        self.win.nextRow()

        self.win.addLabel("Fixed probability")
        self.win.addLabel("Fixed number post")
        self.win.addLabel("Fixed number pre")
        
        self.win.nextRow()
        
        box = self.win.addViewBox(lockAspect=True)
        self.fixedProb = pg.ImageItem()
        box.addItem(self.fixedProb)
        
        box = self.win.addViewBox(lockAspect=True)
        self.fixedPostVis = pg.ImageItem()
        box.addItem(self.fixedPostVis)

        box = self.win.addViewBox(lockAspect=True)
        self.fixedPreVis = pg.ImageItem()
        box.addItem(self.fixedPreVis)
        
        self.win.show()
        

    def update(self):
        # first row
        self.one2oneVis.setImage(testOne2One.receptive_fields())
        self.all2allVis.setImage(testAll2All.receptive_fields())
        self.gaussianVis.setImage(testGaussian.receptive_fields())
        self.dogVis.setImage(testDog.receptive_fields())
        
        # second row
        self.fixedProb.setImage(testFixedProbability.receptive_fields())
        self.fixedPostVis.setImage(testFixedNumberPost.receptive_fields())
        self.fixedPreVis.setImage(testFixedNumberPre.receptive_fields())
                
        # Listen to mouse/keyboard events
        QtGui.QApplication.processEvents()
        
    def run(self):
        
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(0)  
        QtGui.QApplication.instance().exec_()     
        
if __name__=='__main__':

    compile()

    # Create the GUI
    view = Viewer()
    
    # Start the simulation
    view.run()
#===============================================================================
# visOne2One = Visualization( [ { 'proj': testOne2One, 'var': 'w', 'min': 0.0, 'max': 1.0, 'title': 'weights one2one'} ] )
# visOne2One.render()
#    
# visAll2All = Visualization( [ { 'proj': testAll2All, 'var': 'w', 'min': 0.0, 'max': 1.0, 'title': 'weights all2all'} ] )
# visAll2All.render()
# 
# visGaussian = Visualization( [ { 'proj': testGaussian, 'var': 'w', 'min': 0.0, 'max': 0.2, 'title': 'weights gaussian'} ] )
# visGaussian.render()
#  
# visDog = Visualization( [ { 'proj': testDog, 'var': 'w', 'min': -0.1, 'max': 0.1, 'title': 'weights difference of gaussian'} ] )
# visDog.render()
# 
# visFixedPropPattern = Visualization( [ { 'proj': testFixedPropability, 'var': 'w', 'min': 0.0, 'max': 1.0, 'title': 'fixed propability'} ] )
# visFixedPropPattern.render()
# 
# visFixedPrePattern = Visualization( [ { 'proj': testFixedNumberPre, 'var': 'w', 'min': 0.0, 'max': 1.0, 'title': 'fixed number pre'} ] )
# visFixedPrePattern.render()
# 
# visFixedPostPattern = Visualization( [ { 'proj': testFixedNumberPost, 'var': 'w', 'min': 0.0, 'max': 1.0, 'title': 'fixed number post'} ] )
# visFixedPostPattern.render()
# 
# visTestUserPattern = Visualization( [ { 'proj': testUserPattern, 'var': 'w', 'min': 0.0, 'max': 1.0, 'title': 'user defined pattern'} ] )
# visTestUserPattern.render()
# 
# visTestUserPattern2 = Visualization( [ { 'proj': testUserPattern2, 'var': 'w', 'min': 0.0, 'max': 1.0, 'title': 'user defined pattern2'} ] )
# visTestUserPattern2.render()
#===============================================================================

raw_input()
