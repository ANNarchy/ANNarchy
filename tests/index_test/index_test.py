from ANNarchy import *

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

def test_indices():
    test_1d = Population( 100, Simple )
    for i in range(test_1d.size):
        c = test_1d.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_1d._geometry )
        if c != c2:
            print c, c2
        
        r = test_1d.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_1d._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_1d_1 passed'        
    
    test_2d_1 = Population((100, 100), Simple)
    for i in range(test_2d_1.size):
        c = test_2d_1.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_2d_1._geometry )
        if c != c2:
            print c, c2
        
        r = test_2d_1.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_2d_1._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_2d_1 passed'        
     
    test_2d_2 = Population((100, 3), Simple)
    for i in range(test_2d_2.size):
        c = test_2d_2.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_2d_2._geometry )
        if c != c2:
            print c, c2
    
        r = test_2d_2.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_2d_2._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_2d_2 passed'        
     
    test_2d_3 = Population((3, 100), Simple)
    for i in range(test_2d_3.size):
        c = test_2d_3.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_2d_3._geometry )
        if c != c2:
            print c, c2
    
        r = test_2d_3.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_2d_3._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_2d_3 passed'        
    
    test_3d_1 = Population((5, 4, 3), Simple)
    for i in range(test_3d_1.size):
        c = test_3d_1.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_3d_1._geometry )
        if c != c2:
            print c, c2
    
        r = test_3d_1.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_3d_1._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_3d_1 passed'
    
    
    test_3d_2 = Population((3, 4, 5), Simple)
    for i in range(test_3d_2.size):
        c = test_3d_2.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_3d_2._geometry )
        if c != c2:
            print c, c2
    
        r = test_3d_2.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_3d_2._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_3d_2 passed'
     
    test_3d_3 = Population((3, 7, 5), Simple)
    for i in range(test_3d_3.size):
        c = test_3d_3.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_3d_3._geometry )
        if c != c2:
            print c, c2
    
        r = test_3d_3.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_3d_3._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_3d_3 passed'
     
    test_3d_4 = Population((1, 25, 25), Simple)
    for i in range(test_3d_4.size):
        c = test_3d_4.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_3d_4._geometry )
        if c != c2:
            print c, c2
    
        r = test_3d_4.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_3d_4._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_3d_4 passed'
     
    test_3d_5 = Population((50, 50, 50), Simple)
    for i in range(test_3d_5.size):
        c = test_3d_5.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_3d_5._geometry )
        if c != c2:
            print c, c2
    
        r = test_3d_5.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_3d_5._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_3d_5 passed'
    
    test_4d = Population((4, 5, 2, 6), Simple)
    for i in range(test_4d.size):
        c = test_4d.coordinates_from_rank(i)
        c2 = np.unravel_index( i, test_4d._geometry )
        if c != c2:
            print c, c2
    
        r = test_4d.rank_from_coordinates(c)
        r2 = np.ravel_multi_index(c, test_4d._geometry )
        if r - r2 != 0:
            print r, r2, c
    print 'test_4d passed'
    
#
# run test
test_indices()
