from NeuralField import *
import numpy as np
cimport numpy as np
import math

def run(InputPop, FocusPop, proj):

    cdef int w = InputPop.geometry[0]
    cdef int h = InputPop.geometry[1]
    
    cdef float angle = 0.0
    cdef float radius = 0.5
    cdef float sigma = 2.0
    
    
    cdef int i, idx
    cdef float cw, ch, dist, value

    cdef np.ndarray data = np.zeros(w*h)
    
    vis = Visualization( [ { 'pop': InputPop, 'var': 'rate' }, 
                           { 'pop': FocusPop, 'var': 'rate','min': 0.0, 'max': 1.0 },
                           { 'proj': proj, 'var': 'value', 
                             'min': -0.1, 'max': 0.1, 'title': 'Receptive fields'} ]
                         
                       )
    """
    # profile instance
    profiler = Profile()

    #
    # setup the test
    thread_count = [ 6-x for x in xrange(6) ] 
    num_steps = 30
    
    log_sum = profiler.init_log(thread_count, num_steps, 'NF_sum_')
    log_net = profiler.init_log(thread_count, num_steps, 'NF_net_')
    log_step = profiler.init_log(thread_count, num_steps, 'NF_meta_step_')
    
    #
    # pre setup
    diff_runs = []
    index = [ x for x in xrange(num_steps) ]

    for test in range(len(thread_count)):
        diff_runs.append( [0 for x in range(num_steps) ] )

        profiler.set_num_threads(thread_count[test])   
    """
    num_steps = 5000
    
    for step in xrange (num_steps):
        angle += 1.0/float(num_steps)

        cw = w / 2.0 * ( 1.0 + radius * np.cos(2 * math.pi * angle ) )
        ch = h / 2.0 * ( 1.0 + radius * np.sin(2 * math.pi * angle ) )

        for x in xrange(w):
            for y in xrange(h):
                dist = (x-cw)**2 + (y-ch)**2
                value = 0.5 * np.exp(-dist/2.0/sigma**2)
                idx = x+y*w
                data[idx] = value

        InputPop.baseline = data

        simulate(1)
        if step%250 == 0:
            vis.render()

        #save('tmp.mat')
        """            
            log_net[thread_count[test], step] = profiler.average_net( step, (step+1) )
            log_sum[thread_count[test], step] = profiler.average_sum("Population1", step, (step+1))
            log_step[thread_count[test], step] = profiler.average_step("Population1", step, (step+1))
            
            
            diff_runs[test][step] = profiler.average_sum("Population1", step, (step+1))
            #print profiler.average_sum("Population1", trial*trial_dur, (trial+1)*trial_dur)

        profiler.reset_timer()
                
    print 'results:\n'
    for test in range(len(thread_count)):
        print '\ttest',test,'with',thread_count[test],'thread(s)'
        print '\t',diff_runs[test]
        print '\n'

    print diff_runs
    print 'all simulation finished.'

    log_net.save_to_file()
    log_sum.save_to_file()
    log_step.save_to_file()
    """
    print 'done'