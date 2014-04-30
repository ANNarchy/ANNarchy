#
#    ANNarchy-4 nao experiment - just integration of Nao functions / link
#    to ANNarchy is through user functions.
#
#    author: Helge Uelo Dinkelbach
#
#    Hints:
#
#        setup second ethernet connection:
#
#        sudo ifconfig eth1 192.168.1.1 up
#
from ANNarchy4 import *
from pylab import ion, imshow, pause

def simulate_sth(nao):
    print 'Running the simulation'

    ion()

    im = nao.get_image()
    data = np.asarray(im) # PIL to np.array
    image = imshow( data )
    
    for trial in range(500):
        
        im = nao.get_image()
        data = np.asarray(im) # PIL to np.array
        
        image.set_data(data)
        pause(0.01)        
        print 'trial', trial

if __name__ == '__main__':
    
    #
    # NAO experiments
    nao = None
    
    IP = "192.168.1.12"  # Replace here with your NaoQi's IP address.
    PORT = 9559
    nao = Nao(IP, PORT)     
    
    simulate_sth(nao=nao)
    
    nao.disconnect()
        
