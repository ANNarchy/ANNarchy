#   extend this file to add new modules to ANNarchy4
#
#   e. g.
#
#       try:
#
#           from MyModule import MyModuleClass
#
#       except ImportError:
#           print 'Module not found.'
from ANNarchy4.core import Global

if Global.config['verbose']:
    print 'checking for extensions.'

try:
    from Nao import Nao
    
except ImportError:
    pass

else:
    if Global.config['verbose']:
        print '... Nao module imported ...'
