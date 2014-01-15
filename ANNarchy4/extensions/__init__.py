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
try:
    from Nao import Nao
except ImportError:
    pass

try:
    from Profile import Profile
except ImportError:
    pass

def check_extensions():
    Global._debug('checking for extensions.')
    
    try:
        from Nao import Nao
        
    except ImportError:
        pass
    
    else:
        Global._debug('... Nao module available ...')
    
    try:
        from Profile import Profile
        
    except ImportError:
        pass
    
    else:
        Global._debug('... Profile module available ...')
