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

try:
    from Nao import Nao
    
except ImportError:
    pass

else:
    print '... Nao module inited ...'

