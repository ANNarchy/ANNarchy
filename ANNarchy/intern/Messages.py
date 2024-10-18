"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import sys

from ANNarchy.intern import ConfigManagement

class ANNarchyException(Exception):
    """
    Custom exception that can be catched in some cases (IO) instead of quitting.
    """
    def __init__(self, message):
        super(ANNarchyException, self).__init__(message)

        # # Print the error message
        # print('ERROR: ' + message)

        # # Print the trace
        # # tb = traceback.print_stack()
        # tb = traceback.format_stack()
        # for line in tb:
        #     if not '/ANNarchy/core/' in line and \
        #        not '/ANNarchy/parser/' in line and \
        #        not '/ANNarchy/generator/' in line :
        #         print(line)

class CodeGeneratorException(Exception):
    def __init__(self, msg):
        print("An error in the code generation occured:")
        print(msg)

class InvalidConfiguration(Exception):
    def __init__(self, msg):
        print("The configuration you requested is not implemented in ANNarchy:")
        print(msg)

def _print(*var_text, end="\n", flush=False):
    """
    Prints a message to standard out.
    """
    text = ''
    for var in var_text:
        text += str(var) + ' '

    if sys.version_info.major == 3:
        print(text, end=end, flush=flush)
    else:
        print(text)

def _debug(*var_text):
    """
    Prints a message to standard out, if verbose mode set True.
    """
    if not ConfigManagement.get_global_config('verbose'):
        return

    text = ''
    for var in var_text:
        text += str(var) + ' '
    print(text)

def _warning(*var_text):
    """
    Prints a warning message to standard out. Can be suppressed by configuration.
    """
    text = 'WARNING: '
    for var in var_text:
        text += str(var) + ' '
    if not ConfigManagement.get_global_config('suppress_warnings'):
        print(text)

def _info(*var_text):
    """
    Prints a information message to standard out. Can be suppressed by configuration.
    """
    text = 'INFO: '
    for var in var_text:
        text += str(var) + ' '
    if not ConfigManagement.get_global_config('suppress_warnings'):
        print(text)

def _error(*var_text, **args):
    """
    Prints an error message to standard out and exits.

    When passing exit=False, the program will not exit.
    """
    text = ''
    for var in var_text:
        text += str(var) + ' '

    exit = False
    if 'exit' in args.keys():
        if args['exit']:
            exit = True
    else:
        exit = True

    if exit:
        raise ANNarchyException(text)
    else:
        print('ERROR:' + text)
