import sys

class ANNarchyException(Exception):
    """
    Custom exception that can be catched in some cases (IO) instead of quitting.
    """
    def __init__(self, message, exit):
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
        sys.exit(self)

class InvalidConfiguration(Exception):
    def __init__(self, msg):
        print("The configuration you requested is not implemented in ANNarchy.")
        sys.exit(self)

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
    from ANNarchy.core.Global import config
    
    if not config['verbose']:
        return

    text = ''
    for var in var_text:
        text += str(var) + ' '
    print(text)

def _warning(*var_text):
    """
    Prints a warning message to standard out. Can be suppressed by configuration.
    """
    from ANNarchy.core.Global import config

    text = 'WARNING: '
    for var in var_text:
        text += str(var) + ' '
    if not config['suppress_warnings']:
        print(text)

def _info(*var_text):
    """
    Prints a information message to standard out. Can be suppressed by configuration.
    """
    from ANNarchy.core.Global import config

    text = 'INFO: '
    for var in var_text:
        text += str(var) + ' '
    if not config['suppress_warnings']:
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
        raise ANNarchyException(text, exit)
    else:
        print('ERROR:' + text)
