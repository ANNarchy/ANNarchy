
class Variable:
    def __init__(self,  **keyValueArgs):

        self.eq = None
        self.init = None

        for key in keyValueArgs:

            if key == 'eq':
                self.eq = keyValueArgs[key]
            elif key=='init':
                self.init = keyValueArgs[key]
            else:
                print 'unknown key: '+key

    