from lxml import etree 
from PyQt4 import QtGui

class Repository(object):
    
    def __init__(self):
        self._neuron_defs = {}

    def save(self):
        """
        Save the current repository state as xml file.
        """
        root = etree.Element( 'database' )
        
        neur_tree = etree.SubElement( root, 'neurons')
        for name, code in self._neuron_defs.iteritems():
            neur_tag = etree.SubElement( neur_tree, str(name))
            
            neur_name = etree.SubElement( neur_tag, 'name')
            neur_name.text = str(name)

            neur_code = etree.SubElement( neur_tag, 'code')
            i = 0
            for line in str(code).split('\n'):
                code_tag = etree.SubElement( neur_code , 'line'+str(i)  )
                code_tag.text = line
                i+=1                  
        #
        # save the data to file
        fname = open('./neur_rep.xml', 'w')
        fname.write(etree.tostring(root, pretty_print=True))
        fname.close()        
            
    def load(self):
        try:
            doc = etree.parse('./neur_rep.xml')
            
        except IOError:
            print('no neuron definitions found.')
            return
        
        neur_root = doc.findall('neurons') # find neuron root node
        for neur in neur_root[0].getchildren():
            neur_name = neur.find('name').text
            neur_code = ''

            for line in neur.find('code').getchildren():
                if line.text != None:
                    neur_code += str(line.text)+'\n'
                else:
                    neur_code += '\n'

            self._neuron_defs[neur_name] = neur_code
    
    def add_neuron(self, name, code):
        self._neuron_defs[str(name)] = code

    def update_neuron(self, name, code):
        self._neuron_defs[str(name)] = code
    
    def get_neuron_entries(self):
        return self._neuron_defs.keys()
    
    def get_neuron(self, name):
        return self._neuron_defs[str(name)]
     
    def entry_contained(self, name):
        return name in self._neuron_defs.keys()