from lxml import etree 
from PyQt4 import QtGui
from PyQt4.QtCore import QObject

class Repository(QObject):
    
    def __init__(self):
        super(Repository, self).__init__()
        
        self._neuron_defs = {}
        self._synapse_defs = {}

    def save(self):
        """
        Save the current repository state as xml file.
        """
        root = etree.Element( 'database' )
        
        #
        # save neurons
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

        syn_tree = etree.SubElement( root, 'synapses')
        for name, code in self._synapse_defs.iteritems():
            syn_tag = etree.SubElement( syn_tree, str(name))
            
            syn_name = etree.SubElement( syn_tag, 'name')
            syn_name.text = str(name)

            syn_code = etree.SubElement( syn_tag, 'code')
            i = 0
            for line in str(code).split('\n'):
                syn_tag = etree.SubElement( syn_code , 'line'+str(i)  )
                syn_tag.text = line
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

        syn_root = doc.findall('synapses') # find neuron root node
        if syn_root != []:
            for syn in syn_root[0].getchildren():
                syn_name = syn.find('name').text
                syn_code = ''
    
                for line in syn.find('code').getchildren():
                    if line.text != None:
                        syn_code += str(line.text)+'\n'
                    else:
                        syn_code += '\n'
    
                self._synapse_defs[syn_name] = syn_code
    
    def add_object(self, type, name, code):
        if type == 'neuron':
            self._neuron_defs[str(name)] = code
        else:
            self._synapse_defs[str(name)] = code

    def update_object(self, type, name, code):
        if type == 'neuron':
            self._neuron_defs[str(name)] = code
        else:
            self._synapse_defs[str(name)] = code
    
    def get_entries(self, type):
        if type == 'neuron':
            return self._neuron_defs.keys()
        else:
            return self._synapse_defs.keys()
    
    def get_object(self, type, name):
        if type == 'neuron':
            return self._neuron_defs[str(name)]
        else:
            return self._synapse_defs[str(name)]
          
    def entry_contained(self, name):
        exist = (name in self._neuron_defs.keys()) or \
                (name in self._synapse_defs.keys())
                
        return exist
    
