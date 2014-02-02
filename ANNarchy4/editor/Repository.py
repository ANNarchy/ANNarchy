from lxml import etree 
from PyQt4 import QtGui
from PyQt4.QtCore import QObject

from GLObjects import Quad2d, Point2d

class Repository(QObject):
    
    def __init__(self):
        super(Repository, self).__init__()
        
        self._neuron_defs = {}
        self._synapse_defs = {}
        self._network_defs = {}
        
        #
        # function calls
        self._add_obj = { 
         'neuron' : self._add_neuron, 
         'synapse' : self._add_synapse,
         'network' : self._add_network
        }

        self._update_obj = { 
         'neuron' : self._update_neuron, 
         'synapse' : self._update_synapse,
         'network' : self._update_network
        }
        
        self._get_entries = {
         'neuron' : self._neuron_entries, 
         'synapse' : self._synapse_entries,
         'network' : self._network_entries
        }

        self._get_obj = {
         'neuron' : self._get_neuron, 
         'synapse' : self._get_synapse,
         'network' : self._get_network
        }

    def save(self):
        """
        Save the current repository state as xml file.
        """
        root = etree.Element( 'database' )
        
        #
        # save neurons
        neur_tree = etree.SubElement( root, 'neurons')
        for name, data in self._neuron_defs.iteritems():
            neur_tag = etree.SubElement( neur_tree, str(name))
            
            neur_name = etree.SubElement( neur_tag, 'name')
            neur_name.text = str(name)

            neur_data = etree.SubElement( neur_tag, 'code')
            i = 0
            for line in str(data).split('\n'):
                if len(line) > 0:
                    data_tag = etree.SubElement( neur_data , 'line'+str(i)  )
                    data_tag.text = line
                    i+=1                  

        #
        # save neurons
        syn_tree = etree.SubElement( root, 'synapses')
        for name, data in self._synapse_defs.iteritems():
            syn_tag = etree.SubElement( syn_tree, str(name))
            
            syn_name = etree.SubElement( syn_tag, 'name')
            syn_name.text = str(name)

            syn_data = etree.SubElement( syn_tag, 'code')
            i = 0
            for line in str(data).split('\n'):
                if len(line) > 0:
                    syn_tag = etree.SubElement( syn_data , 'line'+str(i)  )
                    syn_tag.text = line
                    i+=1                  

        #
        # save network
        net_tree = etree.SubElement( root, 'networks')
        for name, data in self._network_defs.iteritems():
            net_tag = etree.SubElement( net_tree, str(name))
            net_name = etree.SubElement( net_tag, 'name')
            net_name.text = str(name)
            
            populations = data['pop_data']
            net_pop_data = etree.SubElement( net_tag, 'populations')
            for id, pop in populations.iteritems():
                pop_data = etree.SubElement( net_pop_data, 'pop'+str(id))
                
                pop_name = etree.SubElement( pop_data, 'name')
                pop_name.text = str(pop['name'])

                pop_type = etree.SubElement( pop_data, 'type')
                pop_type.text = str(pop['type'])
                
                pop_geo = etree.SubElement( pop_data, 'geometry')
                pop_geo.text = str(pop['geometry'])
                
                pop_coords = etree.SubElement( pop_data, 'coords')

                pop_p1 = etree.SubElement( pop_coords, 'p1')
                pop_p1.text = str(pop['coords'].p1)
                pop_p2 = etree.SubElement( pop_coords, 'p2')
                pop_p2.text = str(pop['coords'].p2)
                pop_p3 = etree.SubElement( pop_coords, 'p3')
                pop_p3.text = str(pop['coords'].p3)
                pop_p4 = etree.SubElement( pop_coords, 'p4')
                pop_p4.text = str(pop['coords'].p4)

            projections = data['proj_data']
            net_proj_data = etree.SubElement( net_tag, 'projections')
            for id, proj in projections.iteritems():
                proj_data = etree.SubElement( net_proj_data, 'proj'+str(id))
                
                proj_pre = etree.SubElement( proj_data, 'pre')
                proj_pre.text = str(proj['pre'])

                proj_post = etree.SubElement( proj_data, 'post')
                proj_post.text = str(proj['post'])
                
                proj_target = etree.SubElement( proj_data, 'target')
                proj_target.text = str(proj['target'])
                
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
        if neur_root != []:
            for neur in neur_root[0].getchildren():
                neur_name = neur.find('name').text
                neur_data = ''
    
                for line in neur.find('code').getchildren():
                    if line.text != None:
                        neur_data += str(line.text)+'\n'
                    else:
                        neur_data += '\n'
    
                self._neuron_defs[neur_name] = neur_data

        syn_root = doc.findall('synapses') # find synapse root node
        if syn_root != []:
            for syn in syn_root[0].getchildren():
                syn_name = syn.find('name').text
                syn_data = ''
    
                for line in syn.find('code').getchildren():
                    if line.text != None:
                        syn_data += str(line.text)+'\n'
                    else:
                        syn_data += '\n'
    
                self._synapse_defs[syn_name] = syn_data

        net_root = doc.findall('networks') # find network root node
        if net_root != []:
            for net in net_root[0].getchildren():
                net_name = net.find('name').text

                self._network_defs = { net_name : { 'pop_data': {}, 'proj_data': {} } }
                pop_data = net.find('populations')
                
                i = 0
                for pop in pop_data.getchildren():
                    name = pop.find('name').text
                    
                    type = pop.find('type').text
                    
                    geometry = eval(pop.find('geometry').text) # I know, eval is not so good, but I dont know an other conversion 
                    
                    points = []
                    for p in pop.find('coords').getchildren():
                        tmp = eval(p.text)
                        points.append(Point2d(tmp[0], tmp[1]))
                    quad = Quad2d().from_p(points[0],
                                           points[1],
                                           points[2],
                                           points[3])
                        
                    self._network_defs[net_name]['pop_data'].update( { i: {'name': name, 
                                                               'geometry': geometry, 
                                                               'coords': quad, 
                                                               'type' : type
                                                               } } )
                    i += 1

                proj_data = net.find('projections')
                i = 0
                for proj in proj_data.getchildren():
                    pre = int(proj.find('pre').text)
                    
                    post = int(proj.find('post').text)
                    
                    target = str(proj.find('target').text)
                        
                    self._network_defs[net_name]['proj_data'].update( { i: {'pre': pre, 
                                                               'post': post, 
                                                               'target' : target
                                                               } } )
                    i += 1

    def add_object(self, type, name, data):
        try:
            self._add_obj[type](name, data)
        except KeyError:
            print 'Update object: type', type, 'not known.'

    def update_object(self, type, name, data):
        try:
            self._update_obj[type](name, data)
        except KeyError:
            print 'Update object: type', type, 'not known.'
    
    def get_entries(self, type):
        try:
            return self._get_entries[type]()
        except KeyError:
            print 'Get entries: type', type, 'not known.'
    
    def get_object(self, type, name):
        try:
            return self._get_obj[type](name)
        except KeyError:
            print 'Get entries: type', type, 'not known.'
          
    def entry_contained(self, name):
        exist = (name in self._neuron_defs.keys()) or \
                (name in self._synapse_defs.keys())
                
        return exist

    #######################################################
    #
    #    Object handling
    #
    #######################################################
    def _add_neuron(self, name, data):
        self._neuron_defs[str(name)] = data
        
    def _add_synapse(self, name, data):
        self._synapse_defs[str(name)] = data
        
    def _add_network(self, name, data):
        self._network_defs[str(name)] = data

    def _update_neuron(self, name, data):
        self._neuron_defs[str(name)] = data
        
    def _update_synapse(self, name, data):
        self._synapse_defs[str(name)] = data
        
    def _update_network(self, name, data):
        print data

    def _neuron_entries(self):
        return self._neuron_defs.keys()
        
    def _synapse_entries(self):
        return self._synapse_defs.keys()
        
    def _network_entries(self):
        return self._network_defs.keys()
        
    def _get_neuron(self, name):
        return self._neuron_defs[str(name)]
        
    def _get_synapse(self, name):
        return self._synapse_defs[str(name)]
        
    def _get_network(self, name):
        return self._network_defs[str(name)]
