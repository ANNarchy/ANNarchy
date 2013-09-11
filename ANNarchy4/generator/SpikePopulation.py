from Population import Population

class SpikePopulation(Population):
    """
    Population generator class
    """
    
    def __init__(self, population):

        Population.__init__(self, population)
        
    def generate(self):
        
        self._update_neuron_variables()
        
        #   generate files
        with open(self.header, mode = 'w') as w_file:
            w_file.write(self._generate_header())

        with open(self.body, mode = 'w') as w_file:
            w_file.write(self._generate_body())

        with open(self.pyx, mode = 'w') as w_file:
            w_file.write(self._generate_pyx())

    def _generate_header(self):
        return ''
        
    def _generate_body(self):
        return ''

    def _generate_pyx(self):
        return ''        