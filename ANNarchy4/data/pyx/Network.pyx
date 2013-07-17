cdef extern from "../build/Network.h":
	cdef cppclass Network:
		Network()

		int getTime()

		void run(int nbSteps)

cdef extern from "../build/Network.h" namespace "Network":
	cdef Network* instance()

cdef class pyNetwork:

	cdef Network* cInstance

	def __cinit__(self):
		self.cInstance = instance()

	def Time(self):
		return self.cInstance.getTime()

	def Run(self, int nbSteps):
		self.cInstance.run(nbSteps)
