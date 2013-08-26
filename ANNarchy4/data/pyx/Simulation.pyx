# cython: embedsignature=True

from libcpp.vector cimport vector

# import c++ class
cdef extern from "build/ANNarchy.h":
	cdef cppclass ANNarchy:
		ANNarchy()

		void run(int number) nogil

		vector[float] getRates(int populationID)

# wrapper c++ to python
cdef class Simulation:

	cdef ANNarchy *annarchy

	def __cinit__(self):
		self.annarchy = new ANNarchy()

	def run(self, int number):
		
		self.annarchy.run(number)

	def getRates(self, int populationID):
		
		tmp = self.annarchy.getRates(0)
		
		return list(tmp)
