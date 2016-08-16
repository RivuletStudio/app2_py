import numpy as np

class HeapElem():
	def __init__(self,_ind,_value):
		self.heap_id = -1
		self.img_ind = _ind
		self.value = _value

class HeapElemX(HeapElem):
	def __init__(self,_ind,_value):
		HeapElem.__init__(self,_ind,_value)
		self.prev_ind = -1

class BasicHeap():
	def __init__(self):
		self.elems = []

	def delete_min():
		if (not elems):
			return 0
		min_elem = elems[0]

