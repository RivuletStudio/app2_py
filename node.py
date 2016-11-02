class spatial:
    def __init__(self, w, h, d):
        self.w = w
        self.h = h
        self.d = d
        self.parent = None
        self.index = None

    def set_parent(self,parent):
    	self.parent = parent