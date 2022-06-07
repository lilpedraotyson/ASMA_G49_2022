class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    '''def prints(self):
        print("Node")
        print(self.parent)
        print(self.position)
        #print(self.g)
        #print(self.h)
        #print(self.f)'''
