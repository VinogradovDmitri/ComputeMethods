class Segment(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"[{self.left:.12f} ; {self.right:.12f}]"

    def len(self):
        return abs(self.right - self.left)

    def center(self):
        return (self.left + self.right) / 2

    def copy(self):
        return Segment(self.left, self.right)
