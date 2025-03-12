import numpy as np


class DynamicNP:
    array = None
    currlen = None
    maxlen = None

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.array = np.zeros((self.maxlen, 3))
        self.currlen = 0

    def addPoints(self, np_points):
        arrlen = len(np_points)

        if self.currlen + arrlen < self.maxlen:
            self.array[self.currlen:self.currlen + arrlen] = np_points
            self.currlen += arrlen
        else:
            # print("realloc")
            prev_points = self.array[0:self.currlen]

            if self.currlen + arrlen >= self.maxlen * 2: # so many new points that it dont fit in new arraay
                self.maxlen = (self.currlen + arrlen) * 2
            else:
                self.maxlen = self.maxlen * 2

            self.array = np.zeros((self.maxlen, 3))
            self.array[0:self.currlen] = prev_points

            try:
                self.array[self.currlen:self.currlen + arrlen] = np_points
            except ValueError:
                a = 0

            self.currlen += arrlen

    def getPoints(self):
        return self.array[0:self.currlen]

    def getCount(self):
        return self.currlen

    def getMaxCount(self):
        return self.maxlen

    def getHeight(self):
        return self.getMaxHeight() - self.getMinHeight()

    def getMaxHeight(self):
        return np.max(self.array[0:self.currlen][:, 2])

    def getMinHeight(self):
        #return max(float(np.min(self.array[0:self.currlen][:, 2])), -2.0)
        return float(np.min(self.array[0:self.currlen][:, 2]))

