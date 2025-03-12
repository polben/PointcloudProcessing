"""

from collections import defaultdict
from ctypes.wintypes import POINT

import numpy as np

from DynamicNP import DynamicNP
from ShapeUtility import ShapeUtility


class VoxelData:


    voxsize = None





    def __init__(self, boxSize, minpoints):
        # key will be which voxel the point lands in
        self.voxsize = boxSize
        self.minpoints = minpoints
        self.density_flags = {}
        self.voxels = {}
        self.init_length = 0
        self.baked_grid = None
        self.baked_points = None

    def voxelize(self, np_points):

        keys = np.round((np_points[:, :2] / self.voxsize)).astype(np.int32)
        unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        big_counts = counts[ counts > 10]
        sorted_inv = np.argsort(inverse) # would sort the inverse to corresponding buckets, therefore the original points too
        sorted_points = np_points[sorted_inv]
        #test = inverse[sorted_inv]
        #bincount = np.bincount(test)
        #asd = counts - bincount
        self.init_length = np.max(counts)

        last = 0
        for c in counts:
            bucket_points = sorted_points[last:last+c]
            key = VoxelData.hash(bucket_points[0])

            self.appendToDict(key, bucket_points)

            last = last + c

        a = 0

    def getKeys(self):
        return self.voxels.keys()

    def getVoxel(self, key):
        return self.voxels[key]

    @staticmethod
    def hash(arr):
        return tuple(np.round(arr[:2]).astype(np.int32))

    @staticmethod
    def invHash(tup):
        return np.array([tup[0], tup[1], 0])

    def appendToDict(self, key, array):

        if key in self.voxels.keys():
            self.voxels[key].addPoints(array)
        else:
            self.voxels[key] = DynamicNP(self.init_length)
            self.voxels[key].addPoints(array)

        numpoints = self.voxels[key].getCount()
        if key not in self.density_flags.keys():
            if numpoints < self.minpoints:
                self.density_flags[key] = numpoints

            if self.voxels[key].getHeight() < 0.25:
                self.density_flags[key] = numpoints

        else:
            if numpoints >= self.minpoints and self.voxels[key].getHeight() > 0.25:
                del self.density_flags[key]

    def clearNotDense(self):
        for k in self.density_flags.keys():
                del self.voxels[k]
        self.density_flags.clear()

    def getBBGrid(self, density):
        shapes = []

        for k in self.voxels.keys():

            ht = self.voxels[k].getHeight()
            minht = self.voxels[k].getMinHeight()
            # print(minht)
            coord = VoxelData.invHash(k)
            box = ShapeUtility.tallbox( np.array([coord[0] * self.voxsize, coord[1] * self.voxsize, minht]), self.voxsize, ht, density)
            shapes.append(box)

        frame = ShapeUtility.concat(shapes)
        return frame

    def voxelCount(self):
        return len(list(self.voxels.keys()))

    def bake(self, density):
        self.baked_grid = self.getBBGrid(density)

    def bake_points(self):
        pts = []
        for k in self.voxels.keys():
            points = self.getVoxel(k).getPoints()
            pts.append(points)

        self.baked_points = ShapeUtility.concat(pts)


    def get_baked_points(self):
        return self.baked_points

    def get_baked_grid(self):
        return self.baked_grid

    def getStat(self):
        maxpoints = 0
        maxlen = 0
        keys = list(self.voxels.keys())

        for k in keys:
            count = self.voxels[k].getCount()
            maxcount = self.voxels[k].getMaxCount()
            if count > maxpoints:
                maxpoints = count

            if maxcount > maxlen:
                maxlen = maxcount


        print("Max points: " + str(maxpoints) + "; Max len: " + str(maxlen))
        print(str(self.voxelCount()) + " voxels")

"""