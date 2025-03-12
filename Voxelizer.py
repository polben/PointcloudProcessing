import numpy as np

class VoxelData:


    def __init__(self, lidarFilter):
        self.filter = lidarFilter
        self.voxel_map = {}
        self.voxelCenters = []

    def addPoints(self, origin, np_points):
        keys, counts = self.filter.voxelize(origin, np_points)

        self.append(keys, counts)

        return self.voxelCenters


    def getFilteredPoints(self, np_points):
        objects, ground = self.filter.filterGround(np_points)
        return objects

    def append(self, keys, counts):
        for i in range(len(keys)):

            hash = tuple(keys[i])
            count = counts[i]
            if self.voxel_map.__contains__(hash):
                self.voxel_map[hash] = self.voxel_map[hash] + count
            else:
                self.voxel_map[hash] = count
                self.voxelCenters.append(keys[i])