import numpy as np

class LidarFilter:



    def __init__(self, maxRange, minHeight, voxelSize, minPoints):
        self.maxRange = maxRange
        self.voxelSize = voxelSize
        self.minPoints = minPoints
        self.minHeight = minHeight
        self.filteredPoints = None

    def voxelize(self, center, np_points):
        keys, counts = self.getVoxelData(np_points)

        self.filteredPoints = keys
        return keys, counts

    def filterGround(self, np_points):
        return self.getTallVoxelPoints(np_points)

    def getVoxelData(self, np_points):
        keys = np.round(np_points / self.voxelSize) * self.voxelSize
        # sorted_keys = np.sort(keys, axis=0) * self.voxelSize

        keys, counts = self.getUniqueKeysAndCounts(sorted_keys=keys)

        #obj, ground = self.getTallVoxelPoints(keys)

        return keys, counts
        #return obj, np.zeros(len(obj))

    def getUniqueKeysAndCounts(self, sorted_keys):
        return np.unique(sorted_keys, axis=0, return_inverse=False, return_counts=True, return_index=False)

    def getTallVoxelPoints(self, np_points): # sort the keys with numpy.sort, do gpu implementation of unique with counts,
                                             # iterate over the counts and check for "tall" voxels that are not ground
        keys = np.round(np_points[:, [0, 2]] / self.voxelSize) * self.voxelSize
        unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sorted_inv = np.argsort(inverse)
        sorted_points = np_points[sorted_inv]


        points_of_interest = []
        ground = []


        last = 0
        for c in counts:
            if c >= self.minPoints:
                bucket_points = sorted_points[last:last + c]
                y = bucket_points[:, 1]
                miny = np.min(y)
                maxy = np.max(y)
                if maxy - miny > self.minHeight:
                    points_of_interest.append(bucket_points)
                else:
                    ground.append(bucket_points)

            last = last + c

        return np.concatenate(points_of_interest, axis=0), np.concatenate(ground, axis=0)

    def getFilteredPoints(self):
        return self.filteredPoints

    def filterDistance(self, np_points):
        max_distance_sq = self.maxRange ** 2  # Compare squared distances
        squared_distances = np.sum(np_points ** 2, axis=1)  # Sum of squares of coordinates

        mask = squared_distances <= max_distance_sq  # Avoid square root
        return np_points[mask]