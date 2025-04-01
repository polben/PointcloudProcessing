import math
import time

import numpy as np
from numpy.matlib import empty


class HIF:



    def __init__(self, alpha, beta, pillar_delta):
        self.a = alpha #>0.5
        self.b = beta #<0.5
        self.d = pillar_delta
        self.global_pillars = {} #Hmn : Pmn - key/value
        self.vert_subdiv = 10
        # pillar: Pmn (float, [(bk, tk, pk)])

    def intHash(self, x, z):
        return ((x * 73856093) ^ (z * 19349663)) & 0xFFFFFFFF

    def hashPillarCoord(self, x, z):
        hx = (x * 73856093) & 0xFFFFFFFF
        hy = (z * 19349663) & 0xFFFFFFFF
        C = int((2 ** 32) * ((5 ** 0.5 - 1) / 2))  # Golden ratio constant
        return hx ^ (hy + C + (hx << 6) + (hx >> 2))

    def getPillarCoord(self, np_point):
        x = int(math.floor(np_point[0] / self.d))
        z = int(math.floor(np_point[2] / self.d))
        return x, z

    def BayesFilter(self, p, Ps, Pd):
        return (Ps * p) / (Ps * p + Pd * (1.0 - p))

    def getLocalPillars(self, np_points):
        local_pillars = {}  # hash - min, max

        st = time.time()
        for i in range(len(np_points)):
            point = np_points[i]
            pillar = self.getPillarCoord(point)
            hash_val = self.hashPillarCoord(*pillar)

            if hash_val in local_pillars:
                pmin, pmax, x, z = local_pillars[hash_val]
                local_pillars[hash_val] = (min(pmin, point[1]), max(pmax, point[1]), x, z)
            else:
                local_pillars[hash_val] = (point[1], point[1], pillar[0], pillar[1])

        print("local pillats: " + str(time.time() - st))
        return local_pillars

    def addNewScanJustPillars(self, np_points):
        local_pillar_dict = self.getLocalPillars(np_points)

        for local_hash in local_pillar_dict:
            mmin, mmax, x, z = local_pillar_dict[local_hash]

            if local_hash in self.global_pillars:
                glob_empty, x, z, gmin, gmax = self.global_pillars[local_hash]
                glob_empty = self.BayesFilter(glob_empty, self.a, self.b)
                self.global_pillars[local_hash] = glob_empty, x, z, mmin, mmax
            else:
                self.global_pillars[local_hash] = (0.5, x, z, mmin, mmax)


        occupied_pillars = []
        for p in self.global_pillars:
            p_empty , x, z, gmin, gmax = self.global_pillars[p]
            if p_empty < 0.2:
                occupied_pillars.append((x, z, gmin, gmax))

        return HIF.getGlobalStaticPillars(occupied_pillars, self.d)

    def addNewScan(self, np_points):

        local_pillar_dict = self.getLocalPillars(np_points)

        st = time.time()
        for Hl in local_pillar_dict:
            mmin, mmax, _, _ = local_pillar_dict[Hl]
            if Hl in self.global_pillars:
                self.global_pillars[Hl] = self.updateGlobalPillar(phash=Hl, current_min=mmin, current_max=mmax)
            else:
                self.global_pillars[Hl] = (0.5, self.getInitialHeightIntervals(mmin, mmax))

        print("global update: " + str(time.time()-st))



    def getInitialHeightIntervals(self, mmin, mmax):
        rrange = mmax - mmin

        # Calculate the height of each interval
        interval_height = rrange / self.vert_subdiv

        # List to store the height intervals
        intervals = []

        # Generate the intervals based on the range and the interval height
        for i in range(self.vert_subdiv):
            # Bottom and top of the interval
            bk = mmin + i * interval_height
            tk = bk + interval_height

            # Ensure the last interval's top doesn't exceed mmax
            if tk > mmax:
                tk = mmax

            # Default occupancy probability for each interval
            pk = 0.5  # This can be updated later based on your filtering method

            # Add the interval to the list
            intervals.append((bk, tk, pk))

        # Return the list of intervals
        return intervals


    def updateGlobalPillar(self, phash, current_min, current_max):
        p_empty_glob, intervals_glob = self.global_pillars[phash]

        if len(intervals_glob) > 10:
            a = 0

        p_empty_glob = self.BayesFilter(p_empty_glob, 1.0 - self.a, 1.0 - self.b)

        intervals_local = self.getInitialHeightIntervals(current_min, current_max)

        endpoints = []
        for intg in intervals_glob:
            bg, tg, pk = intg
            endpoints.extend([bg, tg])

        for intl in intervals_local:
            bl, tl, pk = intl
            endpoints.extend([bl, tl])

        endpoints = np.array(endpoints)
        endpoints = np.sort(np.unique(endpoints))

        new_intervals = []
        for i in range(len(endpoints) - 1):
            new_interval = (endpoints[i], endpoints[i + 1])
            p_new = self.getNewP(new_interval, intervals_glob, intervals_local, p_empty_glob)
            if p_new is not None:
                new_intervals.append((new_interval[0], new_interval[1], p_new))


        return p_empty_glob, new_intervals

    def getNewP(self, new_interval, intervals_global, intervals_local, p_empty_glob):
        intersected_global = self.heightIntervalIntersects(new_interval, intervals_global)
        intersected_local = self.heightIntervalIntersects(new_interval, intervals_local)

        if not intersected_local and not intersected_global:
            return None

        if intersected_local and intersected_global:
            return self.BayesFilter(np.mean(np.array(intersected_global)), self.a, self.b)

        if not intersected_local and intersected_global:
            return self.BayesFilter(np.mean(np.array(intersected_global)), 1 - self.a, 1 - self.b)

        if intersected_local and not intersected_global:
            return self.BayesFilter(p_empty_glob, self.a, self.b)



    def heightIntervalIntersects(self, height_interval, other_intervals):
        mmin, mmax = height_interval
        intersected_ps = []
        for oi in other_intervals:
            omin, omax, p = oi
            if omin <= mmin <= omax:
                intersected_ps.append(p)
            elif omin <= mmax <= omax:
                intersected_ps.append(p)

        return intersected_ps

    @staticmethod
    def getPillarPoints(pillar_dict, pillar_width):

        pillar_grids = []

        for pi in pillar_dict:
            minh, maxh, x_coord, z_coord = pillar_dict[pi]

            x_coord *= pillar_width
            z_coord *= pillar_width

            resolution = 5
            x_vals = np.linspace(x_coord, x_coord + pillar_width, resolution)
            z_vals = np.linspace(z_coord, z_coord + pillar_width, resolution)
            y_vals = np.linspace(minh, maxh, resolution)

            X, Z, Y = np.meshgrid(x_vals, z_vals, y_vals, indexing='ij')
            grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

            pillar_grids.append(grid_points)

        return pillar_grids

    @staticmethod
    def getGlobalStaticPillars(list_x_z, pillar_width):

        if not list_x_z:
            return None

        grids = []
        for pi in list_x_z:
            x_coord, z_coord, gmin, gmax = pi

            x_coord *= pillar_width
            z_coord *= pillar_width

            resolution = 10
            x_vals = np.linspace(x_coord, x_coord + pillar_width, resolution)
            z_vals = np.linspace(z_coord, z_coord + pillar_width, resolution)
            y_vals = np.linspace(gmin, gmax, resolution)

            X, Z, Y = np.meshgrid(x_vals, z_vals, y_vals, indexing='ij')
            grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

            grids.append(grid_points)

        return grids



