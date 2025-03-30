import time

import numpy as np
from open3d.examples.geometry.voxel_grid_carving import voxel_carving

from Renderer import Renderer


class Voxelizer:

    # voxel: holds an array of point indicies, max 1000 pts, lets say
    # voxel resolution: a chunk can have at most voxres^3 voxels

    # voxel data will be a 2d array: row: pointed to by chunk, each row contains an array of indicies of contained points

    def __init__(self, compute_shader, voxel_size = 1):

        self.voxel_size = voxel_size
        self.max_points = 4096
        self.compute = compute_shader


        self.voxel_index = None # (M, 4) growing * 2, [x, y, z voxel coord, pointer to voxel data]
        self.voxel_data = None # (M, 1 + max_points), growing * 2, [0]: number of points sored [1:] indexes of stored points
        self.stored_voxels = 0

        self.added_points = None
        self.last_added_point = 0
        self.bigint = 9999999

        self.initVoxels()

    def cpuGpuDebug_dispatchVoxels(self, np_points, voxel_index, voxel_data, voxel_size, stored_voxel_num, begin_index):
        unknown_len = 1 + len(np_points)
        modd = 16 - unknown_len % 16
        unknown_len = unknown_len + modd

        unknown_points = np.zeros(unknown_len).astype(np.int32)

        t_voxel_index = voxel_index
        t_voxel_data = voxel_data

        lens_data = np.array([len(np_points), stored_voxel_num, begin_index, 0]).astype(np.uint32)

        calculated_voxel_ids = []
        voxel_index_entries = []

        for i in range(lens_data[0]):
            point = np_points[i]

            voxel_coord = self.getVoxelCoord(point.astype(np.float32))
            calculated_voxel_ids.append(voxel_coord)

            voxel_index_entry = self.findVoxelId_gpu(t_voxel_index, voxel_coord, lens_data)
            voxel_index_entries.append(voxel_index_entry)
            if voxel_index_entry == -1:
                unknown_count = unknown_points[0]
                unknown_points[1 + unknown_count] = begin_index + i
                unknown_points[0] += 1

            else:
                voxel_data_index = t_voxel_index[voxel_index_entry][3]
                stored_points = t_voxel_data[voxel_data_index][0]
                t_voxel_data[voxel_data_index][1 + stored_points] = begin_index + i
                t_voxel_data[voxel_data_index][0] += 1

        return unknown_points, t_voxel_index, t_voxel_data, np.array(calculated_voxel_ids), np.array(voxel_index_entries)

    def cpuGpuDebug_original(self, np_points, voxel_index, voxel_data, voxel_size, stored_voxel_num, begin_index):
        unknown_len = 1 + len(np_points)
        modd = 16 - unknown_len % 16
        unknown_len = unknown_len + modd

        unknown_points = np.zeros(unknown_len).astype(np.int32)

        t_voxel_index = voxel_index
        t_voxel_data = voxel_data

        lens_data = np.array([len(np_points), stored_voxel_num, begin_index, 0]).astype(np.uint32)

        for i in range(lens_data[0]):
            point = np_points[i]

            voxel_coord = self.getVoxelCoord(point)

            voxel_index_entry, _ = self.findVoxelId(t_voxel_index, voxel_coord)

            if voxel_index_entry == -1:
                unknown_count = unknown_points[0]
                unknown_points[1 + unknown_count] = begin_index + i
                unknown_points[0] += 1

            else:
                voxel_data_index = t_voxel_index[voxel_index_entry][3]
                stored_points = t_voxel_data[voxel_data_index][0]
                t_voxel_data[voxel_data_index][1 + stored_points] = begin_index + i
                t_voxel_data[voxel_data_index][0] += 1

        return unknown_points, t_voxel_index, t_voxel_data

    def getVoxelDensities(self):
        return self.voxel_data[:, 0]

    def addPoints(self, np_points, printt=False):
        self.handleAddingPoints(np_points)

        current_stored_begin_index = self.last_added_point - len(np_points) # points exactly to the first point in the current cloud

        start = time.time()
        unknown_points, self.voxel_index, self.voxel_data, _ = self.compute.prepareDispatchVoxelizer(np_points, self.voxel_index, self.voxel_data, self.voxel_size, self.stored_voxels, current_stored_begin_index, debug=False)
        if printt:
            print("gpu> unknown points: " + str(unknown_points[0]) + " " + str(time.time()-start))

        start = time.time()
        self.storeUnknownPoints(unknown_points, current_stored_begin_index)
        if printt:
            print("stored unknown: " + str(time.time()-start))


    def storeUnknownPoints(self, unknown_points, current_stored_begin_index):
        unknown_length = unknown_points[0]

        for i in range(unknown_length):

            point = self.added_points[unknown_points[1 + i]][:3]
            voxel = self.getVoxelCoord(point)

            voxel_index_entry, ins = self.findVoxelId(self.voxel_index, voxel)

            voxel_data_index = self.voxel_index[voxel_index_entry][3]
            if voxel_index_entry == -1:
                values_to_move = self.voxel_index[ins:self.stored_voxels]
                if len(values_to_move) > 0:
                    self.voxel_index[ins + 1:self.stored_voxels + 1] = self.voxel_index[ins:self.stored_voxels]

                self.voxel_index[ins][:3] = voxel
                self.voxel_index[ins][3] = self.stored_voxels

                self.voxel_data[self.stored_voxels][0] = 1
                self.voxel_data[self.stored_voxels][1] = current_stored_begin_index + i

                self.stored_voxels += 1

                # realloc

                if self.stored_voxels == len(self.voxel_index):
                    self.voxel_index = np.vstack([self.voxel_index, np.full_like(self.voxel_index, self.bigint)])
                    self.voxel_data = np.vstack([self.voxel_data, np.zeros_like(self.voxel_data)])

            else:
                stored_points = self.voxel_data[voxel_data_index][0]
                self.voxel_data[voxel_data_index][1 + stored_points] = current_stored_begin_index + i
                self.voxel_data[voxel_data_index][0] += 1

            """print(np.array_equal(self.voxel_index, self.xyzSortedArray(self.voxel_index)))
            print(i)"""



    def initVoxels(self, init_size=100):
        self.voxel_index = np.full((100, 4), self.bigint).astype(np.int32)
        self.voxel_data = np.zeros((100, self.max_points)).astype(np.int32)

    def getVoxelCoord(self, point):
        # this is a grid of voxels of sidelength self.voxel_size
        # a point [0.05, 0.7, 0.1] should have id of [1, 1, 1]

        voxel_id = np.ceil(point / self.voxel_size).astype(np.int32)
        return voxel_id

    def handleAddingPoints(self, np_points):
        if self.added_points is None:
            self.added_points = np.empty((len(np_points) * 2, 4))
            self.added_points[:len(np_points), :3] = np_points  # Assign directly
            self.last_added_point += len(np_points)



        else:
            if self.last_added_point + len(np_points) >= len(self.added_points):
                self.added_points = np.vstack([self.added_points, np.empty_like(self.added_points)])

            self.added_points[self.last_added_point:self.last_added_point + len(np_points), :3] = np_points
            self.last_added_point += len(np_points) # points to last point + 1



    def xyzSortedArray(self, np_array):
        return np_array[np.lexsort((np_array[:, 2], np_array[:, 1], np_array[:, 0]))]

    def findVoxelId_gpu(self, xyz_sorted_array, value, lens_data):
        if lens_data[1] == 0:
            return -1

        axis_x = 0
        axis_y = 1
        axis_z = 2

        x_val = value[0]
        y_val = value[1]
        z_val = value[2]


        first_occ_x = self.bin_search_gpu(axis_x, x_val, xyz_sorted_array, True, 0, int(lens_data[1]) - 1)
        if first_occ_x == -1:
            return -1

        last_occ_x = self.bin_search_gpu(axis_x, x_val, xyz_sorted_array, False, 0, int(lens_data[1]) - 1)



        first_occ_y = self.bin_search_gpu(axis_y, y_val, xyz_sorted_array, True, first_occ_x, last_occ_x)
        if first_occ_y == -1:
            return -1


        last_occ_y = self.bin_search_gpu(axis_y, y_val, xyz_sorted_array, False, first_occ_x, last_occ_x)



        first_occ_z = self.bin_search_gpu(axis_z, z_val, xyz_sorted_array, True, first_occ_y, last_occ_y)
        if first_occ_z == -1:
            return -1
        else:
            return first_occ_z


    def findVoxelId(self, xyz_sorted_array, value):
        if self.stored_voxels == 0:
            return -1, 0


        x = xyz_sorted_array[:, 0]
        first_occ_x, ins_x = self.bin_search(x, value[0], find_first=True)
        if first_occ_x == -1:
            return -1, ins_x

        last_occ_x, _ = self.bin_search(x, value[0], find_first=False)



        y = xyz_sorted_array[:, 1][first_occ_x : last_occ_x + 1]
        first_occ_y, ins_y = self.bin_search(y, value[1], find_first=True) #+ first_occ_x
        if first_occ_y == -1:
            return -1, ins_x + ins_y


        last_occ_y, _ = self.bin_search(y, value[1], find_first=False) #+ first_occ_x




        first_occ_y = first_occ_y + first_occ_x
        last_occ_y = last_occ_y + first_occ_x



        z = xyz_sorted_array[:, 2][first_occ_y : last_occ_y + 1]
        first_occ_z, ins_z = self.bin_search(z, value[2], find_first=True) # + first_occ_y
        if first_occ_z == -1:
            return -1, ins_x + ins_y + ins_z

        last_occ_z, _ = self.bin_search(z, value[2], find_first=False) # + first_occ_y



        if first_occ_z == last_occ_z:
            return first_occ_z + first_occ_y, None
        else:
            raise RuntimeError("In this impl. no same values should be present")
            # return -1


    def bin_search_gpu(self, axis, value, sorted_array_values, find_first, sfrom, sto):
        l = sfrom
        h = sto

        first_occ = -1

        """runs = 0
        vals = []"""
        while l <= h:
            mid = (l + h) // 2
            """runs += 1

            vals.append((l, h, mid, runs))"""
            try:
                if sorted_array_values[mid][axis] == value:
                    first_occ = mid
                    if find_first:
                        h = mid - 1
                    else:
                        l = mid + 1
                elif sorted_array_values[mid][axis] < value:
                    l = mid + 1
                else:
                    h = mid - 1
            except IndexError as e:
                """print("RECURSING INTO FALIED BINSEARCH")
                self.bin_search_gpu(axis, value, sorted_array_values, find_first, sfrom, sto)
                asd3 = vals
                wtf = 0
                asd = l
                asd2 = h"""
                raise e

        return first_occ

    def bin_search(self, sorted_array_values, value, find_first=True):
        l = 0
        h = len(sorted_array_values) - 1

        first_occ = -1

        while l <= h:
            mid = (l + h) // 2

            if sorted_array_values[mid] == value:
                first_occ = mid
                if find_first:
                    h = mid - 1
                else:
                    l = mid + 1
            elif sorted_array_values[mid] < value:
                l = mid + 1
            else:
                h = mid - 1

        return first_occ, l






    def getVoxelPositions(self, normalize=True):
        coords = self.voxel_index[:, :3]
        if normalize:
            return coords * self.voxel_size
        else:
            return coords

    def getVoxelsAsLineGrid(self):
        vox_coords = self.getVoxelPositions()
        lines = []
        offset = -np.array([self.voxel_size, self.voxel_size, self.voxel_size]) / 2
        for i in range(len(vox_coords)):
            v = vox_coords[i]
            edges = Renderer.unitCubeEdges() * self.voxel_size + v + offset
            lines.extend(edges)

        return lines

    @staticmethod
    def voxelGroundFilter(np_points, res_x = 1000, res_y=1000):
        mmin = np.min(np_points, axis=0)
        mmax = np.max(np_points, axis=0)

        # Step 1: Define voxel grid in X-Z (horizontal plane)
        x_bins = np.linspace(mmin[0], mmax[0], res_x + 1)
        z_bins = np.linspace(mmin[2], mmax[2], res_y + 1)

        # Step 2: Assign points to voxel indices
        ind_x = np.digitize(np_points[:, 0], x_bins) - 1
        ind_z = np.digitize(np_points[:, 2], z_bins) - 1

        # Ensure indices stay within valid bounds
        ind_x = np.clip(ind_x, 0, res_x - 1)
        ind_z = np.clip(ind_z, 0, res_y - 1)

        voxel_ids = ind_z * res_x + ind_x  # Flatten (x, z) to 1D index

        # Step 3: Compute min/max height per voxel
        min_heights = np.full(res_x * res_y, np.inf)
        max_heights = np.full(res_x * res_y, -np.inf)

        np.minimum.at(min_heights, voxel_ids, np_points[:, 1])  # Use Y for height
        np.maximum.at(max_heights, voxel_ids, np_points[:, 1])

        # Step 4: Identify ground voxels
        ground_voxels = (max_heights - min_heights) < 0.2

        # Step 5: Filter points based on voxel classification
        is_ground_point = ground_voxels[voxel_ids]
        return np_points[~is_ground_point]  # Keep only non-ground points


