import time
import unittest

import numpy as np

from ComputeShader import ComputeShader
from LidarDataReader import LidarDataReader
from OxtsDataReader import OxtsDataReader
from PointcloudAlignment import PointcloudAlignment
from PointcloudIcpContainer import PointcloudIcpContainer
from Voxelizer import Voxelizer


class VoxelizerTest(unittest.TestCase):

    def setUp(self):
        path = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"
        calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        self.oxtsDataReader = OxtsDataReader(path)
        self.lidarDataReader = LidarDataReader(path=path, oxtsDataReader=self.oxtsDataReader, calibration=calibration,
                                               targetCamera="02", max_read=10)

        self.pointcloudAlignment = PointcloudAlignment(self.lidarDataReader, self.oxtsDataReader)

        self.computeShader = ComputeShader()
        self.icpContainer = PointcloudIcpContainer(self.computeShader, self.pointcloudAlignment)

        self.tolerance = 1e-5  # 0.00001



    def tearDown(self):
        self.computeShader.cleanup()

    def test_binsearchFindsFirstOccurence(self):
        vox = Voxelizer(self.computeShader, 0.5)

        # Test 1: Simple Case (Target Exists Once)
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(vox.bin_search(arr, 5, find_first=True)[0] == 4)
        self.assertTrue(vox.bin_search(arr, 5, find_first=False)[0]  == 4)

        # Test 2: Multiple Occurrences
        arr = np.array([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4])
        self.assertTrue(vox.bin_search(arr, 2, find_first=True)[0]  == 3)
        self.assertTrue(vox.bin_search(arr, 2, find_first=False)[0]  == 6)

        # Test 3: Target at Start
        arr = np.array([0, 0, 0, 1, 2, 3, 4])
        self.assertTrue(vox.bin_search(arr, 0, find_first=True)[0]  == 0)
        self.assertTrue(vox.bin_search(arr, 0, find_first=False)[0]  == 2)

        # Test 4: Target at End
        arr = np.array([1, 2, 3, 3, 3, 3, 4, 5, 5, 5])
        self.assertTrue(vox.bin_search(arr, 5, find_first=True)[0]  == 7)
        self.assertTrue(vox.bin_search(arr, 5, find_first=False)[0]  == 9)

        # Test 5: Single Element (Match)
        arr = np.array([42])
        self.assertTrue(vox.bin_search(arr, 42, find_first=True)[0]  == 0)
        self.assertTrue(vox.bin_search(arr, 42, find_first=False)[0]  == 0)

        # Test 6: Single Element (No Match)
        arr = np.array([99])
        self.assertTrue(vox.bin_search(arr, 42, find_first=True)[0]  == -1)
        self.assertTrue(vox.bin_search(arr, 42, find_first=False)[0]  == -1)

        # Test 7: No Match in Larger List
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(vox.bin_search(arr, 15, find_first=True)[0]  == -1)
        self.assertTrue(vox.bin_search(arr, 15, find_first=False)[0]  == -1)

        # Test 8: All Elements Are the Same
        arr = np.array([7, 7, 7, 7, 7, 7, 7])
        self.assertTrue(vox.bin_search(arr, 7, find_first=True)[0]  == 0)
        self.assertTrue(vox.bin_search(arr, 7, find_first=False)[0]  == 6)

        # Test 9: Large Dataset
        arr = np.arange(1000000)  # 0 to 999999
        self.assertTrue(vox.bin_search(arr, 567890, find_first=True)[0]  == 567890)
        self.assertTrue(vox.bin_search(arr, 567890, find_first=False)[0]  == 567890)

    def generateChunkArray(self, length, minvalue=0, maxvalue=10):
        # Generate an array of shape (M, 3) with random integers between `low` and `high`
        return np.random.randint(minvalue, maxvalue, size=(length, 3))

    def test_xyzSortedArrayBinsearch(self):
        vox = Voxelizer(self.computeShader, 0.5)

        vox.stored_voxels = 1

        # Test Case 1: Basic Test with a small array
        arr = np.array([[1, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [3, 5, 5]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [3, 5, 5]
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertTrue(np.array_equal(sorted[found], value_to_find))  # Should find the value

        # Test Case 2: Value is at the start of the array
        arr = np.array([[1, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [3, 5, 5]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [1, 1, 1]
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertTrue(np.array_equal(sorted[found], value_to_find))  # Should find the value at the start

        # Test Case 3: Value is at the end of the array
        arr = np.array([[1, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [3, 5, 5]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [3, 4, 4]
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertTrue(np.array_equal(sorted[found], value_to_find))  # Should find the value at the end

        # Test Case 4: Value is not in the array
        arr = np.array([[1, 1, 1], [1, 2, 2], [2, 3, 3], [3, 4, 4], [3, 5, 5]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [4, 6, 6]
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertEqual(found, -1)  # Should not find the value, return -1

        # Test Case 5: Value is repeated multiple times in the array
        arr = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [3, 3, 3]

        with self.assertRaises(RuntimeError, msg="In this impl. no same values should be present"):
            found, _ = vox.findVoxelId(sorted, value_to_find)

        # Test Case 6: Value occurs only once in a larger dataset
        arr = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [4, 4, 4]
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertTrue(np.array_equal(sorted[found], value_to_find))  # Should find the value

        # Test Case 7: Edge case with a single value (match)
        arr = np.array([[42, 42, 42]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [42, 42, 42]
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertTrue(np.array_equal(sorted[found], value_to_find))  # Should find the only value

        # Test Case 8: Edge case with a single value (no match)
        arr = np.array([[42, 42, 42]])
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [99, 99, 99]
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertEqual(found, -1)  # Should not find the value, return -1

        # Test Case 9: Large array with random values
        arr = np.random.randint(-100, 100, size=(1000, 3))
        sorted = vox.xyzSortedArray(arr)
        value_to_find = sorted[500]  # Pick a value from the sorted array
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertTrue(np.array_equal(sorted[found], value_to_find))  # Should find the value in the large array

        # Test Case 10: Large array with a missing value
        arr = np.random.randint(-100, 100, size=(1000, 3))
        sorted = vox.xyzSortedArray(arr)
        value_to_find = [100, 100, 100]  # A value that is unlikely to be in the array
        found, _ = vox.findVoxelId(sorted, value_to_find)
        self.assertEqual(found, -1)  # Should not find the value, return -1

    def test_xyzSortedBinsearchIsFast(self):

        vox = Voxelizer(self.computeShader, 0.5)
        vox.stored_voxels = 1

        arr = np.random.randint(-100, 100, size=(10000, 3))
        st = time.time()
        sorted = vox.xyzSortedArray(arr)
        print("lex-sort: " + str(time.time() - st))

        value_to_find = [100, 100, 100]  # A value that is unlikely to be in the array

        st = time.time()
        found, _ = vox.findVoxelId(sorted, value_to_find)
        print("checked: " + str(time.time()-st))

        self.assertEqual(found, -1)  # Should not find the value, return -1

    def test_sameUnknownPoints(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)
        pc = pts[0]

        v = Voxelizer(self.computeShader, voxel_size=0.5)

        v.handleAddingPoints(pc)

        current_stored_begin_index = v.last_added_point - len(pc)

        unknowns_g, vox_ind_g, vox_dat_g, debug_data = v.compute.prepareDispatchVoxelizer(
            pc, v.voxel_index, v.voxel_data, v.voxel_size, v.stored_voxels, current_stored_begin_index, v.max_points,
            debug=True
        )

        unknowns_c, vox_ind_c, vox_dat_c, vox_ind_coords_c, vox_ind_ids_c = v.cpuGpuDebug_dispatchVoxels(
            pc, v.voxel_index, v.voxel_data, v.voxel_size, v.stored_voxels, current_stored_begin_index
        )

        unique_unknowns_c, counts_c = np.unique(unknowns_c, return_counts=True)
        unique_unknowns_g, counts_g = np.unique(unknowns_g, return_counts=True)

        self.assertTrue(np.array_equal(unique_unknowns_c, unique_unknowns_g))


    def test_gpuCanVoxelize(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, voxel_size=0.5)


        for i in range(5):
            print("iteration: " + str(i + 1))
            pc = pts[i]


            v.handleAddingPoints(pc)

            current_stored_begin_index = v.last_added_point - len(pc)


            original_voxel_inds = np.zeros_like(v.voxel_index)
            original_voxel_inds = np.copy(v.voxel_index)

            unknowns_g, vox_ind_g, vox_dat_g, debug_data = v.compute.prepareDispatchVoxelizer(
                pc, v.voxel_index, v.voxel_data, v.voxel_size, v.stored_voxels, current_stored_begin_index, v.max_points, debug=True
            )

            unknowns_c, vox_ind_c, vox_dat_c, vox_ind_coords_c, vox_ind_ids_c = v.cpuGpuDebug_dispatchVoxels(
                pc, v.voxel_index, v.voxel_data, v.voxel_size, v.stored_voxels, current_stored_begin_index
            )



            unique_coords_g = np.unique(debug_data[:, :3], axis=0).astype(np.int32)
            unique_coords_c = np.unique(vox_ind_coords_c, axis=0).astype(np.int32)



            self.assertTrue(np.array_equal(unique_coords_g, unique_coords_c)) # asserting same coordinates for voxels, THIS FAILS IF when appending point to the list, before voxelization is not threated as float32 ( yes :) )

            vox_ind_ids_g = debug_data[:, 3].astype(np.int32)

            uniq_ids_c = np.unique(vox_ind_ids_c)
            uniq_ids_g = np.unique(vox_ind_ids_g)

            self.assertTrue(np.array_equal(uniq_ids_c, uniq_ids_g)) # same voxel data indexes



            voxel_data_counts_c = vox_dat_c[:, 0]
            voxel_data_counts_g = vox_dat_g[:, 0]
            self.assertTrue(np.array_equal(voxel_data_counts_c, voxel_data_counts_g))

            #!# self.assertTrue(np.array_equal(vox_dat_c, vox_dat_g)) # voxel datas are the same > !!!! order of indicies in voxel datas might differ bcs concurrency

            if v.max_points >= 4096:
                for j in range(len(vox_dat_c)):
                    unique_point_indicies_c = np.unique(vox_dat_c[j])
                    unique_point_indicies_g = np.unique(vox_dat_g[j])
                    self.assertTrue(np.array_equal(unique_point_indicies_c, unique_point_indicies_g))


            self.assertTrue(np.array_equal(vox_ind_c, vox_ind_g)) # voxel indicies are equal and unchanged after gpu/cpu computation
            self.assertTrue(np.array_equal(original_voxel_inds, vox_ind_g)) # voxel indicies are equal and unchanged after gpu/cpu computation
            self.assertTrue(np.array_equal(original_voxel_inds, vox_ind_c)) # voxel indicies are equal and unchanged after gpu/cpu computation


            unique_unknowns_c, counts_c = np.unique(unknowns_c, return_counts=True)
            unique_unknowns_g, counts_g = np.unique(unknowns_g, return_counts=True)

            self.assertTrue(np.array_equal(unique_unknowns_c, unique_unknowns_g)) # same unfound points (in the voxels)
            self.assertTrue(np.array_equal(counts_c, counts_g))


            v.voxel_index = vox_ind_c
            v.voxel_data = vox_dat_c

            unknown_points = unknowns_c

            v.storeUnknownPoints(unknown_points, current_stored_begin_index)

            print("passed")


    def test_canGetVoxelDensities(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, voxel_size=0.5)

        for i in range(5):
            pc = pts[i]
            v.addPoints(pc, True)

            dens = v.getVoxelDensities()
            print(np.min(dens), np.max(dens), np.mean(dens), np.median(dens))
            print(np.argmin(dens), np.argmax(dens))
            print(dens)
            print("------------\n")




    def getAlignedLidarPoints(self, num_scans):
        files = self.lidarDataReader.getfilenames()

        prevpos = np.array([0, 0, 0])
        currpos = np.array([0, 0, 0])
        prevtime = None
        currtime = self.oxtsDataReader.getOx(files[0]).getTime()

        aligned_points = []
        positions = [np.array([0,0,0])]
        rotations = []

        load = num_scans
        for i in range(load):
            name = files[i]
            oxts = self.oxtsDataReader.getOx(name)
            rot = np.linalg.inv(oxts.getTrueRotation(np.eye(3)))
            yawrot = oxts.getYawRotation(np.eye(3))
            rot = yawrot @ rot

            if prevtime is not None:
                velocity = -oxts.getVelocity()
                currtime = oxts.getTime()
                deltatime = (currtime - prevtime).total_seconds()
                velocity = (rot @ velocity.T).T

                currpos = prevpos + velocity * deltatime
                positions.append(currpos)

            pts = self.getLidarPoints(i)
            pts = (rot @ pts.T).T + currpos
            rotations.append(rot)

            aligned_points.append(pts)

            prevtime = currtime
            prevpos = currpos

        return aligned_points, positions, rotations

    def getLidarPoints(self, index=0):
        filenames = self.lidarDataReader.getfilenames()
        pts, cols = self.lidarDataReader.getPoints(filenames[index])
        return pts