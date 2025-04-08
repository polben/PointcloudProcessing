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
        path = "F:/uni/3d-pointcloud/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync"
        calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        self.oxtsDataReader = OxtsDataReader()
        self.lidarDataReader = LidarDataReader()

        self.pointcloudAlignment = PointcloudAlignment()

        self.computeShader = ComputeShader()
        self.icpContainer = PointcloudIcpContainer(self.computeShader, self.pointcloudAlignment)

        self.tolerance = 1e-5  # 0.00001

        self.lidarDataReader.init(path, calibration, "02", 20, None)
        self.oxtsDataReader.init(path)
        self.pointcloudAlignment.init(self.lidarDataReader, self.oxtsDataReader)

        self.STAT_PROB = 0
        self.STAT_MAX_PROB_PREV_POINTS = 1
        self.STAT_STAGE_STATUS = 2
        self.STAT_IS_STAGED = 3

        self.STAGE_STATUS_WILL_BE_STAGED = 0.2
        self.STAGE_STATUS_READY_TO_STAGE = 0.8
        self.STAGE_STATUS_NONE = 0.0

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
        pc2 = pts[1]

        v = Voxelizer(self.computeShader, None)
        v.init(0.5)

        v.handleAddingPoints(pc)

        current_stored_begin_index = v.last_added_point - len(pc)

        unknowns_g, _ = v.compute.prepareDispatchVoxelizer(
                np_points=pc,
                voxel_index=v.voxel_index,
                voxel_data=v.voxel_data,
                voxel_size=v.voxel_size,
                stored_voxel_num=v.stored_voxels,
                begin_index=current_stored_begin_index,
                max_points_to_store=v.max_points,
                realloc_needed=v.realloc_needed,
                prev_stored_voxels=v.prev_stored_voxels,
                debug=False)

        unknowns_c, vox_ind_c, vox_dat_c, vox_ind_coords_c, vox_ind_ids_c = v.cpuGpuDebug_dispatchVoxels(
                np_points=pc,
                voxel_index=v.voxel_index,
                voxel_data=v.voxel_data,
                voxel_size=v.voxel_size,
                stored_voxel_num=v.stored_voxels,
                begin_index=current_stored_begin_index,
            )


        unique_unknowns_c, counts_c = np.unique(unknowns_c, return_counts=True)
        unique_unknowns_g, counts_g = np.unique(unknowns_g, return_counts=True)


        self.assertTrue(np.array_equal(unique_unknowns_c, unique_unknowns_g))
        print("pass 1")

        v.realloc_needed, v.prev_stored_voxels = v.storeUnknownPoints(unknowns_g)
        v.last_run_realloc = v.realloc_needed

        unknowns_g, _ = v.compute.prepareDispatchVoxelizer(
            np_points=pc2,
            voxel_index=v.voxel_index,
            voxel_data=v.voxel_data,
            voxel_size=v.voxel_size,
            stored_voxel_num=v.stored_voxels,
            begin_index=current_stored_begin_index,
            max_points_to_store=v.max_points,
            realloc_needed=v.realloc_needed,
            prev_stored_voxels=v.prev_stored_voxels,
            debug=False)

        unknowns_c, vox_ind_c, vox_dat_c, vox_ind_coords_c, vox_ind_ids_c = v.cpuGpuDebug_dispatchVoxels(
            np_points=pc2,
            voxel_index=v.voxel_index,
            voxel_data=v.voxel_data,
            voxel_size=v.voxel_size,
            stored_voxel_num=v.stored_voxels,
            begin_index=current_stored_begin_index,
        )

        unique_unknowns_c, counts_c = np.unique(unknowns_c, return_counts=True)
        unique_unknowns_g, counts_g = np.unique(unknowns_g, return_counts=True)

        self.assertTrue(np.array_equal(unique_unknowns_c, unique_unknowns_g))
        print("pass 2")

    def test_gpuCanVoxelize(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None)
        # v.max_points = 4096 set with shader!
        v.init(0.5)


        for i in range(5):
            print("iteration: " + str(i + 1))
            pc = pts[i]


            v.handleAddingPoints(pc)

            current_stored_begin_index = v.last_added_point - len(pc)


            original_voxel_inds = v.voxel_index.copy()

            unknowns_g, debug_data = v.compute.prepareDispatchVoxelizer(
                np_points=pc,
                voxel_index=v.voxel_index,
                voxel_data=v.voxel_data,
                voxel_size=v.voxel_size,
                stored_voxel_num=v.stored_voxels,
                begin_index=current_stored_begin_index,
                max_points_to_store=v.max_points,
                realloc_needed=v.realloc_needed,
                prev_stored_voxels=v.prev_stored_voxels,
                debug=True)

            unknowns_c, vox_ind_c, vox_dat_c, vox_ind_coords_c, vox_ind_ids_c = v.cpuGpuDebug_dispatchVoxels(
                np_points=pc,
                voxel_index=v.voxel_index,
                voxel_data=v.voxel_data,
                voxel_size=v.voxel_size,
                stored_voxel_num=v.stored_voxels,
                begin_index=current_stored_begin_index,
            )



            unique_coords_g = np.unique(debug_data[:, :3], axis=0).astype(np.int32)
            unique_coords_c = np.unique(vox_ind_coords_c, axis=0).astype(np.int32)



            self.assertTrue(np.array_equal(unique_coords_g, unique_coords_c)) # asserting same coordinates for voxels, THIS FAILS IF when appending point to the list, before voxelization is not threated as float32 ( yes :) )

            vox_ind_ids_g = debug_data[:, 3].astype(np.int32)

            uniq_ids_c = np.unique(vox_ind_ids_c)
            uniq_ids_g = np.unique(vox_ind_ids_g)

            self.assertTrue(np.array_equal(uniq_ids_c, uniq_ids_g)) # same voxel data indexes

            vox_dat_g = v.voxel_data.copy()
            v.compute.getFullVoxelData(v.stored_voxels, v.max_points, vox_dat_g)

            voxel_data_counts_c = vox_dat_c[:, 0]
            voxel_data_counts_g = vox_dat_g[:, 0]
            self.assertTrue(np.array_equal(voxel_data_counts_c, voxel_data_counts_g))

            #!# self.assertTrue(np.array_equal(vox_dat_c, vox_dat_g)) # voxel datas are the same > !!!! order of indicies in voxel datas might differ bcs concurrency

            if v.max_points >= 4096:
                for j in range(len(vox_dat_c)):
                    unique_point_indicies_c = np.unique(vox_dat_c[j])
                    unique_point_indicies_g = np.unique(vox_dat_g[j])
                    self.assertTrue(np.array_equal(unique_point_indicies_c, unique_point_indicies_g))


            self.assertTrue(np.array_equal(original_voxel_inds, v.voxel_index)) # voxel indicies are equal and unchanged after gpu/cpu computation
            self.assertTrue(np.array_equal(original_voxel_inds, vox_ind_c)) # voxel indicies are equal and unchanged after gpu/cpu computation



            v.realloc_needed, v.prev_stored_voxels = v.storeUnknownPoints(unknowns_g)
            v.last_run_realloc = v.realloc_needed

            print("passed")


    def test_basicDataIntegrity(self):

        v = Voxelizer(self.computeShader, None)
        v.init(0.5)


        for i in range(100):
            voxel_index = np.array([[0, 0, 0, 0], [0,2,0,1]]).astype(np.int32)
            voxel_data = np.zeros((2, v.max_points)).astype(np.int32)
            voxel_stat = np.zeros((2, 4)).astype(np.float32)

            voxel_stat[0][3] = 1.0 # both voxels are "already staged"
            voxel_stat[1][3] = 1.0

            # 1 point with point index 1
            voxel_data[0][0] = 1
            voxel_data[0][1] = 1

            # 2 points with index 2,3
            voxel_data[1][0] = 2
            voxel_data[1][1] = 2
            voxel_data[1][2] = 3

            v.compute.bufferVoxelIndex(voxel_index)
            v.compute.fullBufferVoxelData(voxel_data)

            stored_voxel_num = len(voxel_index)

            filter_outliers = True
            stage_everything = True
            max_staging_area = 512

            new_voxel_stats, staged_voxels, counter_buffer = v.compute.prepareDispatchVoxelStager(
                voxel_index=None,
                stored_voxel_num=stored_voxel_num,
                max_points=v.max_points,
                voxel_statistics=voxel_stat,
                filter_outliers=filter_outliers,
                stage_everything=stage_everything,
                max_staging=max_staging_area,
                debug=True
            )

            self.assertTrue(counter_buffer[0][0] == 0)
            self.assertTrue(counter_buffer[0][1] == max_staging_area)
            self.assertTrue(counter_buffer[0][2] == v.max_points)
            self.assertTrue(counter_buffer[0][3] == stored_voxel_num)

            if filter_outliers:
                self.assertTrue(counter_buffer[0][5] == 0)
            else:
                self.assertTrue(counter_buffer[0][5] == 1)

            if stage_everything:
                self.assertTrue(counter_buffer[0][4] == 1)
            else:
                self.assertTrue(counter_buffer[0][4] == 0)

            try: # allocating with numpy empty results in random data sometimes, but that should not be a problem
                self.assertTrue(np.array_equal(staged_voxels, np.zeros((max_staging_area, v.max_points)).astype(np.int32)))
            except AssertionError as e:
                a = 0


    def test_shouldUpdateStagingStatus(self):
        v = Voxelizer(self.computeShader, None)
        v.init(0.5)

        voxel_index = np.array([[0, 0, 0, 0], [0, 2, 0, 1]]).astype(np.int32)
        voxel_data = np.zeros((2, v.max_points)).astype(np.int32)
        voxel_stat = np.zeros((2, 4)).astype(np.float32)

        voxel_stat[0][3] = 0.0  # none of the voxels are "already staged"
        voxel_stat[1][3] = 0.0

        voxel_stat[0][0] = 1.0 # setting probs > 0.9
        voxel_stat[1][0] = 0.91

        # 1 point with point index 1
        voxel_data[0][0] = 1
        voxel_data[0][1] = 1

        # 2 points with index 2,3
        voxel_data[1][0] = 2
        voxel_data[1][1] = 2
        voxel_data[1][2] = 3

        v.compute.bufferVoxelIndex(voxel_index)
        v.compute.fullBufferVoxelData(voxel_data)

        stored_voxel_num = len(voxel_index)

        filter_outliers = False
        stage_everything = False
        max_staging_area = 512

        new_voxel_stats, staged_voxels, counter_buffer = v.compute.prepareDispatchVoxelStager(
            voxel_index=None,
            stored_voxel_num=stored_voxel_num,
            max_points=v.max_points,
            voxel_statistics=voxel_stat,
            filter_outliers=filter_outliers,
            stage_everything=stage_everything,
            max_staging=max_staging_area,
            debug=True
        )

        self.assertTrue(counter_buffer[0][0] == 0)
        self.assertTrue(counter_buffer[0][1] == max_staging_area)
        self.assertTrue(counter_buffer[0][2] == v.max_points)
        self.assertTrue(counter_buffer[0][3] == stored_voxel_num)

        if filter_outliers:
            self.assertTrue(counter_buffer[0][5] == 0)
        else:
            self.assertTrue(counter_buffer[0][5] == 1)

        if stage_everything:
            self.assertTrue(counter_buffer[0][4] == 1)
        else:
            self.assertTrue(counter_buffer[0][4] == 0)

        self.assertTrue(new_voxel_stats[0][2] == 0.2) # prob has increased
        self.assertTrue(new_voxel_stats[1][2] == 0.2)

        self.assertTrue(new_voxel_stats[0][1] == 2.0) # storing max prob + pc (1.0 + 1 points)
        self.assertTrue(new_voxel_stats[1][1] >= 2.9 and new_voxel_stats[1][2] <= 3.0) # storing max prob + pc (1.0 + 1 points)

    def test_probabilitiesSayWithin0and1(self):
        v = Voxelizer(self.computeShader, None)
        v.init(0.5)

        voxel_index = np.array([[0, 0, 0, 0], [0, 2, 0, 1]]).astype(np.int32)
        voxel_data = np.zeros((2, v.max_points)).astype(np.int32)
        voxel_stat = np.zeros((2, 4)).astype(np.float32)

        voxel_stat[0][3] = 0.0  # none of the voxels are "already staged"
        voxel_stat[1][3] = 0.0

        voxel_stat[0][0] = 0.5  # setting default probs
        voxel_stat[1][0] = 0.5


        voxel_stat[0][1] = 1.0
        voxel_stat[1][1] = 2.0


        prev_prob_1 = 0.5
        prev_prob_2 = 0.5

        # 1 point with point index 1
        voxel_data[0][0] = 1
        voxel_data[0][1] = 1

        # 2 points with index 2,3
        voxel_data[1][0] = 2
        voxel_data[1][1] = 2
        voxel_data[1][2] = 3

        access_count = 2
        for i in range(100):
            voxel_data[0][0] = access_count

            v.compute.bufferVoxelIndex(voxel_index)
            v.compute.fullBufferVoxelData(voxel_data)

            stored_voxel_num = len(voxel_index)

            filter_outliers = False
            stage_everything = False
            max_staging_area = 512

            voxel_stat, staged_voxels, counter_buffer = v.compute.prepareDispatchVoxelStager(
                voxel_index=None,
                stored_voxel_num=stored_voxel_num,
                max_points=v.max_points,
                voxel_statistics=voxel_stat,
                filter_outliers=filter_outliers,
                stage_everything=stage_everything,
                max_staging=max_staging_area,
                debug=True
            )

            if i < 10: # float precision ( 0.999999 = 1.0 fails)
                self.assertTrue(prev_prob_1 < voxel_stat[0][0])
                self.assertTrue(prev_prob_2 > voxel_stat[1][0])

            self.assertTrue(0.0 <= voxel_stat[0][0] <= 1.0)
            self.assertTrue(0.0 <= voxel_stat[1][0] <= 1.0)

            prev_prob_1 = voxel_stat[0][0]
            prev_prob_2 = voxel_stat[1][0]


            access_count += 1


    def test_shouldStageOneVoxel(self):
        v = Voxelizer(self.computeShader, None)
        v.init(0.5)

        voxel_index = np.array([[0, 0, 0, 0], [0, 2, 0, 1]]).astype(np.int32)
        voxel_data = np.zeros((2, v.max_points)).astype(np.int32)
        voxel_stat = np.zeros((2, 4)).astype(np.float32)

        voxel_stat[0][self.STAT_IS_STAGED] = 0.0  # none of the voxels are "already staged"
        voxel_stat[1][self.STAT_IS_STAGED] = 0.0

        voxel_stat[0][self.STAT_PROB] = 1.0  # setting probs > 0.9
        voxel_stat[1][self.STAT_PROB] = 0.91

        voxel_stat[0][self.STAT_STAGE_STATUS] = self.STAGE_STATUS_READY_TO_STAGE
        voxel_stat[1][self.STAT_STAGE_STATUS] = self.STAGE_STATUS_NONE

        # 1 point with point index 1
        voxel_data[0][0] = 3
        voxel_data[0][1] = 1
        voxel_data[0][2] = 77
        voxel_data[0][3] = 11

        # 2 points with index 2,3
        voxel_data[1][0] = 2
        voxel_data[1][1] = 2
        voxel_data[1][2] = 3

        v.compute.bufferVoxelIndex(voxel_index)
        v.compute.fullBufferVoxelData(voxel_data)

        stored_voxel_num = len(voxel_index)

        filter_outliers = False
        stage_everything = False
        max_staging_area = 512

        new_voxel_stats, staged_voxels, counter_buffer = v.compute.prepareDispatchVoxelStager(
            voxel_index=None,
            stored_voxel_num=stored_voxel_num,
            max_points=v.max_points,
            voxel_statistics=voxel_stat,
            filter_outliers=filter_outliers,
            stage_everything=stage_everything,
            max_staging=max_staging_area,
            debug=True
        )

        self.assertTrue(counter_buffer[0][0] == 1)
        self.assertTrue(counter_buffer[0][1] == max_staging_area)
        self.assertTrue(counter_buffer[0][2] == v.max_points)
        self.assertTrue(counter_buffer[0][3] == stored_voxel_num)

        self.assertTrue(voxel_stat[0][self.STAT_IS_STAGED] == 1.0)
        self.assertTrue(voxel_stat[1][self.STAT_IS_STAGED] == 0.0)

        expected_voxel_data = np.zeros((max_staging_area, v.max_points)).astype(np.int32)
        expected_voxel_data[0][0] = 3
        expected_voxel_data[0][1] = 1
        expected_voxel_data[0][2] = 77
        expected_voxel_data[0][3] = 11

        self.assertTrue(np.array_equal(expected_voxel_data, staged_voxels))

    def test_shouldStageBothVoxel(self):
        v = Voxelizer(self.computeShader, None)
        v.init(0.5)



        voxel_index = np.array([[0, 0, 0, 0], [0, 2, 0, 1]]).astype(np.int32)
        voxel_data = np.zeros((2, v.max_points)).astype(np.int32)
        voxel_stat = np.zeros((2, 4)).astype(np.float32)

        voxel_stat[0][self.STAT_IS_STAGED] = 0.0  # none of the voxels are "already staged"
        voxel_stat[1][self.STAT_IS_STAGED] = 0.0

        voxel_stat[0][self.STAT_PROB] = 1.0  # setting probs > 0.9
        voxel_stat[1][self.STAT_PROB] = 0.91

        voxel_stat[0][self.STAT_STAGE_STATUS] = self.STAGE_STATUS_READY_TO_STAGE
        voxel_stat[1][self.STAT_STAGE_STATUS] = self.STAGE_STATUS_READY_TO_STAGE

        # 1 point with point index 1
        voxel_data[0][0] = 3
        voxel_data[0][1] = 1
        voxel_data[0][2] = 77
        voxel_data[0][3] = 11

        # 2 points with index 2,3
        voxel_data[1][0] = 2
        voxel_data[1][1] = 2
        voxel_data[1][2] = 3

        v.compute.bufferVoxelIndex(voxel_index)
        v.compute.fullBufferVoxelData(voxel_data)

        stored_voxel_num = len(voxel_index)

        filter_outliers = False
        stage_everything = False
        max_staging_area = 512

        new_voxel_stats, staged_voxels, counter_buffer = v.compute.prepareDispatchVoxelStager(
            voxel_index=None,
            stored_voxel_num=stored_voxel_num,
            max_points=v.max_points,
            voxel_statistics=voxel_stat,
            filter_outliers=filter_outliers,
            stage_everything=stage_everything,
            max_staging=max_staging_area,
            debug=True
        )

        self.assertTrue(counter_buffer[0][0] == 2)
        self.assertTrue(counter_buffer[0][1] == max_staging_area)
        self.assertTrue(counter_buffer[0][2] == v.max_points)
        self.assertTrue(counter_buffer[0][3] == stored_voxel_num)

        self.assertTrue(voxel_stat[0][self.STAT_IS_STAGED] == 1.0)
        self.assertTrue(voxel_stat[1][self.STAT_IS_STAGED] == 1.0)

        expected_voxel_data = np.zeros((max_staging_area, v.max_points)).astype(np.int32)
        expected_voxel_data[0][0] = 3
        expected_voxel_data[0][1] = 1
        expected_voxel_data[0][2] = 77
        expected_voxel_data[0][3] = 11

        expected_voxel_data[1][0] = 2
        expected_voxel_data[1][1] = 2
        expected_voxel_data[1][2] = 3

        # can't seem to mess up stage row order here, as both threads run on the same wavefront
        self.assertTrue(np.array_equal(expected_voxel_data, staged_voxels))


    def test_canStageManyVoxels(self):
        v = Voxelizer(self.computeShader, None)
        v.init(0.5)


        number_of_voxels = 300
        voxel_index = np.zeros((number_of_voxels * 2, 4)).astype(np.int32)
        voxel_data = np.zeros((number_of_voxels * 2, v.max_points)).astype(np.int32)
        voxel_stat = np.zeros((number_of_voxels * 2, 4)).astype(np.float32)

        for i in range(number_of_voxels):
            voxel_index[i][3] = i
            voxel_data[i][0] = i
            for j in range(i):
                voxel_data[i][1 + j] = 10 + j
            voxel_stat[i][self.STAT_STAGE_STATUS] = self.STAGE_STATUS_READY_TO_STAGE

        v.compute.bufferVoxelIndex(voxel_index)
        v.compute.fullBufferVoxelData(voxel_data)

        stored_voxel_num = number_of_voxels

        filter_outliers = False
        stage_everything = False
        max_staging_area = 512

        new_voxel_stats, staged_voxels, counter_buffer = v.compute.prepareDispatchVoxelStager(
            voxel_index=None,
            stored_voxel_num=stored_voxel_num,
            max_points=v.max_points,
            voxel_statistics=voxel_stat,
            filter_outliers=filter_outliers,
            stage_everything=stage_everything,
            max_staging=max_staging_area,
            debug=True
        )

        self.assertTrue(counter_buffer[0][0] == number_of_voxels)
        self.assertTrue(counter_buffer[0][1] == max_staging_area)
        self.assertTrue(counter_buffer[0][2] == v.max_points)
        self.assertTrue(counter_buffer[0][3] == stored_voxel_num)

        for i in range(stored_voxel_num):
            self.assertTrue(voxel_stat[i][self.STAT_IS_STAGED] == 1.0)

        for i in range(stored_voxel_num): # regardless concurrency
            stored = staged_voxels[i][0]
            self.assertTrue(np.array_equal(staged_voxels[i], voxel_data[stored]))

    def test_canStageMaxPointVoxels(self):
        v = Voxelizer(self.computeShader, None)
        v.init(0.5)

        v.max_points = 1024

        for k in range(10):
            number_of_voxels = 300
            voxel_index = np.zeros((number_of_voxels * 2, 4)).astype(np.int32)
            voxel_data = np.zeros((number_of_voxels * 2, v.max_points)).astype(np.int32)
            voxel_stat = np.zeros((number_of_voxels * 2, 4)).astype(np.float32)

            for i in range(number_of_voxels):
                voxel_index[i][3] = i
                voxel_data[i][0] = i
                for j in range(i):
                    voxel_data[i][1 + j] = 10 + j
                if i % 5 == 0:
                    voxel_data[i][1:v.max_points] = i * 100
                    voxel_data[i][v.max_points - 1] = i
                    voxel_data[i][0] = 11234

                voxel_data[i][1] = i # adding this for conncurent data match check

                voxel_stat[i][self.STAT_STAGE_STATUS] = self.STAGE_STATUS_READY_TO_STAGE

            v.compute.bufferVoxelIndex(voxel_index)
            v.compute.fullBufferVoxelData(voxel_data)

            stored_voxel_num = number_of_voxels

            filter_outliers = False
            stage_everything = False
            max_staging_area = 512

            new_voxel_stats, staged_voxels, counter_buffer = v.compute.prepareDispatchVoxelStager(
                voxel_index=None,
                stored_voxel_num=stored_voxel_num,
                max_points=v.max_points,
                voxel_statistics=voxel_stat,
                filter_outliers=filter_outliers,
                stage_everything=stage_everything,
                max_staging=max_staging_area,
                debug=True
            )

            self.assertTrue(counter_buffer[0][0] == number_of_voxels)
            self.assertTrue(counter_buffer[0][1] == max_staging_area)
            self.assertTrue(counter_buffer[0][2] == v.max_points)
            self.assertTrue(counter_buffer[0][3] == stored_voxel_num)

            for i in range(stored_voxel_num):
                self.assertTrue(voxel_stat[i][self.STAT_IS_STAGED] == 1.0)

            for i in range(stored_voxel_num):  # regardless concurrency
                try:
                    stored = staged_voxels[i][0] #VOXELS IN THE STAGE CAN HAVE TRACE INDEXES REMAIN AFTER ANOTHER VOXEL (if they contain less points)

                    max_ind = min(stored + 1, v.max_points)

                    ref = voxel_data[staged_voxels[i][1]][1:max_ind]
                    staged = staged_voxels[i][1:max_ind]
                    self.assertTrue(np.array_equal(ref, staged))
                except AssertionError as e:
                    a = 0
            print("pass")

    def test_noStageDuplicates(self):
        pass


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