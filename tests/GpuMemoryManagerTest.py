import unittest
import numpy as np
import GpuMemoryManager

class TestGpuMemoryManager(unittest.TestCase):

    def setUp(self):
        self.manager = GpuMemoryManager.GpuMemoryManager(maxNumberOfPoints=20)  # Small size for easy testing
        self.tolerance = 1e-5 # 0.00001

    def test_shouldAddPoints(self):
        points = np.random.rand(3, 6)  # 3 points
        ptr = self.manager.addPoints(points)
        self.assertEqual(ptr, 124)
        start, end = self.manager.pointers[ptr]
        self.assertEqual(start, 0)
        self.assertEqual(end, 3)

        actual = self.manager.points[start:end]
        self.assertTrue(np.allclose(actual, points, atol=self.tolerance))

    def test_shouldAddPointclouds(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)

        ptr1 = self.manager.addPoints(p1)
        ptr2 = self.manager.addPoints(p2)

        loc1 = self.manager.pointers[ptr1]
        self.assertEqual(loc1, (0, 2))

        loc1 = self.manager.pointers[ptr2]
        self.assertEqual(loc1, (2, 5))


    def test_shouldFreePoints(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)

        ptr1 = self.manager.addPoints(p1)
        ptr2 = self.manager.addPoints(p2)

        old_ptr2_place = self.manager.pointers[ptr2] # inclusive, exclusive
        self.assertEqual(old_ptr2_place, (2, 5))

        self.manager.freeSpace(ptr1)

        new_ptr2_place = self.manager.pointers[ptr2]
        self.assertEqual(new_ptr2_place, (0, 3))

        p3 = np.random.rand(2, 6)
        ptr3 = self.manager.addPoints(p3)

        new_ptr3_place = self.manager.pointers[ptr3]
        self.assertEqual(new_ptr3_place, (3, 5))


    def test_shouldHandleGapFree(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)
        p3 = np.random.rand(2, 6)

        pt1 = self.manager.addPoints(p1)
        pt2 = self.manager.addPoints(p2)
        pt3 = self.manager.addPoints(p3)


        self.manager.freeSpace(pt2)

        loc1 = self.manager.pointers[pt1]
        self.assertEqual(loc1, (0,2))

        loc3 = self.manager.pointers[pt3]
        self.assertEqual(loc3, (2, 4))

    def test_shouldFreeEnd(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)

        pt1 = self.manager.addPoints(p1)
        pt2 = self.manager.addPoints(p2)

        self.manager.freeSpace(pt2)

        self.assertEqual(len(self.manager.allocationIndexes), 1)
        self.assertEqual(len(self.manager.pointers), 1)

        with self.assertRaises(ValueError):
            self.manager.freeSpace(pt2)

        self.assertFalse(self.manager.freeSpace(pt1))

    def test_shouldRaiseOutOfMemory(self):
        p1 = np.random.rand(20, 6)
        self.manager.addPoints(p1)

        p2 = np.random.rand(1, 6)  # This should fail
        with self.assertRaises(MemoryError):
            self.manager.addPoints(p2)

    def test_shouldFailOnFree(self):
        with self.assertRaises(ValueError):
            self.manager.freeSpace(5)  # No such allocation exists

    def test_canTrackRanges(self):
        p1 = np.random.rand(10, 6)
        self.assertEqual(self.manager.getMaxPointIndex(), 0)

        ptr1 = self.manager.addPoints(p1)
        self.assertEqual(self.manager.getMaxPointIndex(), 10)

        self.manager.freeSpace(ptr1)
        self.assertEqual(self.manager.getMaxPointIndex(), 0)

        ptr1 = self.manager.addPoints(p1)
        self.assertEqual(self.manager.getMaxPointIndex(), 10)
        self.manager.markLastBufferPoint()
        self.assertEqual(self.manager.last_buffered_point, 10)

        ptr2 = self.manager.addPoints(p1)
        self.assertEqual(self.manager.getMaxPointIndex(), 20)
        self.manager.markLastBufferPoint()
        self.assertEqual(self.manager.last_buffered_point, 20)

        self.manager.freeSpace(ptr1)
        self.assertEqual(self.manager.getMaxPointIndex(), 10)
        self.assertEqual(self.manager.last_buffered_point, 0)


    def test_shouldWriteToFreeBlocks(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)
        p3 = np.random.rand(2, 6)

        pt1 = self.manager.addPoints(p1)  # (0,2)
        pt2 = self.manager.addPoints(p2)  # (2,5)
        pt3 = self.manager.addPoints(p3)  # (5,7)

        self.manager.freeSpace(pt1)  # Free first block
        self.manager.freeSpace(pt2)  # Free second block

        # Insert new block of size 5
        p4 = np.random.rand(5, 6)
        start4 = self.manager.addPoints(p4)

        loc_first = self.manager.pointers[pt3]
        self.assertEqual(loc_first, (0, 2))

        loc_last = self.manager.pointers[start4]
        self.assertEqual(loc_last, (2, 7))  # Should use merged space from (0,5)

    def test_shouldntDefragment(self):
        self.manager.addPoints(np.zeros((5, 6)))
        self.manager.addPoints(np.zeros((1, 6))) #
        self.manager.addPoints(np.zeros((3, 6)))

        self.assertEqual(self.manager.allocationIndexes, [(0, 5, 124), (5, 6, 125), (6, 9, 126)])

        result = self.manager.fullDefragment()

        self.assertEqual(self.manager.allocationIndexes, [(0, 5, 124), (5, 6, 125), (6, 9, 126)])
        self.assertEqual(result, False)



if __name__ == '__main__':
    unittest.main()
