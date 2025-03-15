import unittest
import numpy as np
import GpuMemoryManager  # Replace 'your_module' with the actual module name

class TestGpuMemoryManager(unittest.TestCase):

    def setUp(self):
        self.manager = GpuMemoryManager.GpuMemoryManager(maxNumberOfPoints=20)  # Small size for easy testing
        self.tolerance = 1e-5 # 0.00001

    def test_shouldAddPoints(self):
        points = np.random.rand(3, 6)  # 3 points
        start = self.manager.addPoints(points)
        self.assertEqual(start, 0)
        actual = self.manager.points[start:start+3]
        self.assertTrue(np.allclose(actual, points, atol=self.tolerance))

    def test_shouldAddPointclouds(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)

        start1 = self.manager.addPoints(p1)
        start2 = self.manager.addPoints(p2)

        self.assertEqual(start1, 0)
        self.assertEqual(start2, 2)

    def test_shouldFreePoints(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)

        start1 = self.manager.addPoints(p1)
        start2 = self.manager.addPoints(p2)

        self.manager.freeSpace(start1)
        p3 = np.random.rand(2, 6)
        start3 = self.manager.addPoints(p3)

        self.assertEqual(start3, 0)

    def test_shouldRaiseOutOfMemory(self):
        p1 = np.random.rand(20, 6)
        self.manager.addPoints(p1)

        p2 = np.random.rand(1, 6)  # This should fail
        with self.assertRaises(MemoryError):
            self.manager.addPoints(p2)

    def test_shouldFailOnFree(self):
        with self.assertRaises(ValueError):
            self.manager.freeSpace(5)  # No such allocation exists

    def test_shouldWriteToFreeBlocks(self):
        p1 = np.random.rand(2, 6)
        p2 = np.random.rand(3, 6)
        p3 = np.random.rand(2, 6)

        start1 = self.manager.addPoints(p1)  # (0,2)
        start2 = self.manager.addPoints(p2)  # (2,5)
        start3 = self.manager.addPoints(p3)  # (5,7)

        self.manager.freeSpace(start1)  # Free first block
        self.manager.freeSpace(start2)  # Free second block

        # Insert new block of size 5
        p4 = np.random.rand(5, 6)
        start4 = self.manager.addPoints(p4)

        self.assertEqual(start4, 0)  # Should use merged space from (0,5)

    def test_shouldntDefragment(self):
        self.manager.addPoints(np.zeros((5, 6)))
        self.manager.addPoints(np.zeros((1, 6))) # gap
        self.manager.addPoints(np.zeros((3, 6)))

        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 9)])

        result = self.manager.defragmentStep()

        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 9)])
        self.assertEqual(result, False)

    def test_shouldDefragmentSingleGap(self):
        points1 = np.zeros((5, 6))
        points2 = np.zeros((1, 6))
        points3 = np.zeros((4, 6))
        self.manager.addPoints(points1)
        self.manager.addPoints(points2) #gap
        self.manager.addPoints(points3)

        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 10)])

        self.manager.freeSpace(5)
        result = self.manager.defragmentStep()
        self.assertEqual(result, True)

        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 9)])

        missingLength = self.manager.maxNumPoints - (len(points1) + len(points3))

        concatedPoints = np.concatenate((points1, points3, np.zeros((missingLength, 6))), axis=0)
        self.assertTrue(np.allclose(self.manager.points, concatedPoints, atol=self.tolerance))

    def test_defragmentStep_multiple_gaps(self):
        points1 = np.zeros((5, 6))
        points2 = np.zeros((1, 6))
        points3 = np.zeros((3, 6))
        points4 = np.zeros((1, 6))
        points5 = np.zeros((3, 6))

        self.manager.addPoints(points1)  # (0, 5)
        self.manager.addPoints(points2)  # (5, 6)
        self.manager.addPoints(points3)  # (6, 9)
        self.manager.addPoints(points4)  # (9, 10)
        self.manager.addPoints(points5)  # (10, 13)

        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 9), (9, 10), (10, 13)])

        self.manager.freeSpace(5)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (6, 9), (9, 10), (10, 13)])

        self.manager.freeSpace(9)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (6, 9), (10, 13)])


        self.manager.defragmentStep()

        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 8), (10, 13)])


        self.manager.defragmentStep()

        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 8), (8, 11)])

        missingLength = self.manager.maxNumPoints - (len(points1) + len(points3) + len(points5))
        concatedPoints = np.concatenate((points1, points3, points5, np.zeros((missingLength, 6))), axis=0)

        self.assertTrue(np.allclose(self.manager.points, concatedPoints, atol=self.tolerance))

if __name__ == '__main__':
    unittest.main()
