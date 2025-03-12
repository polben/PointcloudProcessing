import unittest
import numpy as np
import GpuMemoryManager  # Replace 'your_module' with the actual module name

class TestGpuMemoryManager(unittest.TestCase):

    def setUp(self):
        """Runs before every test case."""
        self.manager = GpuMemoryManager.GpuMemoryManager(maxNumberOfPoints=20)  # Small size for easy testing

    def test_basic_allocation(self):
        """Tests whether adding points works correctly."""
        points = np.random.rand(3, 3)  # 3 points
        start = self.manager.addPoints(points)
        self.assertEqual(start, 0)
        np.testing.assert_array_equal(self.manager.points[start:start+3], points)

    def test_multiple_allocations(self):
        """Tests multiple consecutive allocations."""
        p1 = np.random.rand(2, 3)  # 2 points
        p2 = np.random.rand(3, 3)  # 3 points

        start1 = self.manager.addPoints(p1)
        start2 = self.manager.addPoints(p2)

        self.assertEqual(start1, 0)
        self.assertEqual(start2, 2)

    def test_free_and_reallocate(self):
        """Tests freeing memory and reusing space."""
        p1 = np.random.rand(2, 3)
        p2 = np.random.rand(3, 3)

        start1 = self.manager.addPoints(p1)
        start2 = self.manager.addPoints(p2)

        self.manager.freeSpace(start1)  # Free first allocation

        p3 = np.random.rand(2, 3)
        start3 = self.manager.addPoints(p3)

        self.assertEqual(start3, 0)  # Should reuse the freed space

    def test_out_of_memory(self):
        """Tests that exceeding memory throws MemoryError."""
        p1 = np.random.rand(20, 3)
        self.manager.addPoints(p1)

        p2 = np.random.rand(1, 3)  # This should fail
        with self.assertRaises(MemoryError):
            self.manager.addPoints(p2)

    def test_invalid_free(self):
        """Tests that freeing an invalid range raises an error."""
        with self.assertRaises(ValueError):
            self.manager.freeSpace(5)  # No such allocation exists

    def test_writing_to_freed_blocks(self):
        """Tests if adjacent free spaces merge correctly."""
        p1 = np.random.rand(2, 3)
        p2 = np.random.rand(3, 3)
        p3 = np.random.rand(2, 3)

        start1 = self.manager.addPoints(p1)  # (0,2)
        start2 = self.manager.addPoints(p2)  # (2,5)
        start3 = self.manager.addPoints(p3)  # (5,7)

        self.manager.freeSpace(start1)  # Free first block
        self.manager.freeSpace(start2)  # Free second block

        # Insert new block of size 5
        p4 = np.random.rand(5, 3)
        start4 = self.manager.addPoints(p4)

        self.assertEqual(start4, 0)  # Should use merged space from (0,5)

    def test_shouldntDefragment(self):
        """
        Test the defragmentStep method to ensure that it moves blocks into available gaps.
        """
        # Add some points to create fragmentation
        self.manager.addPoints(np.zeros((5, 3)))  # (0, 5)
        self.manager.addPoints(np.zeros((1, 3)))  # (5, 6) (small gap)
        self.manager.addPoints(np.zeros((3, 3)))  # (6, 16)

        # Initially, memory is fragmented
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 9)])

        # Call defragmentStep to move the block (6, 16) into the gap (5, 6)
        result = self.manager.defragmentStep()

        # After one defragmentation, the blocks should be moved together:
        # (0, 5), (5, 15) (block moved into gap)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 9)])
        self.assertEqual(result, False)

    def test_shouldDefragmentSingleGap(self):
        """
        Test the defragmentStep method to ensure that it moves blocks into available gaps.
        """
        # Add some points to create fragmentation
        points1 = np.zeros((5, 3))
        points2 = np.zeros((1, 3))
        points3 = np.zeros((4, 3))
        self.manager.addPoints(points1)  # (0, 5)
        self.manager.addPoints(points2)  # (5, 6) (small gap)
        self.manager.addPoints(points3)  # (6, 16)

        # Initially, memory is fragmented
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 10)])

        self.manager.freeSpace(5)
        # Call defragmentStep to move the block (6, 16) into the gap (5, 6)
        result = self.manager.defragmentStep()
        self.assertEqual(result, True)

        # After one defragmentation, the blocks should be moved together:
        # (0, 5), (5, 15) (block moved into gap)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 9)])

        missingLength = self.manager.maxNumPoints - (len(points1) + len(points3))

        concatedPoints = np.concatenate((points1, points3, np.zeros((missingLength, 3))), axis=0)
        np.testing.assert_array_equal(self.manager.points, concatedPoints)

    def test_defragmentStep_multiple_gaps(self):
        """
        Test that defragmentStep works when there are multiple gaps between blocks.
        """
        # Add points with multiple gaps
        points1 = np.zeros((5, 3))  # (0, 5)
        points2 = np.zeros((1, 3))  # (5, 6)
        points3 = np.zeros((3, 3))  # (6, 9) (gap after 9)
        points4 = np.zeros((1, 3))  # (9, 10) (gap after 10)
        points5 = np.zeros((3, 3))  # (10, 13)

        self.manager.addPoints(points1)  # (0, 5)
        self.manager.addPoints(points2)  # (5, 6)
        self.manager.addPoints(points3)  # (6, 9)
        self.manager.addPoints(points4)  # (9, 10)
        self.manager.addPoints(points5)  # (10, 13)

        # Initial allocation state (multiple gaps between blocks)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 6), (6, 9), (9, 10), (10, 13)])

        # Free the (5, 6) space and check
        self.manager.freeSpace(5)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (6, 9), (9, 10), (10, 13)])

        # Free the (9, 10) space and check
        self.manager.freeSpace(9)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (6, 9), (10, 13)])


        # Now call defragmentStep — it should move (6, 9) into the gap between (0, 5) and (5, 6)
        self.manager.defragmentStep()

        # After defragmentation, the (6, 9) block should move to (5, 9)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 8), (10, 13)])


        # Call defragmentStep again — it should move (10, 13) into the gap (9, 10)
        self.manager.defragmentStep()

        # After second defragmentation, the (10, 13) block should move to (9, 13)
        self.assertEqual(self.manager.allocationIndexes, [(0, 5), (5, 8), (8, 11)])

        # Now the blocks are compacted into [(0, 5), (5, 9), (9, 13)]
        # Create the expected final concatenated points
        missingLength = self.manager.maxNumPoints - (len(points1) + len(points3) + len(points5))
        concatedPoints = np.concatenate((points1, points3, points5, np.zeros((missingLength, 3))), axis=0)

        # Check if the points array has been correctly defragmented
        np.testing.assert_array_equal(self.manager.points, concatedPoints)

if __name__ == '__main__':
    unittest.main()
