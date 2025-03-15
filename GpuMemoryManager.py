import bisect

import numpy as np


class GpuMemoryManager:

    points = None
    maxNumPoints = 0


    def __init__(self, maxNumberOfPoints = 1000000):
        self.maxNumPoints = maxNumberOfPoints
        self.points = np.zeros((maxNumberOfPoints, 6)).astype(np.float32)
        self.allocationIndexes = []


    def getFreeSpace(self, required):
        # begin
        if len(self.allocationIndexes) == 0:
            if required <= self.maxNumPoints:
                self.allocationIndexes.append((0, required))
                return 0, required
            else:
                raise MemoryError("Out of available point memory")

        # gap
        prev_end = 0
        for i, (current_begin, current_end) in enumerate(self.allocationIndexes):
            if current_begin - prev_end >= required:
                bisect.insort(self.allocationIndexes, (prev_end, prev_end + required))
                return prev_end, prev_end + required
            prev_end = current_end

        # end
        if self.maxNumPoints - prev_end >= required:
            self.allocationIndexes.append((prev_end, prev_end + required))
            return prev_end, prev_end + required

        raise MemoryError("Out of available point memory")



    def addPoints(self, np_points):
        if isinstance(np_points, list):
            np_points = np.array(np_points).astype(np.float32)

        required = len(np_points)
        start, end = self.getFreeSpace(required)
        self.points[start:end] = np_points.astype(np.float32)  # Copy points into allocated space


        return start  # Return index where points were inserted

    def freeSpace(self, start):
        for allocStart, allocEnd in self.allocationIndexes:
            if allocStart == start:
                self.allocationIndexes.remove((allocStart, allocEnd))
                self.points[allocStart : allocEnd] = np.zeros((allocEnd - allocStart, 6))
                return

        raise ValueError("Invalid range to free.")

    def defragmentStep(self):
        for i in range(len(self.allocationIndexes) - 1):
            current_start, current_end = self.allocationIndexes[i]
            next_start, next_end = self.allocationIndexes[i + 1]

            gap = next_start - current_end
            if gap > 0:
                size_to_move = next_end - next_start

                self.allocationIndexes[i + 1] = (current_end, current_end + size_to_move)
                self.points[current_end:current_end + size_to_move] = self.points[next_start:next_end]
                self.points[next_start:next_end] = np.zeros((next_end - next_start, 6))

                return True

        return False

    def getMaxPointIndex(self):
        if len(self.allocationIndexes) > 0:
            return self.allocationIndexes[-1][1]
        else:
            return 0

    def fullDefragment(self):
        while True:
            if not self.defragmentStep():
                break
