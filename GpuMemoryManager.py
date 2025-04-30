import numpy as np


class GpuMemoryManager:

    points = None
    maxNumPoints = 0


    def __init__(self, maxNumberOfPoints = 1000000):
        self.maxNumPoints = maxNumberOfPoints
        self.points = np.zeros((maxNumberOfPoints, 6)).astype(np.float32)
        self.allocationIndexes = []
        self.last_buffered_point = 0
        self.pointers = {}
        self.currentPointer = 123

    def markLastBufferPoint(self):
        self.last_buffered_point = self.getMaxPointIndex()

    def getFreeSpace(self, required):
        self.currentPointer += 1

        # begin
        if len(self.allocationIndexes) == 0:
            if required <= self.maxNumPoints:
                self.allocationIndexes.append((0, required, self.currentPointer))
                return 0, required, self.currentPointer
            else:
                raise MemoryError("Out of available point memory")

        prev_end = self.allocationIndexes[-1][1]

        # end
        if self.maxNumPoints - prev_end >= required:
            self.allocationIndexes.append((prev_end, prev_end + required, self.currentPointer))
            return prev_end, prev_end + required, self.currentPointer

        raise MemoryError("Out of available point memory")



    def addPoints(self, np_points):
        if isinstance(np_points, list):
            np_points = np.array(np_points).astype(np.float32)

        required = len(np_points)
        start, end, pointer = self.getFreeSpace(required)
        self.points[start:end] = np_points.astype(np.float32)  # Copy points into allocated space
        self.pointers[pointer] = start, end

        return pointer  # Return index where points were inserted

    def freeSpace(self, pointer):
        if pointer in self.pointers:
            points_start, points_end = self.pointers[pointer]


            self.allocationIndexes.remove((points_start, points_end, pointer))
            del self.pointers[pointer]

            # self.points[points_start : points_end] = np.zeros((points_end - points_start, 6))

            self.last_buffered_point = 0
            return self.fullDefragment()
        else:
            raise ValueError("Invalid range to free.")

    def dropPointers(self):
        self.allocationIndexes = []
        keys = list(self.pointers.keys())
        for k in keys:
            del self.pointers[k]

        self.last_buffered_point = 0

    def getMaxPointIndex(self):
        if len(self.allocationIndexes) > 0:
            return self.allocationIndexes[-1][1]
        else:
            return 0

    def fullDefragment(self):
        prev_end = 0
        was_fragmented = False
        for i in range(len(self.allocationIndexes)):
            current_start, current_end, pointer = self.allocationIndexes[i]
            size = current_end - current_start
            if prev_end != current_start:
                self.points[prev_end:prev_end + size] = self.points[current_start:current_end]
                self.allocationIndexes[i] = (prev_end, prev_end + size, pointer)
                self.pointers[pointer] = (prev_end, prev_end + size)
                was_fragmented = True

            prev_end = prev_end + size

        return was_fragmented
