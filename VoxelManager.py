from xml.etree.ElementPath import prepare_self

import numpy as np


class VoxelManager:


    def __init__(self, voxelizer, renderer):
        self.voxelizer = voxelizer
        self.renderer = renderer

        self.frame_counter = 0
        self.trackedVoxels = {}

        self.FLAG_STATIC = 0
        self.FLAG_FREE_SPACE = 1
        self.FLAG_PENDING = 2

        self.D_ENTRY_COUNTER = 0
        self.D_PREV_COUNT = 1
        self.D_FLAG = 2

        self.a = 0.55
        self.b = 0.45

        self.prev_frame = None

    def frameVoxelized(self, frame_points):
        self.frame_counter += 1

        if self.prev_frame is not None:
            self.renderer.freePoints(self.prev_frame)

        self.prev_frame = self.renderer.addPoints(frame_points, np.array([100,100,100]))

        for i in range(self.voxelizer.getStoredVoxelCount()):
            voxel_data_index = self.voxelizer.getVoxelDataIndexAt(i)

            if voxel_data_index not in self.trackedVoxels:
                self.trackedVoxels[voxel_data_index] = 0.5, self.voxelizer.getStoredCount(voxel_data_index), None, False, False
            else:
                old_prob, old_count, render_id, final_render, final_static = self.trackedVoxels[voxel_data_index]

                if final_static:
                    continue


                new_count = self.voxelizer.getStoredCount(voxel_data_index)
                if final_render:
                    if new_count < self.voxelizer.max_points - 1:
                        continue
                    else:
                        self.renderer.freePoints(render_id)
                        render_id = self.renderer.addPoints(self.voxelizer.getStoredPoints(voxel_data_index), np.array([255,0,0]))
                        final_static = True


                if new_count != old_count:
                    new_prob = self.BayesFilter(old_prob, self.a, self.b)
                else:
                    new_prob = self.BayesFilter(old_prob, 1.0 - self.a, 1.0 - self.b)


                if new_prob > 0.9:
                    final_render = True
                    render_id = self.renderer.addPoints(self.voxelizer.getStoredPoints(voxel_data_index),
                                                        self.randcolor())




                if new_prob >= 0.5 and render_id is None:
                    # state change to static
                    pass
                    # render_id = self.renderer.addPoints(self.voxelizer.getStoredPoints(voxel_data_index), self.randcolor())

                if new_prob < 0.5 and render_id is not None:
                    # state change to dynamic
                    pass
                    # self.renderer.freePoints(render_id)
                    # render_id = None
                    # final_render = False

                self.trackedVoxels[voxel_data_index] = new_prob, new_count, render_id, final_render, final_static

    def BayesFilter(self, p, Ps, Pd):
        return (Ps * p) / (Ps * p + Pd * (1.0 - p))






        a = 0


    def randcolor(self):
        return np.random.rand(3, )