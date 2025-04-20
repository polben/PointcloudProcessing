import threading
import time

import numpy as np

from OutlierFilter import OutlierFilter
from Voxelizer import Voxelizer


class EnvironmentConstructor:


    def __init__(self, renderer, oxtsdatareader, lidardatareader, icpcontainer, computeShader):
        self.prev_position = None
        self.renderer = renderer
        self.oxts = oxtsdatareader
        self.lidar = lidardatareader
        self.icp = icpcontainer
        self.compute = computeShader

        self.voxelizer = Voxelizer(self.compute, self.renderer)

        self.framekeys = None
        self.frameindex = None

        self.reference_points = None
        self.prev_origin = None

        self.red = np.array([255, 0, 0])
        self.green = np.array([0, 255, 0])
        self.blue = np.array([0, 0, 255])
        self.white = np.array([255, 255, 255])
        self.colors = [self.red, self.green, self.blue, self.white]

        self.local_frame_counter = None
        self.total_frame_counter = None
        self.reset_origin_treshold = 5




        self.prev_poses = None

        self.inited = False


    def init(self, voxel_size):
        self.inited = False

        self.framekeys = self.lidar.getfilenames()
        self.frameindex = 0
        self.local_frame_counter = 0
        self.total_frame_counter = 0
        self.prev_poses = []

        self.voxelizer.init(voxel_size)

        self.reference_points = None
        self.prev_origin = None

        self.inited = True


    def cleanup(self):
        self.inited = False

        self.frameindex = 0
        self.total_frame_counter = 0
        self.local_frame_counter = 0
        self.prev_poses = []
        self.framekeys = []
        self.reference_points = None
        self.prev_origin = None

        self.voxelizer.cleanup()

    def getNextFrameData(self, offset):
        key = self.framekeys[self.frameindex + offset]

        lidar_points = self.lidar.getPointsWithIntensities(key)
        oxts = self.oxts.getOx(key)

        self.frameindex += 1
        if self.frameindex == len(self.framekeys):
            self.frameindex = 0

        return lidar_points, oxts


    def getDeltaVelocity(self, current_velocity, current_rotation, current_time, prev_time):
        delta_time = (current_time - prev_time).total_seconds()
        current_velocity *= delta_time
        delta_current_velocity = (current_rotation @ current_velocity.T).T

        return delta_current_velocity

    def getRefinedPosition(self, r_opt, t_opt, estimated_position):
        refined_position = (r_opt @ estimated_position.T).T - t_opt
        return refined_position

    def applyOptimalTranslation(self, r_opt, t_opt, points):
        return (r_opt @ points.T).T - t_opt

    def getCurrentOxtsData(self, current_oxts):
        return -current_oxts.getVelocity(), current_oxts.getTrueRotation(np.eye(3)), current_oxts.getYawRotation(np.eye(3)), current_oxts.getTime()

    def cullColorPoints(self, points, colors, cullColors = False):
        if not cullColors:
            return points, colors

        defaultColor = self.lidar.DEFAULT_COLOR
        mask = np.all(colors != defaultColor, axis=1)
        return points[mask], colors[mask]


    def getRefinedTransition(self, reference_points, reference_origin, points_to_refine, iterations, point_to_plane, renderer=None, debug=False, full_iter=False):
        if debug:
            if point_to_plane:
                t_opt, R_opt = self.icp.full_pt2pl(reference_points, points_to_refine, reference_origin,
                                                   iterations=iterations, renderer=renderer, full_iter=full_iter)
            else:
                t_opt, R_opt = self.icp.full_pt2pt(reference_points, points_to_refine, reference_origin,
                                                   iterations=iterations, renderer=renderer)
        else:
            if point_to_plane:
                t_opt, R_opt = self.icp.full_pt2pl(reference_points, points_to_refine, reference_origin,
                                                   iterations=iterations, full_iter=full_iter)
            else:
                t_opt, R_opt = self.icp.full_pt2pt(reference_points, points_to_refine, reference_origin,
                                                   iterations=iterations)

        return t_opt, R_opt




    def calculateTransition_imu(self, current_lidar, current_oxts, point_to_plane=True, debug=False, iterations=20, separate_colors=False, removeOutliers=False, pure_imu=False):

        points, colors, intensities = current_lidar

        if not separate_colors:
            colors = intensities

        current_velocity, current_horz_rot, current_vert_rot, current_time = self.getCurrentOxtsData(current_oxts=current_oxts)
        current_horz_rot = np.linalg.inv(current_horz_rot)

        current_rotation = current_vert_rot @ current_horz_rot



        if self.total_frame_counter == 0:
            self.reference_points = (current_rotation @ points.T).T
            self.prev_origin = np.array([0, 0, 0])
            self.prev_position = np.array([0,0,0])

            self.renderer.addPoints([self.prev_position], self.red)


            imu_position = np.array([0.0, 0.0, 0.0])
            refined_position = np.array([0.0, 0.0, 0.0])
            imu_rotation = current_rotation
            refined_rotation = current_rotation
            c_time = current_time
            self.prev_poses.append((imu_position, refined_position, imu_rotation, refined_rotation, c_time))

            self.voxelizer.separate_colors = separate_colors
            self.voxelizer.filter_outliers = removeOutliers

            self.total_frame_counter += 1


            self.voxelizer.addPoints(self.reference_points, colors)




        else:
            total_frame_time = time.time()

            start_alignment = time.time()
            p_imu_pos, p_refined_pos, p_imu_rotation, p_refined_rotation, p_time = self.prev_poses[self.total_frame_counter - 1]

            delta_velocity = self.getDeltaVelocity( current_velocity=current_velocity,
                                                    current_time=current_time,
                                                    current_rotation=current_rotation,
                                                    prev_time=p_time )

            estimated_position = p_refined_pos + delta_velocity
            next_pure_imu_position = p_imu_pos + delta_velocity


            aligned_points = (current_rotation @ points.T).T + estimated_position
            print("alignment time: " + str(time.time()-start_alignment))


            start_refinement = time.time()
            if not pure_imu:

                full_iteration = False
                iterations_to_do = iterations
                if self.local_frame_counter + 1 >= self.reset_origin_treshold:
                    full_iteration = True
                    iterations_to_do = 15


                t_opt, R_opt = self.getRefinedTransition(reference_points=self.reference_points,
                                                         reference_origin=self.prev_origin,
                                                         points_to_refine=aligned_points,
                                                         iterations=iterations_to_do,
                                                         point_to_plane=point_to_plane,
                                                         renderer=self.renderer,
                                                         debug=debug,
                                                         full_iter=full_iteration)


                refined_points = self.applyOptimalTranslation(r_opt=R_opt, t_opt=t_opt, points=aligned_points)
                refined_position = self.getRefinedPosition(r_opt=R_opt, t_opt=t_opt, estimated_position=estimated_position)
                refined_rotation = R_opt @ current_rotation
            else:
                refined_points = aligned_points
                refined_position = estimated_position
                refined_rotation = current_rotation
                if debug:
                    time.sleep(1)
            print("refinement time: " + str(time.time()-start_refinement))

            start_voxel = time.time()
            self.voxelizer.addPoints(refined_points, colors)
            print("voxelization: " + str(time.time()-start_voxel))

            self.local_frame_counter += 1

            if self.local_frame_counter >= self.reset_origin_treshold:
                print("reference update")
                self.reference_points = refined_points
                self.prev_origin = refined_position
                self.local_frame_counter = 0


            self.prev_poses.append((next_pure_imu_position, refined_position, current_rotation, refined_rotation, current_time))


            self.renderer.addPoints([next_pure_imu_position], self.red)
            self.renderer.addPoints([refined_position], self.green)




            self.total_frame_counter += 1


            print("frame " + str(self.total_frame_counter) + " time: " + str(time.time() - total_frame_time))
            print("---\n")




    def coloreq(self, color1, color2):
        return np.array_equal(color1, color2)