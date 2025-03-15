import time

import numpy as np

from PointcloudAlignment import PointcloudAlignment


class EnvironmentConstructor:


    def __init__(self, renderer, oxtsdatareader, lidardatareader, icpcontainer):
        self.renderer = renderer
        self.oxts = oxtsdatareader
        self.lidar = lidardatareader
        self.icp = icpcontainer

        self.framekeys = self.lidar.getfilenames()
        self.frameindex = 0

        self.rotation = np.eye(3)
        self.time = 0
        self.prev_points = None
        self.prev_position = None
        self.prev_velocity = None
        self.prev_rotation = None
        self.angular_velocity = None

        self.red = np.array([255, 0, 0])
        self.green = np.array([0, 255, 0])
        self.blue = np.array([0, 0, 255])
        self.white = np.array([255, 255, 255])
        self.colors = [self.red, self.green, self.blue, self.white]


    def getNextFrameData(self, offset):
        key = self.framekeys[self.frameindex + offset]

        lidar_points = self.lidar.getPoints(key)

        self.frameindex += 1
        if self.frameindex == len(self.framekeys):
            self.frameindex = 0

        return lidar_points

    def calculateTransition(self, current_lidar, point_to_plane=True):

        points, colors = current_lidar
        if self.prev_points is None:

            self.prev_position = np.array([0, 0, 0])
            self.prev_velocity = np.array([0, 0, 0])
            self.angular_velocity = np.eye(3)
            self.prev_rotation = np.eye(3)

            self.prev_points = points
            self.renderer.addPoints(points, colors)
            self.renderer.addPoints([self.prev_position], self.red)


        else:

            # estimated_rotation = self.angular_velocity @ self.prev_rotation
            estimated_rotation = self.angular_velocity @ self.prev_rotation
            translated_points = (estimated_rotation @ points.T).T + self.prev_position + self.prev_velocity

            if point_to_plane:
                t_opt, R_opt = self.icp.full_pt2pl(self.prev_points, translated_points, self.prev_position, iterations=20, renderer=self.renderer)
            else:
                t_opt, R_opt = self.icp.full_pt2pt(self.prev_points, translated_points, self.prev_position, iterations=20, renderer=self.renderer)

            """if point_to_plane:
                t_opt, R_opt = self.icp.full_pt2pl(self.prev_points, translated_points, self.prev_position,
                                                   iterations=20)
            else:
                t_opt, R_opt = self.icp.full_pt2pt(self.prev_points, translated_points, self.prev_position,
                                                   iterations=20)"""

            corrected_lidar_points = (R_opt @ translated_points.T).T - t_opt
            self.prev_points = corrected_lidar_points

            translation_delta = ((R_opt @ np.array([0, 0, 0]).T).T - t_opt) - np.array([0, 0, 0])


            curr_pos = self.prev_position
            self.prev_position = self.prev_position + self.prev_velocity + translation_delta
            self.prev_velocity = self.prev_position - curr_pos

            curr_rot = self.prev_rotation
            self.prev_rotation = R_opt @ self.angular_velocity @ self.prev_rotation
            self.angular_velocity = self.prev_rotation @ np.transpose(curr_rot)

            self.renderer.addPoints([self.prev_position], self.red)
            self.renderer.addPoints(corrected_lidar_points[::5], colors[::5])
            # time.sleep(1)


    def averageResults(self, prev, current):
        minlen = min(len(prev), len(current)) - 1
        return (prev[:minlen] + current[minlen])

