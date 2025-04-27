import array
import os

import numpy as np

from PIL import Image


class LidarDataReader:



    VELODYNE_DATA = "//velodyne_points//data//"

    DEFAULT_COLOR = np.array([255, 178, 102])

    def __init__(self):
        self.datamap = {}
        self.currentName = 0
        self.root = None
        self.calibrationPath = None
        self.width = None
        self.height = None
        self.targetCamera = None
        self.IMAGES = None
        self.R_velo_to_cam = None
        self.t_velo_to_cam = None
        self.proj = None
        self.rect_00 = None
        self.c2c_rot = None
        self.count = None
        self.MAX_DATA_READ = None

        self.inited = False

    def init(self, path, calibration, targetCamera="02", max_read=10, ui=None):


        self.root = path
        self.calibrationPath = calibration
        self.width = 0
        self.height = 0
        self.targetCamera = targetCamera
        self.IMAGES = "//image_" + self.targetCamera + "//data//"

        self.R_velo_to_cam, self.t_velo_to_cam = self.getVeloToCam(self.calibrationPath)
        self.proj, self.rect_00, self.c2c_rot = self.getCameraCalibration(self.calibrationPath)

        self.count = 0

        self.MAX_DATA_READ = max_read

        files = os.listdir(self.root + self.VELODYNE_DATA)

        for filename in files:
            if filename.endswith(".bin"):
                np_points, intensities = self.eatBytes(filename)
                # filter car artifacts
                dists = np.linalg.norm(np_points, axis=1)
                m = 2.5
                np_points, intensities = np_points[dists > m], intensities[dists > m]

                image = self.readImage(filename.strip(".bin") + ".png")
                # np_points = self.filter_points(np_points)

                rot = LidarDataReader.aligningRotation()  # this actually is very close to the velo to cam rotation
                points_3d = (rot @ np_points.T).T

                # points_3d = np_points

                # https://github.com/sinaenjuni/Sensor_fusion_of_LiDAR_and_Camera_from_KITTI_dataset
                lidar_to_view = self.R_velo_to_cam @ np_points.T
                cam_0_points = lidar_to_view + self.t_velo_to_cam
                cam_0_rect = self.rect_00 @ cam_0_points

                color_indexes, colors, points2d = self.project_np(cam_0_rect.T, self.proj, self.width, self.height,
                                                                  image)

                int_cls = np.tile(intensities.T.reshape(-1, 1), 3)
                colors = self.createColors(points_3d, color_indexes, colors, int_cls)
                self.datamap[filename.strip('.bin')] = (points_3d, colors, int_cls)
                self.count += 1

                if ui is not None:
                    ui.setFrameCounter(self.count, len(files))

            print(str(len(files)) + " / " + str(self.count))
            if self.count >= self.MAX_DATA_READ:
                break

        self.inited = True

    def cleanup(self):
        self.inited = False

        keys = list(self.datamap.keys())
        for k in keys:
            del self.datamap[k]


    def createColors(self, points, color_indexes, colors, int_cls):
        point_count = len(points)
        # ret_colors = np.tile(self.DEFAULT_COLOR, (point_count, 1))
        ret_colors = int_cls.copy()
        ret_colors[color_indexes] = colors / 255.0
        return ret_colors

    def project_np(self, points, projection_mat, width, height, image):

        points = np.hstack([points, np.ones((len(points), 1))])
        indexes = np.arange(0, len(points))

        points2d = projection_mat @ points.T

        mask = points2d[2] > 1e-6
        points2d = points2d[:, mask] / points2d[2][mask]
        indexes = indexes[mask]
        depths = points2d[2]

        valid_in_image_mask = (
                (points2d[0] >= 0) & (points2d[0] <= width) &  # x-coordinates
                (points2d[1] >= 0) & (points2d[1] <= height)  # y-coordinates
        )
        points2d = points2d[:, valid_in_image_mask]
        indexes = indexes[valid_in_image_mask]
        depths = depths[valid_in_image_mask]

        points2d[1] = (self.height - 1) - points2d[1]

        color_x = np.round(np.clip(points2d[0], 0, self.width - 1)).astype(np.int32)
        color_y = (self.height - 1) - np.round(np.clip(points2d[1], 0, self.height - 1)).astype(np.int32)

        colors = image[color_y, color_x]

        return indexes, colors, points2d
        # return self.filterOnDepth(color_x, color_y, image, depths, indexes, points2d)


    @staticmethod
    def aligningRotation():
        rot = LidarDataReader.rotation(-np.pi / 2, 0, np.pi / 2)
        return rot

    def eatBytes(self, filename):
        with open(self.root + self.VELODYNE_DATA + filename, "rb") as f:
            bin_data = f.read()
            float_data = array.array('f', bin_data)
            numpy_array = np.frombuffer(float_data, dtype=np.float32)
            reshaped_array = numpy_array.reshape(-1, 4)
        return reshaped_array[:, :3].astype(np.float32), reshaped_array[:, 3].astype(np.float32)

    def readImage(self, filename):
        with open(self.root + self.IMAGES + filename, "rb") as f:
            image = Image.open(f)  # Load image
            self.width = image.width
            self.height = image.height

            return np.array(image)  # Convert to NumPy array



    def getPoints(self, filename):
        return np.ascontiguousarray(self.datamap[filename][0]), np.ascontiguousarray(self.datamap[filename][1]) # the actual points,

    def getPointsWithIntensities(self, filename):
        return np.ascontiguousarray(self.datamap[filename][0]), np.ascontiguousarray(self.datamap[filename][1]), self.datamap[filename][2] # the actual points,



    def getfilenames(self):
        return list(self.datamap.keys())



    @staticmethod
    def rotation(x, y, z):


        xr = x
        yr = y
        zr = z

        R_z = np.array([
            [np.cos(zr), -np.sin(zr), 0],
            [np.sin(zr), np.cos(zr), 0],
            [0, 0, 1]
        ])

        R_y = np.array([
            [np.cos(yr), 0, np.sin(yr)],
            [0, 1, 0],
            [-np.sin(yr), 0, np.cos(yr)]
        ])

        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(xr), -np.sin(xr)],
            [0, np.sin(xr), np.cos(xr)]
        ])


        return R_x @ R_y @ R_z

    def getVeloToCam(self, calibrationPath):
        with open(self.calibrationPath + "//calib_velo_to_cam.txt", "r") as f:
            lines = f.readlines()

            R_values = list(map(float, lines[1].split(":")[1].strip().split())) # R:
            R = np.array(R_values).reshape(3, 3)

            T_values = list(map(float, lines[2].split(":")[1].strip().split())) # T:
            T = np.array(T_values).reshape(3, 1)

            return R, T

    def getCameraCalibration(self, calibrationPath):

        target_cam_proj = "P_rect_" + self.targetCamera
        target_cam_rect = "R_rect_00"

        # cam0_to_cam_target = "T_" + self.targetCamera
        cam0_to_cam_target_rot = "R_" + self.targetCamera

        with open(self.calibrationPath + "//calib_cam_to_cam.txt", "r") as f:
            lines = f.readlines()

            line_proj = None
            line_c2c_t = None
            line_rect = None
            line_c2c_r = None
            for l in lines:
                parts = l.split(":")
                if parts[0] == target_cam_proj:
                    line_proj = parts[1]

                """if parts[0] == cam0_to_cam_target:
                    line_c2c_t = parts[1]
                """
                if parts[0] == target_cam_rect:
                    line_rect = parts[1]

                if parts[0] == cam0_to_cam_target_rot:
                    line_c2c_r = parts[1]


        proj_values = list(map(float, line_proj.strip().split()))
        proj = np.array(proj_values).reshape(3, 4)

        rect_00_values = list(map(float, line_rect.strip().split()))
        rect_00 = np.array(rect_00_values).reshape(3, 3)

        """c2c_t_values = list(map(float, line_c2c_t.strip().split()))
        c2c_t = np.array(c2c_t_values).reshape(3, 1)  # Convert to 3x1 column vector
        """

        c2c_rot_values = list(map(float, line_c2c_r.strip().split()))
        c2c_rot = np.array(c2c_rot_values).reshape(3, 3)

        return proj, rect_00, c2c_rot

