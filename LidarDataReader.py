import array
import os
from datetime import datetime

import numpy as np

from utils import UNDEFINED
from PIL import Image

class LidarDataReader:

    root = UNDEFINED
    currentName = 0
    datamap = {}
    timestamps = []

    VELODYNE_DATA = "//velodyne_points//data//"
    TIMESTAMPS = "//velodyne_points//timestamps.txt"

    oxtsDataReader = None

    def getRoot(self, path):
        files = os.listdir(path)
        if len(files) == 1:
            return self.getRoot(path + "//" + files[0])
        else:
            return path



    def __init__(self, path, oxtsDataReader, calibration, targetCamera, max_read=10):
        self.root = self.getRoot(path)
        self.oxtsDataReader = oxtsDataReader
        self.calibrationPath = calibration
        self.width = 0
        self.height = 0
        self.targetCamera = targetCamera
        self.IMAGES = "//image_" + self.targetCamera + "//data//"

        self.R_velo_to_cam, self.t_velo_to_cam = self.getVeloToCam(self.calibrationPath)
        self.proj, self.rect_00, self.c2c_rot = self.getCameraCalibration(self.calibrationPath)

        count = 0

        self.MAX_DATA_READ = max_read
        self.DEFAULT_COLOR = np.array([255, 178, 102])

        self.timestamps = self.readTimestamps(self.root + self.TIMESTAMPS)


        files = os.listdir(self.root + self.VELODYNE_DATA)



        for filename in files:
            if filename.endswith(".bin"):
                np_points = self.eatBytes(filename)
                image = self.readImage(filename.strip(".bin") + ".png")
                np_points = self.filter_points(np_points)


                rot = LidarDataReader.aligningRotation() # this actually is very close to the velo to cam rotation
                points_3d = (rot @ np_points.T).T


                # points_3d = np_points

                # https://github.com/sinaenjuni/Sensor_fusion_of_LiDAR_and_Camera_from_KITTI_dataset
                lidar_to_view = self.R_velo_to_cam @ np_points.T
                cam_0_points = lidar_to_view + self.t_velo_to_cam
                cam_0_rect =  self.rect_00 @ cam_0_points


                color_indexes, colors, points2d = self.project_np(cam_0_rect.T, self.proj, self.width, self.height, image)


                # points2d *= 0.001
                # self.datamap[filename.strip('.bin')] = (points2d.T, self.timestamps[count], colors)

                colors = self.createColors(points_3d, color_indexes, colors)
                self.datamap[filename.strip('.bin')] = (points_3d, self.timestamps[count], colors)
                count += 1

            print(str(len(files)) + " / " + str(count))
            if count >= self.MAX_DATA_READ:
                break

        if not len(self.datamap.keys()) == len(self.timestamps):
            raise RuntimeError("Timestamps and farmes do not match: keys[" + str(len(self.datamap.keys())) + "] timestamps[" + str(len(self.timestamps)) + "]")
        else:
            print("Frames and timestamps loaded")

    def filter_points(self, np_points):

        plane_normal = np.array([0, 0, 1])
        lidar_height = 1.73
        lidar_offset = np.array([0, 0, -lidar_height])

        mask_plane = self.mask_ground(np_points, plane_normal, lidar_offset, 0.3)
        #return np_points[mask_plane]


        distances = np.linalg.norm(np_points, axis = 1)
        max = np.max(distances) / 3.0
        probabilities = (distances / max) ** 2
        randoms = np.random.rand(len(probabilities))
        mask_distance = probabilities > randoms
        # return np_points[mask_distance]


        return np_points[~mask_plane | mask_distance]

        a= 0

    def mask_ground(self, points, normal, point_on_plane, threshold):

        a, b, c = normal
        x0, y0, z0 = point_on_plane

        # Compute the plane's d constant
        d = -(a * x0 + b * y0 + c * z0)

        # Compute distances of all points to the plane
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.linalg.norm(normal)

        # Filter points that are farther than the threshold
        return distances < threshold

    def createColors(self, points, color_indexes, colors):
        point_count = len(points)
        ret_colors = np.tile(self.DEFAULT_COLOR, (point_count, 1))
        ret_colors[color_indexes] = colors
        return ret_colors

    def project_np(self, points, projection_mat, width, height, image):

        points = np.hstack([points, np.ones((len(points), 1))])
        indexes = np.arange(0, len(points))

        points2d = projection_mat @ points.T

        mask = points2d[2] > 1e-6
        points2d = points2d[:, mask] / points2d[2][mask]
        indexes = indexes[mask]

        valid_in_image_mask = (
                (points2d[0] >= 0) & (points2d[0] <= width) &  # x-coordinates
                (points2d[1] >= 0) & (points2d[1] <= height)  # y-coordinates
        )
        points2d = points2d[:, valid_in_image_mask]
        indexes = indexes[valid_in_image_mask]

        points2d[1] = (self.height - 1) - points2d[1]

        color_x = np.round(np.clip(points2d[0], 0, self.width - 1)).astype(np.int32)
        color_y = (self.height - 1) - np.round(np.clip(points2d[1], 0, self.height - 1)).astype(np.int32)

        colors = image[color_y, color_x]

        return indexes, colors, points2d

    @staticmethod
    def aligningRotation():
        rot = LidarDataReader.rotation(-np.pi / 2, 0, np.pi / 2)
        return rot

    def readTimestamps(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]

        date_format = "%Y-%m-%d %H:%M:%S.%f"  # Format string to match the input format
        timestamps = []

        count = 0
        for i in range(len(lines)):
            count += 1
            time = lines[i]
            time = time.split('.')
            time = time[0] + "." + time[1][:6]
            timestamps.append( datetime.strptime(time, date_format) )

            if count >= self.MAX_DATA_READ:
                break

        return timestamps

    def eatBytes(self, filename):
        with open(self.root + self.VELODYNE_DATA + filename, "rb") as f:
            bin_data = f.read()
            float_data = array.array('f', bin_data)
            numpy_array = np.frombuffer(float_data, dtype=np.float32)
            reshaped_array = numpy_array.reshape(-1, 4)
        return reshaped_array[:, :3].astype(np.float64)

    def readImage(self, filename):
        with open(self.root + self.IMAGES + filename, "rb") as f:
            image = Image.open(f)  # Load image
            self.width = image.width
            self.height = image.height

            return np.array(image)  # Convert to NumPy array

    def getCurrentName(self):
        return self.getfilenames()[self.currentName]

    def getNextWait(self):
        currentTimestamp = self.datamap[self.getCurrentName()][1]
        nextFrame = self.peekNextName()
        if nextFrame == UNDEFINED:
            return 0

        nextTimestamp = self.datamap[nextFrame][1]

        delay = nextTimestamp - currentTimestamp
        ms = delay.total_seconds() * 1000
        return ms

    def getPoints(self, filename):
        return np.ascontiguousarray(self.datamap[filename][0]), np.ascontiguousarray(self.datamap[filename][2]) # the actual points,

    def getTimeStamp(self, filename):
        return self.datamap[filename][1] # the timestamp

    def getDepth(self, filename):
        return np.ascontiguousarray(self.datamap[filename][2]) # the timestamp

    def calcDepth(self, np_points):
        distances = np.linalg.norm(np_points, axis=1).astype(np.float64)

        dist_color = distances / np.max(distances)

        color_buffer = np.zeros((3, len(np_points)), np.float64)
        color = np.sqrt(dist_color)

        color_buffer[0] = color
        color_buffer[1] = color
        color_buffer[2] = color

        return np.ascontiguousarray(color_buffer.T)
        # had problems with passing this array to point_cloud.colors; might happen because of transpose

    def getfilenames(self):
        return list(self.datamap.keys())

    def peekNextName(self):
        if self.currentName < len(self.datamap.keys()) - 1:
            return self.getfilenames()[self.currentName + 1]
        else:
            return UNDEFINED

    def getNextName(self):
        names = self.getfilenames()

        ret = names[self.currentName]
        self.currentName += 1
        if self.currentName >= len(names):
            self.currentName = 0
            return False
        return ret

    def resetNameIterator(self):
        self.currentName = 0

    def getOxtsReader(self):
        return self.oxtsDataReader

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

            # Extract the rotation matrix (R)
            R_values = list(map(float, lines[1].split(":")[1].strip().split())) # R:
            R = np.array(R_values).reshape(3, 3)  # Convert to 3x3 matrix

            # Extract the translation vector (T)
            T_values = list(map(float, lines[2].split(":")[1].strip().split())) # T:
            T = np.array(T_values).reshape(3, 1)  # Convert to 3x1 column vector

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
        proj = np.array(proj_values).reshape(3, 4)  # Convert to 3x1 column vector

        rect_00_values = list(map(float, line_rect.strip().split()))
        rect_00 = np.array(rect_00_values).reshape(3, 3)

        """c2c_t_values = list(map(float, line_c2c_t.strip().split()))
        c2c_t = np.array(c2c_t_values).reshape(3, 1)  # Convert to 3x1 column vector
        """

        c2c_rot_values = list(map(float, line_c2c_r.strip().split()))
        c2c_rot = np.array(c2c_rot_values).reshape(3, 3)

        return proj, rect_00, c2c_rot

