import numpy as np
import cupy as cp



class Camera:
    K = np.identity(3)
    R = np.identity(3)
    t = np.array([0,0,0])
    resx = 0
    resy = 0

    pos = None

    v2cr = np.array([0.0])
    v2ct = np.array([0.0])

    Kpath = "//calib_cam_to_cam.txt"
    V2Cpath = "//calib_velo_to_cam.txt"

    def __init__(self, pathToClaib):
        self.pos = np.array([0, 0, 0])

        with open(pathToClaib + self.Kpath, 'r') as file:
            lines = file.readlines()

            for line in lines:
                data = line.split(' ')
                if data[0] == "K_00:":
                    self.K = np.array(data[1:])
                    self.K = self.K.reshape(3, 3).astype(np.float64)
                    self.K[0, 2] = 0
                    self.K[1, 2] = 0
                    # self.K[2, 2] = 0.5
                    # print(self.K)

                if data[0] == "R_02:":
                    # self.R = np.array(data[1:])
                    # self.R = self.R.reshape(3, 3).astype(np.float64)
                    self.R = np.identity(3)

                    print(self.R)

                if data[0] == "S_00:":
                    num = np.array(data[1:]).astype(np.float64)
                    #self.resx = int(num[0])
                    #self.resy = int(num[1])
                    self.resx = 1600
                    self.resy = 800


                if data[0] == "T_00:":
                    self.t = np.array(data[1:]).astype(np.float64)

        with open(pathToClaib + self.V2Cpath, 'r') as file:
            lines = file.readlines()

            for line in lines:
                data = line.split(' ')
                if data[0] == "R:":
                    self.v2cr = np.array(data[1:])
                    self.v2cr = self.v2cr.reshape(3, 3).astype(np.float64)

                    # print(self.v2cr)

                if data[0] == "T:":
                    self.v2ct = np.array(data[1:]).astype(np.float64)
                    # print(self.v2ct)


    def setPosition(self, xyz):
        self.pos = xyz
        # print(self.pos)

    def setRotation(self, x, y, z):
        self.R = self.rotation(x, y, z)

    def setResolution(self, w, h):
        self.resx = w
        self.resy = h

    def getForward(self):
        forward_vector_camera = np.array([0, 0, 1])


        forward = np.linalg.inv(self.R) @ forward_vector_camera
        return forward


    def getLeft(self):
        return np.linalg.cross(self.getForward(), np.linalg.inv(self.R) @ np.array([0, 1, 0]))

    def getUp(self):
        return -np.array([0, 0, 1])



    def showForward(self, points):

        forward = self.getForward()


        focus = forward - self.pos

        side_length = 0.05  # Side length of the cube
        # Calculate half the side length
        half_side = side_length / 2
        num_points_per_axis = 5  # Adjust this value for more or fewer points

        # Generate linearly spaced points for each axis
        x_coords = np.linspace(focus[0] - half_side, focus[0] + half_side, num_points_per_axis)
        y_coords = np.linspace(focus[1] - half_side, focus[1] + half_side, num_points_per_axis)
        z_coords = np.linspace(focus[2] - half_side, focus[2] + half_side, num_points_per_axis)

        # Create a meshgrid for the points in 3D
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords)

        # Combine into a single array of points (N x 3)
        cpoints = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        cpoints = cpoints[np.linalg.norm(cpoints - focus, axis=1) < side_length / 2]

        points = np.concatenate((points, cpoints), axis=0)

        return points



    def project_np(self, points):


        # print(self.R)
        tpoints = self.R @ (points + self.pos).T
        #print(self.R)

        points2d = self.K @ tpoints

        mask = points2d[2] > 0
        points2d = points2d[:, mask] / points2d[2][mask]

        valid_in_image_mask = (
                (points2d[0] >= -self.resx / 2) & (points2d[0] < self.resx / 2) &  # x-coordinates
                (points2d[1] >= -self.resy / 2) & (points2d[1] < self.resy / 2)  # y-coordinates
        )
        points2d = points2d[:, valid_in_image_mask]

        return mask, valid_in_image_mask, points2d


    def projectAsNumpy(self, np_points, color):


        points = np_points

        #points = self.showForward(points)

        image_buffer = np.zeros((self.resy, self.resx, 3), dtype=np.uint8)

        #depth = self.calcDepth(points)

        mask, image_mask, points2d = self.project_np(points)

        #points2d[2] = depth[mask][image_mask]


        if len(points2d[0]) == 0: # ON FIRST RUN NO POINTS PASS THE FILTER, WHYY?:D
            return image_buffer

        points2d[0] += self.resx / 2
        points2d[1] += self.resy / 2

        #image_buffer[points2d[1].astype(np.int32), points2d[0].astype(np.int32)] \
        #    = points2d[2][:, np.newaxis] * np.array([100, 255, 255])

        # without depth for debug
        image_buffer[points2d[1].astype(np.int32), points2d[0].astype(np.int32)] \
            = color


        return image_buffer

    def calcDepth(self, np_points):
        distances = np.linalg.norm(self.pos + np_points, axis=1).astype(np.float64)
        depth = 1 - distances / np.max(distances)
        return depth

    def calcDepthCp(self, cp_points):
        pos = cp.array(self.pos)
        distances = cp.linalg.norm(pos + cp_points, axis=1).astype(cp.float64)
        depth = 1 - distances / cp.max(distances)
        return depth



    def rotation(self, x, y, z):


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

    def project(self, lidarDataReader, rotation):

        points = lidarDataReader.getPoints(lidarDataReader.getCurrentName())

        # self.t = np.array([0, 0, 0])
        # Combine R and t into an extrinsic matrix (3x4)
        v2cthom = np.array([0, 0.2, 0]) - self.v2ct / 2
        v2cthom = v2cthom.reshape(3, 1)


        Rt = np.hstack((self.v2cr, v2cthom ))
        # Rt = np.hstack((self.rotation(x = rotation[0] * np.pi, y = rotation[1] * np.pi, z = rotation[2] * np.pi), self.v2ct.reshape(3, 1)))

        points_3d_hom = np.hstack((points, np.ones((points.shape[0], 1))))

        points_2d_hom = self.K @ Rt @ points_3d_hom.T  # This results in 3xN homogeneous 2D points

        w = points_2d_hom[2]
        mask = w > 0

        points_2d_hom = points_2d_hom[:, mask] / w[mask]

        valid_in_image_mask = (
                (points_2d_hom[0] >= -self.resx / 2) & (points_2d_hom[0] < self.resx / 2) &  # x-coordinates
                (points_2d_hom[1] >= -self.resy / 2) & (points_2d_hom[1] < self.resy / 2)  # y-coordinates
        )
        points_2d_hom = points_2d_hom[:, valid_in_image_mask]



        points_2d_hom = points_2d_hom.T

        depth = lidarDataReader.getDepth(lidarDataReader.getCurrentName())

        depth = depth[mask][valid_in_image_mask]

        return points_2d_hom, depth

