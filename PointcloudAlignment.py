import random

import numpy as np


class PointcloudAlignment:

    datamap = {} # hold position and rotation data

    MODE_PREDICTIVE = 0
    MODE_REACTIVE = 1
    MODE_LERP = 2

    VELOCITY_MODE_ACCELERATION = 0
    VELOCITY_MODE_RAW_VELOCITY = 1

    def __init__(self):

        self.inited = False
        self.lidarData = None
        self.oxtsData = None



    def init(self, lidarData, oxtsData):
        self.inited = False

        self.lidarData = lidarData
        self.oxtsData = oxtsData
        VELOCITY_MODE = self.VELOCITY_MODE_RAW_VELOCITY

        keyframes = self.lidarData.getfilenames()

        origin_oxts = self.oxtsData.getOriginOx()
        prev_oxts = None
        curr_oxts = None
        next_oxts = None

        pos = np.array([0, 0, 0])
        rot = None

        if VELOCITY_MODE == self.VELOCITY_MODE_RAW_VELOCITY:
            vel = np.array([0, 0, 0])
        else:
            vel = -origin_oxts.getVelocity()

        for i in range(len(keyframes)):
            if i == 0:
                prev_oxts = None
            else:
                prev_oxts = self.oxtsData.getOx(keyframes[i - 1])

            if i == len(keyframes) - 1:
                next_oxts = None
            else:
                next_oxts = self.oxtsData.getOx(keyframes[i + 1])

            curr_oxts = self.oxtsData.getOx(keyframes[i])

            if VELOCITY_MODE == self.VELOCITY_MODE_ACCELERATION:
                pos, vel, rot = self.getNextPose_Acceleration(
                    prev_pos=pos,
                    prev_vel=vel,
                    origin_oxts=origin_oxts,
                    prev_oxts=prev_oxts,
                    current_oxts=curr_oxts,
                    next_oxts=next_oxts,
                    mode=self.MODE_PREDICTIVE)
            else:
                pos, rot = self.getNextPose_Velocity(
                    prev_pos=pos,
                    origin_oxts=origin_oxts,
                    prev_oxts=prev_oxts,
                    current_oxts=curr_oxts,
                    next_oxts=next_oxts,
                    mode=self.MODE_REACTIVE)

            self.datamap[keyframes[i]] = pos, rot

        self.inited = True

    def cleanup(self):
        self.inited = False
        for k in self.datamap:
            del self.datamap[k]

    def getTranslations(self):
        positions = []
        for pos, rot in self.datamap.values():
            positions.append(pos)

        return np.array(positions)

    def getNextPose_Acceleration(self, prev_pos, prev_vel, origin_oxts, prev_oxts, current_oxts, next_oxts, mode):
        baseRotation = np.linalg.inv(origin_oxts.getTrueRotation(np.eye(3, 3)))


        prev_time = None if prev_oxts is None else prev_oxts.getTime()
        current_time = current_oxts.getTime()
        next_time = None if next_oxts is None else next_oxts.getTime()

        if prev_time is not None:
            prev_delta = (current_time - prev_time).total_seconds()
        else:
            prev_delta = (next_time - current_time).total_seconds()

        if next_time is not None:
            next_delta = (next_time - current_time).total_seconds()
        else:
            next_delta = (current_time - prev_time).total_seconds()

        velocity = None
        delta = None

        vehicleRelativeAcc = False


        if mode == self.MODE_PREDICTIVE:
            delta = next_delta
            acceleration = -current_oxts.getAcceleration()
            rotation = current_oxts.getTrueRotation(baseRotation)

            acceleration = (rotation @ acceleration.T).T
            velocity = prev_vel + acceleration * delta

        if mode == self.MODE_REACTIVE:
            delta = prev_delta
            acceleration = -current_oxts.getAcceleration()
            rotation = current_oxts.getTrueRotation(baseRotation)

            acceleration = (rotation @ acceleration.T).T

            velocity = prev_vel + acceleration * delta

        return prev_pos + velocity * delta, velocity, current_oxts.getYawRotation(baseRotation)


    def getNextPose_Velocity(self, prev_pos, origin_oxts, prev_oxts, current_oxts, next_oxts, mode):
        baseRotation = np.linalg.inv(origin_oxts.getYawRotation(np.eye(3, 3)))


        prev_time = None if prev_oxts is None else prev_oxts.getTime()
        current_time = current_oxts.getTime()
        next_time = None if next_oxts is None else next_oxts.getTime()

        if prev_time is not None:
            prev_delta = (current_time - prev_time).total_seconds()
        else:
            prev_delta = (next_time - current_time).total_seconds()

        if next_time is not None:
            next_delta = (next_time - current_time).total_seconds()
        else:
            next_delta = (current_time - prev_time).total_seconds()

        velocity = np.array([0,0,0])
        rotation = np.eye(3, 3)
        delta = 0

        if mode == self.MODE_PREDICTIVE:
            velocity = -current_oxts.getVelocity()
            rotation = current_oxts.getYawRotation(baseRotation)

            velocity = (rotation @ velocity.T).T
            delta = next_delta

        if mode == self.MODE_REACTIVE:
            velocity = -current_oxts.getVelocity()
            rotation = current_oxts.getYawRotation(baseRotation)

            velocity = (rotation @ velocity.T).T
            delta = prev_delta

        return prev_pos + velocity * delta, current_oxts.getYawRotation(baseRotation)





    def getTimeStampsAndRotations(self):

        originOx = self.oxtsData.getOriginOx()
        baseRotation = np.linalg.inv(originOx.getRotationMatrix(np.eye(3, 3)))

        for k in self.oxtsData.datamap.keys():
            ox = self.oxtsData.datamap[k]

            self.timestamps.append(ox.getTime())
            self.rotations.append(ox.getRotationMatrix(baseRotation))

    def getElapsedSec(self, before, after):
        delta = after - before
        return delta.total_seconds()



    def getAlignedPathPoints(self):
        points = []
        for k in self.datamap.keys():
            #points.append(self.datamap[k][0]) # pos (not rot)
            points.append(self.datamap[k][0]) # pos (not rot)

        return np.array(points)

    def align(self, key, points):
        return self.transform(self.rotate(points, key), key)
        #return self.transform(points, key)

    def transform(self, points, key):
        # return points + self.datamap[key][0] # pos, not rot
        return points + self.datamap[key][0] # pos, not rot

    def getPose(self, filename):
        return self.datamap[filename][0], self.datamap[filename][1] # (pos, rot)

    def applyPose(self, np_points, pose):
        t, R = pose
        return (R @ np_points.T).T + t

    def rotate(self, points, key):
        rot = self.datamap[key][1]
        rotation = rot

        points = rotation @ points.T
        points = points.T

        return points

    @staticmethod
    def rotation(x, y=None, z=None):


        if isinstance(x, np.ndarray):
            xr = x[0]
            yr = x[1]
            zr = x[2]
        else:
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

    @staticmethod
    def randomRotation(scale):
        x = random.uniform(-1, 1) * np.pi * 2 * scale
        #x = 0
        y = random.uniform(-1, 1) * np.pi * 2 * scale
        #z = 0
        z = random.uniform(-1, 1) * np.pi * 2 * scale
        return PointcloudAlignment.rotation(x,y,z)

    @staticmethod
    def randomTranslation1():
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        return np.array([x, y, z])