import math
import time
import unittest
from random import randint

import numpy as np

from ComputeShader import ComputeShader
from LidarDataReader import LidarDataReader
from OxtsDataReader import OxtsDataReader
from PointcloudAlignment import PointcloudAlignment
from PointcloudIcpContainer import PointcloudIcpContainer
from Renderer import Renderer



class FunctionalTesting(unittest.TestCase):

    def setUp(self):
        path = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"
        calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        self.oxtsDataReader = OxtsDataReader(path)
        self.lidarDataReader = LidarDataReader(path=path, oxtsDataReader=self.oxtsDataReader, calibration=calibration,
                                          targetCamera="02")

        self.pointcloudAlignment = PointcloudAlignment(self.lidarDataReader, self.oxtsDataReader)

        VOXEL_SIZE = 0.01
        self.renderer = Renderer(VOXEL_SIZE)
        self.renderingThread = self.renderer.getRenderingThread()

        self.computeShader = ComputeShader()
        self.icpContainer = PointcloudIcpContainer(self.computeShader, self.pointcloudAlignment)

        self.tolerance = 1e-5 # 0.00001

        self.red = np.array([255, 0, 0])
        self.green = np.array([0, 255, 0])
        self.blue = np.array([0, 0, 255])
        self.white = np.array([255, 255, 255])
        self.grey = np.array([255, 255, 255]) // 2

    def tearDown(self):
        self.computeShader.cleanup()
        self.renderingThread.join()


    def getLidarPoints(self):
        filenames = self.lidarDataReader.getfilenames()
        pts, cols = self.lidarDataReader.getPoints(filenames[0])
        return pts

    def test_canDisplayPoints(self):
        pts = self.getLidarPoints()

        self.renderer.addPoints(pts)

    def test_getNormalsTest(self):
        lines = []
        line_colors = []



        def addScanLineRender(sl, points):
            s, e = sl
            lps = points[s: e +1]
            lps = self.icpContainer.connectPointsInOrder(lps)
            color = randcolor()


            for p in lps:
                lines.append(p)
                line_colors.append(color)

        def addLine(p1, p2, color = None):
            lines.extend([p1, p2])
            if color is None:
                color = randcolor()


            line_colors.extend([color, color])

        def cos(index, refvec, origin, points):
            lidar_tall = 0.254
            lidarOffset = np.array([0, lidar_tall / 2.0, 0])

            circvec = points[index] - origin - lidarOffset
            # v[1] = 0
            v = circvec / np.linalg.norm(circvec)
            return np.dot(v, refvec)

        def indmod(index, clen, cbegin):
            return (index + clen) % clen

        def dir(index, circlen, begin, refVec, origin, points):
            p1 = begin + indmod(index - 1, circlen, begin)
            p2 = begin + indmod(index + 1, circlen, begin)
            """renderer.setLines([points[p1], origin, points[p2], origin],
                              [np.array([255,0,0]), np.array([255,0,0]), np.array([0,255,0]),np.array([0,255,0])])
            """

            cos1 = cos(p1, refVec, origin, points) + 1
            cos2 = cos(p2, refVec, origin, points) + 1

            if cos1 > cos2:
                return -1.0
            else:
                return 1.0

        def randcolor():
            return np.array([randint(0, 255), randint(0, 255), randint(0, 255)])

        def getcolor(r, g, b):
            return np.array([r, g, b])

        def distsq(p1, p2):
            diff = p1 - p2
            return np.dot(diff, diff)

        def findClosestPoint(scan_lines, origin, points, refp):

            mind = 1.23e20
            minind = 0
            minscan = 0
            for i in range(len(scan_lines)):
                closest_on_line = binarySearchAndCheck(scan_lines, i, refp, points, origin)
                d = distsq(points[closest_on_line], refp)
                if d < mind:
                    mind = d
                    minind = closest_on_line
                    minscan = i

            return minind, minscan, mind



        def binaryScanCircleCheck(begin, end, refp, origin, points):
            refvec = self.icpContainer.sphereProjectPont(refp, origin)
            circlen = end -begin + 1
            index = (begin + end) // 2
            half_c = circlen // 2

            runs = int(math.ceil(math.log2(circlen)))

            for i in range(runs):
                d = dir(index, circlen, begin, refvec, origin, points)
                index = indmod(index + int(half_c * d), circlen, begin)
                half_c = int(math.ceil(half_c / 2.0))

            return begin + index

        def binarySearchAndCheck(scan_lines, scan_index, refp, points, origin, next_closest=None):

            begin, end = scan_lines[scan_index]
            closest = binaryScanCircleCheck(begin, end, refp, origin, points)

            check = 5
            circlen = end - begin + 1
            index = closest - begin

            mindist = 1.23e30
            for i in range(-check, check):
                test = begin + indmod(index + i, circlen, begin)

                d = distsq(refp, points[test])

                if d < mindist and test != next_closest:
                    mindist = d
                    closest = test

            return closest

        def getScanlineOfPoint(scanlines, pointindex):
            for i in range(len(scanlines)):
                begin, end = scanlines[i]
                if begin <= pointindex and pointindex <= end:
                    return i

        def getClosestAboveOrBelow(scan_lines, minscan, refp, points, origin):
            if minscan == len(scan_lines) - 1:
                return binarySearchAndCheck(scan_lines, minscan - 1, refp, points, origin, -1)

            if minscan == 0:
                return binarySearchAndCheck(scan_lines, minscan + 1, refp, points, origin, -1)

            minabove = binarySearchAndCheck(scan_lines, minscan - 1, refp, points, origin)
            minbelow = binarySearchAndCheck(scan_lines, minscan + 1, refp, points, origin)

            minabove_point = points[minabove]
            minbelow_point = points[minbelow]

            # addLine(refp, minabove_point + np.array([0, 0.01, 0]), getcolor(100,100,100))
            # addLine(refp, minbelow_point + np.array([0, 0.01, 0]), getcolor(100,100,100))

            if distsq(refp, minbelow_point) < distsq(refp, minabove_point):
                return minbelow
            else:
                return minabove

        def getNormal(scan_lines, point_index, points, origin):
            refp = points[point_index]

            minscan = getScanlineOfPoint(scan_lines, point_index)

            closest_online = binarySearchAndCheck(scan_lines, minscan, refp, points, origin, point_index)
            closest_next = getClosestAboveOrBelow(scan_lines, minscan, refp, points, origin)

            v1 = points[closest_online] - refp
            v2 = points[closest_next] - refp
            # addLine(refp, v1 + refp, getcolor(0,255,0))
            # addLine(refp, v2 + refp, getcolor(0,0,255))

            cross = np.cross(v1, v2)
            return cross / np.linalg.norm(cross)

        origin = np.array([0, 0, 0])
        pts = self.getLidarPoints() + origin

        self.renderer.addPoints(pts)
        scan_lines = self.icpContainer.getScanLines(pts, origin)

        for i in range(0, len(pts), 1):
            # randind = randint(1, len(pts))
            randind = i
            refp = pts[randind]

            # addLine(refp, origin, getcolor(255,0,0))

            normal = getNormal(scan_lines, randind, pts, origin)

            addLine(refp, refp + normal * 0.1, getcolor(255, 255, 255))
            # addLine(refp, origin)

            if i % 1000 == 0:
                self.renderer.setLines(lines, line_colors)

        self.assertTrue(True)

    def test_pointToPointLSShouldConverge(self):
        origin = np.array([0, 0, 0])




        pts1 = self.getLidarPoints()

        pts2 = self.getLidarPoints()

        pts1 += origin
        pts2 += origin

        randrot = PointcloudAlignment.randomRotation(0.01)
        randtrans = PointcloudAlignment.randomTranslation1() * 0.01

        pts2 = (randrot @ pts2.T).T + randtrans

        self.renderer.addPoints(pts1, self.green)
        pp2 = self.renderer.addPoints(pts2, self.red)

        reference_grid = self.icpContainer.getUniformGrid(10)
        rgp = self.renderer.addPoints(reference_grid, self.blue)

        time.sleep(3)

        self.icpContainer.initLS(pts1, pts2, origin)

        init_err = 1e20

        for i in range(30):
            t, R = self.icpContainer.iteration_of_ls(pts2)

            R = PointcloudAlignment.rotation(R[0], R[1], R[2])
            pts2 = self.icpContainer.apply_iteration_of_ls(pts2, t, R)
            reference_grid = self.icpContainer.applyIcpStep(reference_grid, R, t)

            self.renderer.freePoints(rgp)
            rgp = self.renderer.addPoints(reference_grid, self.blue)

            self.renderer.freePoints(pp2)
            pp2 = self.renderer.addPoints(pts2, self.white)
            time.sleep(0)
            print(i)



        R_opt, t_opt = self.icpContainer.uniform_optimal_icp(reference_grid, self.icpContainer.getUniformGrid(10))

        time.sleep(5)

        self.renderer.freePoints(pp2)  # free icp visualized points
        time.sleep(1)

        pts2 = self.getLidarPoints()
        pts2 += origin

        pts2 = (randrot @ pts2.T).T + randtrans

        pp2 = self.renderer.addPoints(pts2, self.blue)  # add original lidar transformed points
        time.sleep(1)
        self.renderer.freePoints(pp2)

        pts2 = (R_opt @ pts2.T).T - t_opt  # transform points with optimal icp
        time.sleep(1)

        self.renderer.addPoints(pts2, self.red)  # display transformed points