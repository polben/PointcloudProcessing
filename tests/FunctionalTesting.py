import math
import time
import unittest

import numpy as np

from ComputeShader import ComputeShader
from EnvironmentConstructor import EnvironmentConstructor
from LidarDataReader import LidarDataReader
from OutlierFilter import OutlierFilter
from OxtsDataReader import OxtsDataReader
from PointcloudAlignment import PointcloudAlignment
from PointcloudIcpContainer import PointcloudIcpContainer
from Renderer import Renderer
from Voxelizer import Voxelizer


class FunctionalTesting(unittest.TestCase):

    def setUp(self):
        self.path = "F:/uni/3d-pointcloud/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync"
        self.calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        self.oxtsDataReader = OxtsDataReader()
        self.lidarDataReader = LidarDataReader()

        self.pointcloudAlignment = PointcloudAlignment()

        self.VOXEL_SIZE = 1
        self.renderer = Renderer(self.VOXEL_SIZE)
        self.renderingThread = self.renderer.getRenderingThread()

        self.computeShader = ComputeShader()
        self.icpContainer = PointcloudIcpContainer(self.computeShader, self.pointcloudAlignment)

        self.tolerance = 1e-5 # 0.00001

        self.environment = EnvironmentConstructor(self.renderer, self.oxtsDataReader, self.lidarDataReader, self.icpContainer, self.computeShader)

        self.red = np.array([255, 0, 0])
        self.green = np.array([0, 255, 0])
        self.blue = np.array([0, 0, 255])
        self.white = np.array([255, 255, 255])
        self.grey = np.array([255, 255, 255]) // 2

    def tearDown(self):
        self.computeShader.cleanup()
        self.renderingThread.join()


    def getLidarPoints(self, index=0):
        filenames = self.lidarDataReader.getfilenames()
        pts, cols = self.lidarDataReader.getPoints(filenames[index])
        return pts

    def randcolor(self):
        return np.random.rand(3, )

    def test_canDisplayPoints(self):
        self.lidarDataReader.init(self.path, self.calibration, "02", 10, None)
        pts1 = self.getLidarPoints()
        pts2 = self.getLidarPoints() + np.array([1, 1, 1])

        time.sleep(1)

        pp1 = self.renderer.addPoints(pts1, np.array([255,0,0]))


        time.sleep(1)

        pp2 = self.renderer.addPoints(pts2, np.array([0,255,0]))



        time.sleep(1)

        self.renderer.freePoints(pp1)

        time.sleep(1)

        self.renderer.freePoints(pp2)

        time.sleep(1)

        self.renderer.addPoints(pts1)





    @staticmethod
    def getNormals(scan_lines, points, origin, icpContainer):
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

        def distsq(p1, p2):
            diff = p1 - p2
            return np.dot(diff, diff)

        def binaryScanCircleCheck(begin, end, refp, origin, points):
            refvec = icpContainer.sphereProjectPont(refp, origin)
            circlen = end - begin + 1
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

        def getClosestAboveOrBelow(scan_lines, minscan, refp, points, origin, point_index):
            if minscan == len(scan_lines) - 1:
                return binarySearchAndCheck(scan_lines, minscan - 1, refp, points, origin, point_index)

            if minscan == 0:
                return binarySearchAndCheck(scan_lines, minscan + 1, refp, points, origin, point_index)

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
            closest_next = getClosestAboveOrBelow(scan_lines, minscan, refp, points, origin, point_index)

            v1 = points[closest_online] - refp
            v2 = points[closest_next] - refp
            # addLine(refp, v1 + refp, getcolor(0,255,0))
            # addLine(refp, v2 + refp, getcolor(0,0,255))

            cross = np.cross(v1, v2)
            return cross / np.linalg.norm(cross)

        def normalTowardsOrigin(origin, normal, pointind, points):
            to_origin = origin - points[pointind]
            to_origin_n = to_origin / np.linalg.norm(to_origin)

            if np.dot(normal, to_origin_n) > np.dot(-normal, to_origin_n):
                return normal
            else:
                return -normal


        normals = []
        for i in range(0, len(points), 1):
            normal = getNormal(scan_lines, i, points, origin)
            normal = normalTowardsOrigin(origin, normal, i, points)
            normals.append(normal)

            if i % 10000 == 0:
                print(i)

        return normals

    def test_shouldGetScanLines(self):
        pts_l = self.getLidarPoints()
        pts = self.icpContainer.sphereProjectPoints(pts_l, np.array([0, 0, 0]))

        sphere = self.icpContainer.sphereProjectPoints(pts_l, np.array([0, 0, 0])) * 1.01
        self.renderer.addPoints(pts_l, np.array([0.5, 0.5, 0.5]))

        stt = time.time()
        scan_lines = self.icpContainer.getScanLines(pts_l, np.array([0, 0, 0]))

        print("scans: " + str(time.time() - stt))

        for start, end in scan_lines:
            scan_points = pts_l[start:end]
            self.renderer.setLines(self.icpContainer.connectPointsInOrder(scan_points), None)
            time.sleep(0.1)

    def test_uniformGridICP_LS(self):
        grid1 = self.icpContainer.getUniformShape()
        R = PointcloudAlignment.randomRotation(0.4)
        t = PointcloudAlignment.randomTranslation1() * 10
        grid2 = (R @ self.icpContainer.getUniformShape().T).T + t

        red = np.array([1, 0, 0])
        green = np.array([0, 1, 0])
        p1 = self.renderer.addPoints(grid1, green)
        p2t = self.renderer.addPoints(grid2, red)

        time.sleep(3)

        for i in range(10):
            t, R = self.icpContainer.icp_step_LS_vector(grid1, grid2)
            # mean = np.mean(grid2, axis=0)
            # grid2 = grid2 - mean
            grid2 = (PointcloudAlignment.rotation(R[0], R[1], R[2]) @ grid2.T).T
            grid2 -= t
            # grid2 += mean

            self.renderer.freePoints(p2t)
            p2t = self.renderer.addPoints(grid2)


            time.sleep(1)

    def test_getNormalsTest(self):
        lines = []






        def addLine(p1, p2, color = None):
            lines.extend([p1, p2])



        origin = np.array([0, 0, 0])
        pts = self.getLidarPoints() + origin

        self.renderer.addPoints(pts)
        scan_lines = self.icpContainer.getScanLines(pts, origin)


        normals = FunctionalTesting.getNormals(scan_lines, pts, origin, self.icpContainer)

        for i in range(0, len(pts), 1):
            refp = pts[i]
            normal = normals[i]
            addLine(refp, refp + normal * 0.1)

            if i % 1000 == 0:
                self.renderer.setLines(lines)

        self.assertTrue(True)

    def test_gpuNormals(self):

        lines = []
        colors = []


        def addLine(p1, p2):
            lines.extend([p1, p2])
            colors.extend([np.array([255,0,255]), np.array([255,0,255])])


        def getcolor(r, g, b):
            return np.array([r, g, b])

        origin = np.array([0, 0, 0])
        pts = self.getLidarPoints() + origin

        self.renderer.addPoints(pts)
        scan_lines = self.icpContainer.getScanLines(pts, origin)

        self.computeShader.prepareDispatchNormals(pts, scan_lines, origin)
        normals = self.computeShader.normals_out_a

        for i in range(0, len(pts), 1):

            refp = pts[i]
            normal = normals[i][:3]

            addLine(refp, refp + normal * 0.1)

            if i % 1000 == 0:
                self.renderer.setLines(lines, colors)


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

        self.icpContainer.preparePointToPoint(pts1, origin, pts2)

        init_err = 1e20

        for i in range(30):
            st = time.time()
            t, R = self.icpContainer.dispatchPointToPoint(pts2)
            print("that closest point took: " + str(time.time() - st))

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


    def test_findClosestPointGpu(self):
        origin = np.array([0, 0, 0])

        frame1 = 0
        frame2 = 1

        filenames = self.lidarDataReader.getfilenames()

        pts1, cols = self.lidarDataReader.getPoints(filenames[frame1])
        pts1 += origin
        # pts1 = self.pointcloudAlignment.align(filenames[frame1], pts1)

        pts2, cols = self.lidarDataReader.getPoints(filenames[frame2])
        # pts2 = self.pointcloudAlignment.align(filenames[frame2], pts2) + np.array([0.01, 0, 0])

        self.renderer.addPoints(pts1, np.array([255, 0, 0]))

        self.renderer.addPoints(pts2, np.array([0, 255, 0]))

        scan_lines = self.icpContainer.getScanLines(pts1, origin)
        self.computeShader.prepareNNS(pts1, scan_lines, pts2, origin)

        st = time.time()
        corrs, dists = self.computeShader.dispatchNNS(pts2)
        print("that closest point took: " + str(time.time() - st))

        lines = []
        for i in range(len(corrs)):
            c = corrs[i]
            lines.extend([pts2[i], pts1[c]])
        self.renderer.setLines(lines)


    def test_pointToPlaneLSShouldConverge(self):
        origin = np.array([0, 0, 0])

        key1 = 36
        key2 = 38


        pts1 = self.getLidarPoints(key1)

        pts2 = self.getLidarPoints(key2)

        pts1 += origin
        pts2 += origin

        randrot = PointcloudAlignment.randomRotation(0.0001)
        randtrans = PointcloudAlignment.randomTranslation1() * 0.001

        pts2 = (randrot @ pts2.T).T + randtrans

        self.renderer.addPoints(pts1, self.green)
        pp2 = self.renderer.addPoints(pts2, self.red)

        reference_grid = self.icpContainer.getUniformGrid(10)
        rgp = self.renderer.addPoints(reference_grid, self.blue)


        time.sleep(2)

        start = time.time()
        self.icpContainer.preparePointToPlane(pts1, origin, pts2)
        print("prep pointplane: " + str(time.time()-start))


        for i in range(100):
            start = time.time()
            t, R, delta_transf = self.icpContainer.dispatchPointToPlane(pts2)
            print("dispatch pointplane: " + str(time.time() - start))

            pts2 = self.icpContainer.apply_iteration_of_ls(pts2, t, R)
            reference_grid = self.icpContainer.applyIcpStep(reference_grid, R, t)

            self.renderer.freePoints(rgp)
            rgp = self.renderer.addPoints(reference_grid, self.blue)

            self.renderer.freePoints(pp2)
            pp2 = self.renderer.addPoints(pts2, self.white)
            # time.sleep(0.1)
            print(i)

        R_opt, t_opt = self.icpContainer.uniform_optimal_icp(reference_grid, self.icpContainer.getUniformGrid(10))


        refgrid = self.icpContainer.getUniformGrid(10)
        g2p = self.renderer.addPoints(refgrid, np.array([255,0,0]))
        time.sleep(1)
        self.renderer.freePoints(g2p)
        time.sleep(1)
        refgrid_opt = (R_opt @ refgrid.T).T + t_opt
        g2_op = self.renderer.addPoints(refgrid_opt, np.array([255,0,0]))

        pts2 = self.getLidarPoints(key2)
        pts2 = (randrot @ pts2.T).T + randtrans
        pts2 = (R_opt @ pts2.T).T - t_opt

        self.renderer.freePoints(pp2)
        self.renderer.addPoints(pts2, np.array([255,0,0]))

    def test_vectorizedSphereEqualManual(self):
        self.assertTrue(False)

    def test_binaryScanLineSearch(self):
        def sphereProject(point, origin):
            # icp sphere project point also takes into lidar height which seems off
            v = point - origin
            return v / np.linalg.norm(v)

        def binHeightSearch(points_a, scan_lines, refp, origin):
            uppen_scan = 0
            lower_scan = len(scan_lines) - 1

            target_height = self.icpContainer.sphereProjectPont(refp, origin)[1]

            prevmid = -1
            mid = 0
            steps = 0
            while prevmid != mid:
                time.sleep(0.01)

                prevmid = mid
                mid = (lower_scan + uppen_scan) // 2

                begin, end = scan_lines[mid]
                pt_a = points_a[begin]

                spherePoint = self.icpContainer.sphereProjectPont(pt_a, origin)
                height = spherePoint[1]

                self.renderer.setLines([origin, refp, origin, pt_a])
                # print("step: " + str(steps))
                if height < target_height:
                    lower_scan = mid
                else:
                    uppen_scan = mid

                steps += 1

            return mid


        pts = self.getLidarPoints(0)
        origin = np.array([0, 0, 0])
        self.renderer.addPoints(pts)
        # sphere = self.icpContainer.sphereProjectPoints(pts, np.array([0,0,0]))
        # self.renderer.addPoints(sphere)

        scan_lines = self.icpContainer.getScanLines(pts, origin)

        ok = 0
        errors = 0
        for i in range(len(scan_lines)):
            target_scan = i

            reference = pts[scan_lines[target_scan][0] + 500]

            found_scan = binHeightSearch(pts, scan_lines, reference, origin)

            if target_scan == found_scan:
                ok += 1
            else:
                errors += 1
                print(str(i) + " sl was error, found: " + str(found_scan))



    def test_findClosestPoint(self): # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        def binaryScanLineSearch(points_a, scanline, reference_point, origin):




            def cos(index, refvec, origin):
                v = self.icpContainer.sphereProjectPont(points_a[index], origin)
                return np.dot(v, refvec)

            def indmod(index, clen, cbegin):
                return (index + clen) % clen

            def dir(index, circlen, begin, refVec, origin):
                p1 = begin + indmod(index - 1, circlen, begin)
                p2 = begin + indmod(index + 1, circlen, begin)
                """renderer.setLines([points[p1], origin, points[p2], origin],
                                  [np.array([255,0,0]), np.array([255,0,0]), np.array([0,255,0]),np.array([0,255,0])])
                """

                cos1 = cos(p1, refVec, origin) + 1
                cos2 = cos(p2, refVec, origin) + 1

                if cos1 > cos2:
                    return -1.0, (2 - cos1) / 2.0
                else:
                    return 1.0, (2 - cos1) / 2.0


            refvec = reference_point - origin
            refvec /= np.linalg.norm(refvec)  # Normalize

            begin, end = scanline
            circlen = end - begin + 1

            index = (begin + end) // 2
            half = int(circlen * (1.0/2.0))

            runs = int(math.ceil(math.log2(circlen)))

            counter = 0
            for i in range(runs):
                counter += 1
                d, scale = dir(index=index, circlen=circlen, begin=begin, refVec=refvec, origin=origin)
                index = indmod(index + int(half * d), circlen, begin)
                lines.extend([origin, points_a[begin + index]])

                half = int(math.ceil(half *  1.0/2.0))
                self.renderer.setLines(lines)
                # time.sleep(0.3)
                lines.pop()
                lines.pop()

            # print(counter)
            return begin + index

        lines = []

        points_a = self.getLidarPoints(0)
        points_b = self.getLidarPoints(1)
        origin = np.array([0,0,0])

        refp = np.array([5, 5, 0]).astype(np.float32)

        lines.extend([origin, refp])




        scan_lines = self.icpContainer.getScanLines(points_a, origin)



        lines = []

        self.renderer.addPoints(points_a, None)
        # self.renderer.addPoints(points_b, None)



        for i in range(0, len(scan_lines), 1):
            scan_line = scan_lines[i]
            closest = binaryScanLineSearch(points_a, scan_line, refp, origin)
            lines.extend([refp, points_a[closest]])


    def test_showScanLines(self):
        path = "F://uni//3d-pointcloud//sample2"

        calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        oxtsDataReader = OxtsDataReader(path)
        lidarDataReader = LidarDataReader(path=path, oxtsDataReader=oxtsDataReader, calibration=calibration,
                                               targetCamera="02", max_read=50)

        names = lidarDataReader.getfilenames()
        pts, cols = lidarDataReader.getPoints(names[49])

        scan_lines = self.icpContainer.getScanLines(pts, np.array([0,0,0]))
        # print(len(scan_lines))
        self.renderer.addPoints(pts)
        lines = []
        colors = []
        for start, end in scan_lines:
            scan_points = pts[start:end]
            line = self.icpContainer.connectPointsInOrder(scan_points)
            lines.extend(line)
            colors.extend(self.renderer.handleColors(line, self.randcolor()))
            self.renderer.setLines(lines, colors)
            time.sleep(0.1)


    def test_projectLidarToSphere(self):
        filenames = self.lidarDataReader.getfilenames()
        pts1, cols1 = self.lidarDataReader.getPoints(filenames[0])
        pts2, cols2 = self.lidarDataReader.getPoints(filenames[1])
        origin = np.array([0, 0, 0])

        pc = self.icpContainer.sphereProjectPoints(pts1, origin)

        lines = self.icpContainer.connectPointsInOrder(pc)
        self.renderer.addPoints(pc, np.array([1, 1, 1]))
        # self.renderer.setLines(lines)


    def test_shouldGetApproximatePath(self):
        lidar, oxts = self.environment.getNextFrameData(0)
        self.environment.setupTransitions(oxts)

        self.renderer.addPoints(*lidar)


        lines = []
        colors = []


        def addLine(p1, p2, color):
            lines.extend([p1, p2])
            colors.extend([color, color])

        offset = 0
        for i in range(self.lidarDataReader.MAX_DATA_READ):
            lidardata, oxts = self.environment.getNextFrameData(offset)

            oxts_data = self.environment.getCurrentOxtsData(current_oxts=oxts)
            delta_velocity = self.environment.getDeltaVelocity(*oxts_data)
            current_velocity, current_rotation, current_time = self.environment.getCurrentOxtsData(current_oxts=oxts)

            addLine(self.environment.prev_position, self.environment.prev_position + delta_velocity, self.red)
            # addLine(self.environment.prev_position, self.environment.prev_position + current_velocity, self.green)

            self.environment.prev_position = self.environment.prev_position + delta_velocity

            self.environment.prev_time = current_time
            self.renderer.addPoints([self.environment.prev_position], self.red)

            self.renderer.setLines(lines, colors)
            time.sleep(0.1)


    def test_shouldAlignBasedOnImu(self): ### !!! this is the real deal
        ptr = None
        files = self.lidarDataReader.getfilenames()

        prevpos = np.array([0, 0, 0])
        currpos = np.array([0, 0, 0])
        prevtime = None
        currtime = self.oxtsDataReader.getOx(files[0]).getTime()

        for i in range(4):
            name = files[i]
            oxts = self.oxtsDataReader.getOx(name)
            rot = np.linalg.inv(oxts.getTrueRotation(np.eye(3)))
            yawrot = oxts.getYawRotation(np.eye(3))
            rot = yawrot @ rot

            if prevtime is not None:
                velocity = -oxts.getVelocity()
                currtime = oxts.getTime()
                deltatime = (currtime - prevtime).total_seconds()
                velocity = (rot @ velocity.T).T

                currpos = prevpos + velocity * deltatime


            pts = self.getLidarPoints(i)
            pts = (rot @ pts.T).T + currpos

            if ptr is not None:
                # self.renderer.freePoints(ptr)
                a=0
            _, cls, ints = self.lidarDataReader.getPointsWithIntensities(i)
            ptr = self.renderer.addPoints(pts, ints)

            time.sleep(1)

            prevtime = currtime
            prevpos = currpos



    def test_showPointsWithoutTranslation(self):

        ptr = None

        counter = 0
        while True:
            counter += 1
            pts = self.getLidarPoints(counter)
            if ptr is not None:
                self.renderer.freePoints(ptr)

            ptr = self.renderer.addPoints(pts)
            time.sleep(0.1)
            if counter > 40:
                counter = 0


    def test_accumulatePoints(self):
        for i in range(10):

            pts = self.getLidarPoints(i)

            ptr = self.renderer.addPoints(pts)
            time.sleep(0.5)

    def test_shouldAlignVertically(self):
        prev_time = None
        for i in range(10):
            lidar, oxts = self.environment.getNextFrameData(0)

            vel, rot, ttime = self.environment.getCurrentOxtsData(oxts)
            vertical = np.array([0, vel[1], 0])


            pts = lidar[0]
            if prev_time is not None:
                delvel = vertical * (ttime - prev_time).total_seconds()
                pts += delvel

            ptr = self.renderer.addPoints(pts)
            prev_time = ttime
            time.sleep(0.5)

    def test_shouldGet64ScanLines(self):
        path = "F://uni//3d-pointcloud//sample2"

        calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        oxtsDataReader = OxtsDataReader(path)
        lidarDataReader = LidarDataReader(path=path, oxtsDataReader=oxtsDataReader, calibration=calibration,
                                          targetCamera="02", max_read=50)

        names = lidarDataReader.getfilenames()

        self.renderer.addPoints(lidarDataReader.getPoints(names[0])[0])
        origin = np.array([0,0,0])
        for i in range(50):
            name = names[i]
            pts, cols = lidarDataReader.getPoints(name)
            sphere = self.icpContainer.sphereProjectPoints(pts + origin, origin) + origin * 10
            self.renderer.addPoints(sphere)
            scan_lines = self.icpContainer.getScanLines(pts + origin, origin)
            print(len(scan_lines))
            origin += np.array([1, 1, 0])


    def test_shouldProjectIntoCameraSpace(self):
        path = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"

        calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        oxtsDataReader = OxtsDataReader(path)
        lidarDataReader = LidarDataReader(path=path, oxtsDataReader=oxtsDataReader, calibration=calibration,
                                          targetCamera="02", max_read=10)

        filenames = lidarDataReader.getfilenames()

        name1 = filenames[0] + ".bin"

        np_points, intensities  = lidarDataReader.eatBytes(name1)
        image = lidarDataReader.readImage(name1.strip(".bin") + ".png")

        lidar_to_view = lidarDataReader.R_velo_to_cam @ np_points.T
        cam_0_points = lidar_to_view + lidarDataReader.t_velo_to_cam
        cam_0_rect = lidarDataReader.rect_00 @ cam_0_points

        color_indexes, colors, points2d = lidarDataReader.project_np(cam_0_rect.T, lidarDataReader.proj, lidarDataReader.width, lidarDataReader.height, image)
        points2d = points2d.T * 0.001

        # self.renderer.addPoints(points2d, colors)
        pts, cls = lidarDataReader.getPoints(filenames[0])
        self.renderer.addPoints(pts, cls)


    """def test_shouldVoxelizePoints(self):
        filenames = self.lidarDataReader.getfilenames()
        pts, cols = self.lidarDataReader.getPoints(filenames[0])

        voxelizer = Voxelizer(0.1)


        voxelizer.addPoints(pts, cols)

        v_pts, v_cls = voxelizer.getPoints()
        self.renderer.addPoints(v_pts, v_cls)


    def test_shouldVoxelizeSequence(self):
        ptr = None
        files = self.lidarDataReader.getfilenames()

        prevpos = np.array([0, 0, 0])
        currpos = np.array([0, 0, 0])
        prevtime = None
        currtime = self.oxtsDataReader.getOx(files[0]).getTime()

        voxelizer = Voxelizer(self.renderer, voxelSize=self.VOXEL_SIZE, maxDensity=1000)

        for i in range(10):
            name = files[i]
            oxts = self.oxtsDataReader.getOx(name)
            rot = np.linalg.inv(oxts.getTrueRotation(np.eye(3)))
            yawrot = oxts.getYawRotation(np.eye(3))
            rot = yawrot @ rot

            if prevtime is not None:
                velocity = -oxts.getVelocity()
                currtime = oxts.getTime()
                deltatime = (currtime - prevtime).total_seconds()
                velocity = (rot @ velocity.T).T

                currpos = prevpos + velocity * deltatime


            pts, cols = self.lidarDataReader.getPoints(files[i])
            pts = (rot @ pts.T).T + currpos


            voxelizer.addPoints(pts, cols)


            time.sleep(0.5)

            prevtime = currtime
            prevpos = currpos"""

    def getAlignedLidarPoints(self, num_scans):
        files = self.lidarDataReader.getfilenames()

        prevpos = np.array([0, 0, 0])
        currpos = np.array([0, 0, 0])
        prevtime = None
        currtime = self.oxtsDataReader.getOx(files[0]).getTime()

        aligned_points = []
        positions = [np.array([0,0,0])]
        rotations = []

        load = num_scans
        for i in range(load):
            name = files[i]
            oxts = self.oxtsDataReader.getOx(name)
            rot = np.linalg.inv(oxts.getTrueRotation(np.eye(3)))
            yawrot = oxts.getYawRotation(np.eye(3))
            rot = yawrot @ rot

            if prevtime is not None:
                velocity = -oxts.getVelocity()
                currtime = oxts.getTime()
                deltatime = (currtime - prevtime).total_seconds()
                velocity = (rot @ velocity.T).T

                currpos = prevpos + velocity * deltatime
                positions.append(currpos)

            pts = self.getLidarPoints(i)
            pts = (rot @ pts.T).T + currpos
            rotations.append(rot)

            aligned_points.append(pts)

            prevtime = currtime
            prevpos = currpos

        return aligned_points, positions, rotations

    def test_closestPointOutlier(self):
        aligned_points, poses, rots = self.getAlignedLidarPoints(10)

        outlier_filter = OutlierFilter(self.icpContainer, 1, 20)


        for i in range(9):
            reference = aligned_points[i]

            next = aligned_points[i + 1]

            start = time.time()
            outliers = outlier_filter.temporalClosestPointOutliers(next, reference)
            print("temp: " + str(time.time()-start))

            v = self.renderer.addPoints(outliers)
            time.sleep(1)
            self.renderer.freePoints(v)


            clusters = outlier_filter.getClusters(outliers)
            added_clusters = []
            for c in clusters:
                added_clusters.append(self.renderer.addPoints(c, self.randcolor()))

            time.sleep(10)

            for a in added_clusters:
                self.renderer.freePoints(a)

    def test_showClusters(self):


        pts = self.getLidarPoints()


        outlier_filter = OutlierFilter(icpContainer=self.icpContainer, dbscan_eps=0.5, dbscan_minpt=10)

        pts = Voxelizer.voxelGroundFilter(pts)

        clusters_a = outlier_filter.getClusters(pts)

        for c in clusters_a:
            self.renderer.addPoints(c, self.randcolor())


    def test_canAlignFilteredPoints(self):

        of = OutlierFilter(self.icpContainer, 0.5, 20, 50, True, False)
        ev = EnvironmentConstructor(self.renderer, self.oxtsDataReader, self.lidarDataReader, self.icpContainer)


        aligned, poses, rots = self.getAlignedLidarPoints(10)
        pc1, pc2, pc3 = aligned[0], aligned[1], aligned[2]


        fp1 = ev.filterPoints(pc2, pc1)
        fp2 = ev.filterPoints(pc3, pc2)
        """fp1 = pc1
        fp2 = pc2"""

        self.renderer.addPoints(fp1, np.array([255,0 ,0 ]))


        # fp2 += np.array([1, 0, 1])

        t_opt, r_opt = ev.getRefinedTransition(fp1 , np.array([0,0,0]), fp2, 30, True, self.renderer, False)

        fp2 = (r_opt @ fp2.T).T - t_opt

        self.renderer.addPoints(fp2, np.array([0,255,0]))



    def test_pointToPlane(self):
        environmentConstructor = EnvironmentConstructor(self.renderer, self.oxtsDataReader, self.lidarDataReader, self.icpContainer)

        start_from = 0
        until = 50
        for i in range(start_from, until):
            lidardata, oxts = environmentConstructor.getNextFrameData(start_from)
            environmentConstructor.calculateTransition_imu(lidardata,
                                                           oxts,
                                                           point_to_plane=True,
                                                           debug=False,
                                                           iterations=30,
                                                           cullColors=False,
                                                           removeOutliers=False,
                                                           pure_imu=False)
            # time.sleep(0.1)

    def test_shouldCluster(self):
        def randcolor():
            r = np.random.rand(1, 3)[0]
            return r




        ptr = None

        aligned_points, positions, rotations = self.getAlignedLidarPoints(30)



        # for i in range(len(aligned_points) - 1):

        index = 1
        ref_pts = aligned_points[index]
        other_pts = aligned_points[index - 1]

        """t_opt, R_opt = self.icpContainer.full_pt2pl(ref_pts, other_pts, positions[index], 20, None)

        other_pts = (R_opt @ other_pts.T).T - t_opt"""


        """v1 = self.renderer.addPoints(ref_pts)
        v2 = self.renderer.addPoints(other_pts)
        time.sleep(5)
        self.renderer.freePoints(v1)
        self.renderer.freePoints(v2)"""


        outlier_filter = OutlierFilter(icpContainer=self.icpContainer, dbscan_eps=0.5, dbscan_minpt=10, do_threading=True)

        start = time.time()
        outlier_mask = outlier_filter.getOutlierIndexes(ref_pts, other_pts, self.renderer, use_threads=True)
        print("total outlier indexing time: " + str(time.time()-start))


        self.renderer.addPoints(ref_pts[outlier_mask])


    def test_groundVoxelize(self):


        pts = self.getLidarPoints(0)
        pts = Voxelizer.voxelGroundFilter(pts)
        self.renderer.addPoints(pts)



    def test_showIntensities(self):
        pts, cls, ints = self.lidarDataReader.getPointsWithIntensities(0)
        self.renderer.addPoints(pts, ints)

    def test_canRenderVoxels(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None, voxel_size=0.2)

        pc = pts[0]

        st = time.time()
        v.addPoints(pc)
        print("expanson: " + str(time.time() - st))

        voxels = v.getVoxelPositions()

        self.renderer.addPoints(voxels)

        lines = v.getVoxelsAsLineGrid()
        self.renderer.setLines(lines)

    def generate_3d_grid(self, x_width, y_width, z_width, spacing=1.0):
        x = np.arange(0, x_width, spacing)
        y = np.arange(0, y_width, spacing)
        z = np.arange(0, z_width, spacing)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # 'ij' for proper 3D ordering
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # Shape (N, 3)

        return grid_points

    def test_voxelIntegrityLidar(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None, voxel_size=0.4)

        pc = pts[0]


        v.addPoints(pc)
        v.refreshVoxelData()

        for i in range(v.stored_voxels):
            vox_data_ptr = v.voxel_index[i][3]
            count = v.voxel_data[vox_data_ptr][0]
            if count > 1:
                pt_indexes = v.voxel_data[vox_data_ptr][1:count]
                pp = v.added_points[pt_indexes][:, :3]

                self.renderer.addPoints(pp, self.randcolor())




    def test_voxelIntegrityGrid(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None, voxel_size=0.4)

        # pc = pts[0]
        points = []
        added = []
        for i in range(10):
            pc = self.generate_3d_grid(0.6, 0.6, 0.6, 0.04) + PointcloudAlignment.randomTranslation1() * 2
            added.append(self.renderer.addPoints(pc))

            points.append(pc)

        time.sleep(2)
        for a in added:
            self.renderer.freePoints(a)

        for p in points:
            v.addPoints(p)

        v.refreshVoxelData()

        for i in range(v.getStoredVoxelCount()):
            vox_dat_ind = v.getVoxelDataIndexAt(i)
            count = v.getStoredCount(vox_dat_ind)
            if count > 1:
                pp = v.getStoredPoints(vox_dat_ind)

                self.renderer.addPoints(pp, self.randcolor())


                time.sleep(0.1)

        lines = v.getVoxelsAsLineGrid()
        self.renderer.setLines(lines)

    def test_idle(self):
        time.sleep(10)


    def test_showColors(self):
        self.lidarDataReader.init(self.path, self.calibration, "02", 10, None)
        self.oxtsDataReader.init(self.path)

        points, colors, ints = self.lidarDataReader.getPointsWithIntensities(self.lidarDataReader.getfilenames()[0])
        self.renderer.addPoints(points, colors)

    def test_canFilterStaticVoxels(self):
        self.lidarDataReader.init(self.path, self.calibration, "02", 200, None)
        self.oxtsDataReader.init(self.path)

        self.environment.init(voxel_size=0.5)

        for i in range(150):
            lidardata, oxts = self.environment.getNextFrameData(0)
            self.environment.calculateTransition_imu(lidardata, oxts, point_to_plane=True, debug=False,
                                                           iterations=10, separate_colors=True, removeOutliers=True, pure_imu=False )
        self.environment.voxelizer.stageAllRemaningVoxels()

    def test_can_renderVoxelData(self):
            pts, poses, rots = self.getAlignedLidarPoints(10)

            v = Voxelizer(self.computeShader, None, voxel_size=0.4)



            voxels = []


            for i in range(0, 10):
                pc = pts[i]

                st = time.time()
                v.addPoints(pc)
                print("expanson: " + str(time.time() - st))

            v.refreshVoxelData()

            voxels = []
            for i in range(v.getStoredVoxelCount()):
                vdat_ind = v.getVoxelDataIndexAt(i)
                stored = v.getStoredCount(vdat_ind)
                if stored > 10:
                    vox_points = v.getStoredPoints(vdat_ind)
                    voxels.append(self.renderer.addPoints(vox_points, self.randcolor()))






            """lines = v.getVoxelsAsLineGrid()
            self.renderer.setLines(lines)"""




    def test_shouldExpand(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None, voxel_size=0.4)

        pc = pts[0]

        st = time.time()
        v.addPoints(pc, True)
        print("expanson: " + str(time.time()-st))


        voxels = v.getVoxelPositions()



        st = time.time()
        v.addPoints(pts[1], True)
        print("2nd expanson: " + str(time.time()-st))

        st = time.time()
        v.addPoints(pts[2], True)
        print("3rd expanson: " + str(time.time() - st))

        self.renderer.addPoints(voxels)




    def test_rendererGraphics(self):
        self.renderer.addPoints(Renderer.unitCube())
        self.renderer.setLines(Renderer.unitCubeEdges())



    def test_gpuCanVoxelize(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None, voxel_size=0.5)

        pc = pts[0]

        st = time.time()
        v.addPoints(pc)
        print("expanson: " + str(time.time() - st))

        pc = pts[1]

        st = time.time()
        v.addPoints(pc)
        print("expanson: " + str(time.time() - st))

        pc = pts[2]

        st = time.time()
        v.addPoints(pc)
        print("expanson: " + str(time.time() - st))


    def test_voxelStats(self):
        pts, poses, rots = self.getAlignedLidarPoints(30)

        v = Voxelizer(self.computeShader, self.renderer, voxel_size=0.5)

        time.sleep(2)

        for i in range(0, 30):
            pc = pts[i]

            st = time.time()
            v.addPoints(pc)
            print("expanson: " + str(time.time() - st))










