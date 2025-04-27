import math
import time
import unittest

import numpy as np

from ComputeShader import ComputeShader
from EnvironmentConstructor import EnvironmentConstructor
from LidarDataReader import LidarDataReader
from OxtsDataReader import OxtsDataReader
from PointcloudAlignment import PointcloudAlignment
from PointcloudIcpContainer import PointcloudIcpContainer
from Renderer import Renderer
from Voxelizer import Voxelizer


class FunctionalTesting(unittest.TestCase):

    def setUp(self):
        self.path = "./../sample_data"
        self.calibration = "./../sample_data/sample_calib/2011_09_26"

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

        self.lidarDataReader.init(self.path, self.calibration, "02", 50, None)
        self.oxtsDataReader.init(self.path)
        self.environment.init(0.5)

    def tearDown(self):
        self.lidarDataReader.cleanup()
        self.oxtsDataReader.cleanup()
        self.environment.cleanup()

        self.computeShader.cleanup()
        self.renderingThread.join()


    def getLidarPoints(self, index=0):
        filenames = self.lidarDataReader.getfilenames()
        pts, cols = self.lidarDataReader.getPoints(filenames[index])
        return pts

    def randcolor(self):
        return np.random.rand(3, )

    def test_canDisplayPoints(self):
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


        self.renderer.close()


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
        for i in range(0, len(points[:50000]), 1):
            normal = getNormal(scan_lines, i, points, origin)
            normal = normalTowardsOrigin(origin, normal, i, points)
            normals.append(normal)

            if i % 2000 == 0:
                print(str(i) + "/50000")

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

        self.renderer.close()


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

        self.renderer.close()

    def test_getNormalsTest(self):
        lines = []






        def addLine(p1, p2, color = None):
            lines.extend([p1, p2])



        origin = np.array([0, 0, 0])
        pts = self.getLidarPoints() + origin

        self.renderer.addPoints(pts)
        scan_lines = self.icpContainer.getScanLines(pts, origin)


        normals = FunctionalTesting.getNormals(scan_lines, pts, origin, self.icpContainer)

        for i in range(0, len(normals), 1):
            refp = pts[i]
            normal = normals[i]
            addLine(refp, refp + normal * 0.1)

            if i % 1000 == 0:
                self.renderer.setLines(lines)

        self.assertTrue(True)
        self.renderer.close()

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
            if refp[1] > -1.6 and np.linalg.norm(refp) < 20:
                addLine(refp, refp + normal * 0.1)

            if i % 1000 == 0 and len(lines) > 1:
                self.renderer.setLines(lines, colors)

        self.renderer.close()


    def test_pointToPointLSShouldConverge(self):

        origin = np.array([0, 0, 0])




        pts1 = self.getLidarPoints(1)

        pts2 = self.getLidarPoints(2)

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

        for i in range(100):
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
        self.renderer.close()


    def test_findClosestPointGpu(self):
        self.lidarDataReader.init(self.path, self.calibration, "02", 10, None)

        origin = np.array([0, 0, 0])

        frame1 = 0
        frame2 = 3

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

        self.renderer.close()

    def test_pointToPlaneLSShouldConverge(self):



        origin = np.array([0, 0, 0])

        key1 = 0
        key2 = 4


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

        time_acc = 0
        for i in range(100):
            start = time.time()
            t, R, delta_transf = self.icpContainer.dispatchPointToPlane(pts2)
            print("dispatch pointplane: " + str(time.time() - start))
            time_acc += (time.time()-start)

            start = time.time()
            pts2 = self.icpContainer.apply_iteration_of_ls(pts2, t, R)
            print("application of icp: " + str(time.time()-start))
            reference_grid = self.icpContainer.applyIcpStep(reference_grid, R, t)

            self.renderer.freePoints(rgp)
            rgp = self.renderer.addPoints(reference_grid, self.blue)

            self.renderer.freePoints(pp2)
            pp2 = self.renderer.addPoints(pts2, self.white)
            # time.sleep(0.1)
            print(i)

        print("average ICP time:" + str(time_acc/100.0))


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
        self.renderer.close()

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


        self.renderer.close()

    def test_findClosestPoint(self):
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

        self.renderer.close()

    def test_showScanLines(self):

        names = self.lidarDataReader.getfilenames()
        pts, cols = self.lidarDataReader.getPoints(names[0])

        scan_lines = self.icpContainer.getScanLines(pts, np.array([0,0,0]))
        # print(len(scan_lines))
        self.renderer.addPoints(pts)
        lines = []
        colors = []
        for start, end in scan_lines:
            scan_points = pts[start:end+1]
            line = self.icpContainer.connectPointsInOrder(scan_points)
            lines.extend(line)
            colors.extend(self.renderer.handleColors(line, self.randcolor()))
            self.renderer.setLines(lines, colors)
            time.sleep(0.1)

        self.renderer.close()

    def test_projectLidarToSphere(self):
        filenames = self.lidarDataReader.getfilenames()
        pts1, cols1 = self.lidarDataReader.getPoints(filenames[0])
        pts2, cols2 = self.lidarDataReader.getPoints(filenames[1])
        origin = np.array([0, 0, 0])

        pc = self.icpContainer.sphereProjectPoints(pts1, origin)

        lines = self.icpContainer.connectPointsInOrder(pc)
        self.renderer.addPoints(pc, np.array([1, 1, 1]))
        # self.renderer.setLines(lines)

        self.renderer.close()

    def test_shouldGetApproximatePath(self):
        lidar, oxts = self.environment.getNextFrameData(0)


        lines = []
        colors = []


        def addLine(p1, p2, color):
            lines.extend([p1, p2])
            colors.extend([color, color])

        offset = 0
        prev_time = oxts.getTime()
        prev_position = np.array([0,0,0])
        for i in range(self.lidarDataReader.count - 1):
            lidardata, oxts = self.environment.getNextFrameData(offset)

            current_velocity, current_horz_rot, current_vert_rot, current_time = self.environment.getCurrentOxtsData(
                current_oxts=oxts)
            current_horz_rot = np.linalg.inv(current_horz_rot)

            current_rotation = current_vert_rot @ current_horz_rot
            delta_velocity = self.environment.getDeltaVelocity(current_velocity, current_rotation, current_time, prev_time)

            addLine(prev_position, prev_position + delta_velocity, self.red)
            # addLine(self.environment.prev_position, self.environment.prev_position + current_velocity, self.green)

            prev_position = prev_position + delta_velocity

            prev_time = current_time
            self.renderer.addPoints([prev_position], self.red)

            self.renderer.setLines(lines, colors)
            time.sleep(0.1)

        self.renderer.close()

    def test_shouldAlignBasedOnImu(self): ### !!! this is the real deal
        self.lidarDataReader.init(self.path, self.calibration, "02", 10, None)
        self.oxtsDataReader.init(self.path)
        ptr = None
        files = self.lidarDataReader.getfilenames()

        prevpos = np.array([0, 0, 0])
        currpos = np.array([0, 0, 0])
        prevtime = None
        currtime = self.oxtsDataReader.getOx(files[0]).getTime()

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


            pts = self.getLidarPoints(i)
            pts = (rot @ pts.T).T + currpos

            if ptr is not None:
                # self.renderer.freePoints(ptr)
                a=0
            _, cls, ints = self.lidarDataReader.getPointsWithIntensities(files[i])
            ptr = self.renderer.addPoints(pts, ints)
            self.renderer.addPoints([currpos], np.array([255,0,0]))

            time.sleep(1)

            prevtime = currtime
            prevpos = currpos

        self.renderer.close()


    def test_showPointsWithoutTranslation(self):

        ptr = None

        counter = 0
        for i in range(50):
            counter += 1
            pts = self.getLidarPoints(counter)
            if ptr is not None:
                self.renderer.freePoints(ptr)

            ptr = self.renderer.addPoints(pts)
            time.sleep(0.1)
            if counter >= self.lidarDataReader.count - 1:
                counter = 0

        self.renderer.close()

    def test_accumulatePoints(self):
        for i in range(10):

            pts = self.getLidarPoints(i)

            ptr = self.renderer.addPoints(pts)
            time.sleep(0.5)

        self.renderer.close()

    def test_shouldGet64ScanLines(self):


        names = self.lidarDataReader.getfilenames()

        self.renderer.addPoints(self.lidarDataReader.getPoints(names[0])[0])
        origin = np.array([0,0,0])
        for i in range(self.lidarDataReader.count - 1):
            name = names[i]
            pts, cols = self.lidarDataReader.getPoints(name)
            sphere = self.icpContainer.sphereProjectPoints(pts + origin, origin) + origin * 10
            self.renderer.addPoints(sphere)
            scan_lines = self.icpContainer.getScanLines(pts + origin, origin)
            print(len(scan_lines))
            origin += np.array([1, 1, 0])

        self.renderer.close()


    def test_shouldProjectIntoCameraSpace(self):



        filenames = self.lidarDataReader.getfilenames()

        name1 = filenames[0] + ".bin"

        np_points, intensities  = self.lidarDataReader.eatBytes(name1)
        image = self.lidarDataReader.readImage(name1.strip(".bin") + ".png")

        lidar_to_view = self.lidarDataReader.R_velo_to_cam @ np_points.T
        cam_0_points = lidar_to_view + self.lidarDataReader.t_velo_to_cam
        cam_0_rect = self.lidarDataReader.rect_00 @ cam_0_points

        color_indexes, colors, points2d = self.lidarDataReader.project_np(cam_0_rect.T, self.lidarDataReader.proj, self.lidarDataReader.width, self.lidarDataReader.height, image)
        points2d = points2d.T * 0.001

        # self.renderer.addPoints(points2d, colors)
        pts, cls = self.lidarDataReader.getPoints(filenames[0])
        self.renderer.addPoints(pts, cls)

        time.sleep(1)

        self.renderer.close()


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


    def test_groundVoxelize(self):


        pts = self.getLidarPoints(0)
        pts = Voxelizer.voxelGroundFilter(pts)
        self.renderer.addPoints(pts)
        time.sleep(2)
        self.renderer.close()


    def test_showIntensities(self):
        pts, cls, ints = self.lidarDataReader.getPointsWithIntensities(self.lidarDataReader.getfilenames()[0])
        self.renderer.addPoints(pts, ints)
        time.sleep(2)
        self.renderer.close()

    def test_canRenderVoxels(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None)
        v.init(0.5)

        pc = pts[0]

        st = time.time()
        v.addPoints(pc, None)
        print("expanson: " + str(time.time() - st))

        voxels = v.getVoxelPositions()

        self.renderer.addPoints(voxels)

        lines = v.getVoxelsAsLineGrid()
        self.renderer.setLines(lines)

        time.sleep(2)
        self.renderer.close()

    def generate_3d_grid(self, x_width, y_width, z_width, spacing=1.0):
        x = np.arange(0, x_width, spacing)
        y = np.arange(0, y_width, spacing)
        z = np.arange(0, z_width, spacing)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # 'ij' for proper 3D ordering
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # Shape (N, 3)

        return grid_points

    def test_voxelIntegrityLidar(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None)
        v.init(0.4)

        pc = pts[0]


        v.addPoints(pc, None)
        v.refreshVoxelData()

        for i in range(v.stored_voxels):
            vox_data_ptr = v.voxel_index[i][3]
            count = v.voxel_data[vox_data_ptr][0]
            if count > 1:
                pt_indexes = v.voxel_data[vox_data_ptr][1:count]
                pp = v.added_points[pt_indexes][:, :3]

                self.renderer.addPoints(pp, self.randcolor())

        time.sleep(2)
        self.renderer.close()

    def test_voxelIntegrityGrid(self):
        pts, poses, rots = self.getAlignedLidarPoints(10)

        v = Voxelizer(self.computeShader, None)
        v.init(0.3)

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
            v.addPoints(p, None)

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
        time.sleep(2)
        self.renderer.close()

    def test_showColors(self):
        self.lidarDataReader.init(self.path, self.calibration, "02", 10, None)
        self.oxtsDataReader.init(self.path)

        points, colors, ints = self.lidarDataReader.getPointsWithIntensities(self.lidarDataReader.getfilenames()[0])
        self.renderer.addPoints(points, colors)

        time.sleep(2)
        self.renderer.close()
    def test_canFilterStaticVoxels(self):


        self.environment.cleanup()
        self.environment.init(voxel_size=0.5)

        for i in range(30):
            lidardata, oxts = self.environment.getNextFrameData(0)
            self.environment.calculateTransition_imu(lidardata, oxts, point_to_plane=True, debug=False,
                                                           iterations=10, separate_colors=True, removeOutliers=True, pure_imu=True )
        self.environment.voxelizer.stageAllRemaningVoxels()
        time.sleep(2)
        self.renderer.close()



    def test_rendererGraphics(self):
        self.renderer.addPoints(Renderer.unitCube())
        self.renderer.setLines(Renderer.unitCubeEdges())

        time.sleep(2)
        self.renderer.close()







