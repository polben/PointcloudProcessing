import math
import time
import unittest
import numpy as np

from ComputeShader import ComputeShader
from LidarDataReader import LidarDataReader
from OxtsDataReader import OxtsDataReader
from PointcloudAlignment import PointcloudAlignment
from PointcloudIcpContainer import PointcloudIcpContainer


class GpuTest(unittest.TestCase):

    def setUp(self):
        self.path = "./../sample_data"
        self.calibration = "./../sample_data/sample_calib/2011_09_26"

        self.oxtsDataReader = OxtsDataReader()
        self.lidarDataReader = LidarDataReader()

        self.pointcloudAlignment = PointcloudAlignment()

        self.computeShader = ComputeShader()
        self.icpContainer = PointcloudIcpContainer(self.computeShader, self.pointcloudAlignment)

        self.tolerance = 1e-3 # 0.001

        self.baseComputeShader = "FunctionalBase.glsl"
        self.mainRegex = self.computeShader.main_shadercode_regex

        self.lidarDataReader.init(self.path, self.calibration, "02", 30, None)

    def tearDown(self):
        self.computeShader.cleanup()

    def getLidarPoints(self):
        filenames = self.lidarDataReader.getfilenames()
        pts, cols = self.lidarDataReader.getPoints(filenames[0])
        return pts

    def test_shouldInitPrograms(self):
        self.assertTrue(self.computeShader.least_squares_point is not None)
        self.assertTrue(self.computeShader.nearest_neighbour is not None)

    def test_shouldInitBuffers(self):
        self.assertTrue(self.computeShader.ssbo_Hs is not None)
        self.assertTrue(self.computeShader.ssbo_correspondence is not None)
        self.assertTrue(self.computeShader.ssbo_points_b is not None)
        self.assertTrue(self.computeShader.ssbo_points_a is not None)
        self.assertTrue(self.computeShader.ssbo_scan_lines is not None)

        self.assertTrue(self.computeShader.nearest_neighbour is not None)
        self.assertTrue(self.computeShader.least_squares_point is not None)

    def test_shouldCleanupBuffers(self):
        self.computeShader.cleanup()

        self.assertTrue(self.computeShader.ssbo_Hs is None)
        self.assertTrue(self.computeShader.ssbo_correspondence is None)
        self.assertTrue(self.computeShader.ssbo_points_b is None)
        self.assertTrue(self.computeShader.ssbo_points_a is None)
        self.assertTrue(self.computeShader.ssbo_scan_lines is None)

        self.assertTrue(self.computeShader.nearest_neighbour is None)
        self.assertTrue(self.computeShader.least_squares_point is None)


    def test_initialOutRealloc(self):
        self.assertEqual(len(self.computeShader.hs_out), self.computeShader.maxPointsPerCloud)
        self.assertEqual(len(self.computeShader.corr_out), self.computeShader.maxPointsPerCloud)

    def test_shouldCompileBaseShader(self):
        code ="""
            void main(){
            
            
            }
        """
        program = self.computeShader.create_shader_program(code)
        self.computeShader.deleteProgram(program)

    def test_shouldFindCodeInBaseShader(self):
        basecode = self.computeShader.glslFile(self.baseComputeShader)
        self.assertTrue(basecode.find(self.mainRegex) > 0)

        basecode = self.computeShader.extend_functional_code("//#e_e#")
        self.assertTrue(basecode.find(self.mainRegex) == -1)
        self.assertTrue(basecode.find("//#e_e#") > 1)



    def test_shouldDispatchCorrentNumberOfThreads(self):

        points_a = np.array([[0, 0, 0]])
        points_b = np.array([[1, 1, 1]])
        origin = np.array([0, 0, 0])



        code = """
        void main(){
            uvec2 id = gl_GlobalInvocationID.xy;
            uint idx = id.x;
            uint idy = id.y;
        
            if (idx >= lens_data.y) {
                return;
            }
        
            corr[idx] = vec4(float(idx), 6.0, 9.0, -1.0);
        }
        """


        program = self.computeShader.create_shader_program(code)
        self.computeShader.setActiveProgram(program)


        self.computeShader.setUniform(program, "lens_data", np.array([1, 1, 0, 0]))

        self.computeShader.dispatchCurrentProgramWait(1)

        self.computeShader.getBufferFullData(self.computeShader.ssbo_correspondence)
        self.assertTrue(len(self.computeShader.correspondences) == self.computeShader.maxPointsPerCloud)

        read_corrs = self.computeShader.correspondences
        self.assertTrue(read_corrs.shape == (self.computeShader.maxPointsPerCloud, 4))
        self.assertTrue(np.array_equal(read_corrs[0], np.array([0.0, 6.0, 9.0, -1.0])))

        zeros = np.zeros((self.computeShader.maxPointsPerCloud - 1, 4))
        self.assertTrue(np.array_equal(read_corrs[1:], zeros))

        zeros = np.zeros((self.computeShader.maxPointsPerCloud, 4))
        self.assertFalse(np.array_equal(read_corrs, zeros))



    def test_shouldSetPoints(self):

        points_a = np.array([[0, -123, 0]]).astype(np.float32)
        points_b = np.array([[1, 1, 1]]).astype(np.float32)
        origin = np.array([0, 0, 0]).astype(np.float32)



        result_ssbo = self.computeShader.bufferSubdata(points_a, self.computeShader.ssbo_points_a)
        self.assertTrue(result_ssbo == self.computeShader.ssbo_points_a)

        result_ssbo = self.computeShader.bufferSubdata(points_b, self.computeShader.ssbo_points_b)
        self.assertTrue(result_ssbo == self.computeShader.ssbo_points_b)





        self.computeShader.getBufferFullData(self.computeShader.ssbo_points_a)
        points_a_from_shader = np.array([self.computeShader.points_a[0][:3]])
        self.assertTrue(np.array_equal(points_a_from_shader, points_a))

        self.computeShader.getBufferFullData(self.computeShader.ssbo_points_b)
        points_b_from_shader = np.array([self.computeShader.points_b[0][:3]])
        self.assertTrue(np.array_equal(points_b_from_shader, points_b))



    def test_shouldPersistBuffersOverDispatches(self):

        code = """
            void main(){
                uvec2 id = gl_GlobalInvocationID.xy;
                uint idx = id.x;
                uint idy = id.y;
            
                if (idx >= lens_data.y) {
                    return;
                }
                corr[idx] = vec4(corr[idx][0] + 1.0, 6.0, 9.0, -1.0);
            }
        """




        program = self.computeShader.create_shader_program(code)
        self.computeShader.setActiveProgram(program)

        points_a = np.array([[0, 0, 0]])
        points_b = np.array([[1, 1, 1]])
        origin = np.array([0, 0, 0])

        lens_data = np.array([1, 1, 0, 0]).astype(np.uint32)
        self.computeShader.setUniform(program, "lens_data", lens_data)
        self.computeShader.lens_data = lens_data

        self.computeShader.dispatchCurrentProgramWait(1)

        self.computeShader.getBufferSubdata(self.computeShader.ssbo_correspondence)
        self.assertTrue(self.computeShader.corr_out.shape == (1, 4))

        expected_value = np.array([1.0, 6.0, 9.0, -1.0])
        self.assertTrue(np.array_equal(self.computeShader.corr_out[0], expected_value))

        for i in range(5):
            self.computeShader.dispatchCurrentProgramWait(1)
            self.computeShader.getBufferSubdata(self.computeShader.ssbo_correspondence)
            self.assertTrue(self.computeShader.corr_out.shape == (1, 4))

            expected_value = np.array([2.0 + float(i), 6.0, 9.0, -1.0])
            self.assertTrue(np.array_equal(self.computeShader.corr_out[0], expected_value))

    def test_shouldAccessPointsB(self):
        code = """
            void main(){
                uvec2 id = gl_GlobalInvocationID.xy;
                uint idx = id.x;
                uint idy = id.y;
            
                if (idx >= lens_data.y) {
                    return;
                }
            
                corr[idx] = points_b[idx];
            }
        """



        program = self.computeShader.create_shader_program(code)
        self.computeShader.setActiveProgram(program)

        points_a = np.tile(np.array([1, 1, 1]), (100, 1))
        points_b = np.tile(np.array([123, 123, 123, 123]), (200, 1)).astype(np.float32)
        origin = np.array([0, 0, 0])
        scan_lines = np.zeros((len(points_a), 4))

        self.computeShader.saveUniformInfo(points_a, points_b, origin, scan_lines, True)
        self.computeShader.setCommonUniforms(program)

        self.computeShader.bufferSubdata(points_a, self.computeShader.ssbo_points_a)
        self.computeShader.bufferSubdata(points_b, self.computeShader.ssbo_points_b)
        self.computeShader.bufferSubdata(scan_lines, self.computeShader.ssbo_scan_lines)

        self.computeShader.dispatchCurrentProgramWait(len(points_b))


        self.computeShader.getBufferSubdata(self.computeShader.ssbo_correspondence)
        corrs = self.computeShader.corr_out
        self.assertTrue(corrs.shape == points_b.shape)
        self.assertTrue(np.array_equal( corrs, points_b))

    def test_shouldAccessPointsA(self):
        code = """
            void main(){
                uvec2 id = gl_GlobalInvocationID.xy;
                uint idx = id.x;
                uint idy = id.y;
            
                if (idx >= lens_data.y) {
                    return;
                }
                corr[idx] = points_a[idx];
            }
        """


        program = self.computeShader.create_shader_program(code)
        self.computeShader.setActiveProgram(program)

        points_a = np.tile(np.array([1, 1, 1, 1]), (100, 1)).astype(np.float32)
        points_b = np.tile(np.array([123, 123, 123, 123]), (200, 1)).astype(np.float32)
        origin = np.array([0, 0, 0])
        scan_lines = np.zeros((len(points_a), 4))

        self.computeShader.setUniform(program, "lens_data", np.array([0, 100, 0, 0]))

        self.computeShader.bufferSubdata(points_a, self.computeShader.ssbo_points_a)
        self.computeShader.bufferSubdata(points_b, self.computeShader.ssbo_points_b)
        self.computeShader.bufferSubdata(scan_lines, self.computeShader.ssbo_scan_lines)

        self.computeShader.dispatchCurrentProgramWait(len(points_a))


        self.computeShader.getBufferSubdata(self.computeShader.ssbo_correspondence)
        corrs = self.computeShader.corr_out
        self.assertTrue(corrs.shape == points_a.shape)
        self.assertTrue(np.array_equal( corrs, points_a))


    def test_shouldGetDataInNormals(self):
        points_a = np.tile(np.array([1, 1, 1, 1]), (100, 1)).astype(np.float32)
        origin = np.array([0, 0, 0])
        scan_lines = np.zeros((len(points_a), 4))
        self.computeShader.prepareDispatchNormals(points_a, scan_lines, origin)
        self.assertTrue(self.computeShader.normals_out_a.shape == points_a.shape)

    def test_shouldGetSameNormalsFromGpu(self):
        pts = self.getLidarPoints()
        origin = np.array([0, 0, 0])
        scan_lines = self.icpContainer.getScanLines(pts, origin)

        start = time.time()
        self.computeShader.prepareDispatchNormals(pts, scan_lines, origin)
        print("gpu normal time: " + str(time.time() - start))
        normals = self.computeShader.normals_out_a[:50000]

        normals_ref = GpuTest.getNormals(scan_lines, pts, origin, self.icpContainer)
        normals_ref = np.array(normals_ref).astype(np.float32)
        normals_ref = np.insert(normals_ref, 3, 0, axis=1)
        diff = normals - normals_ref # there is a single normal that is different?

        ok = 0
        err = 0
        for p in diff:
            if np.linalg.norm(p) < self.tolerance:
                ok += 1
            else:
                err += 1

        ok_percent = float(ok) / len(diff)
        print(ok_percent)
        print("error: " + str(err) + " / " + str(len(diff)))
        self.assertTrue(ok_percent > 0.999)

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

    def test_shouldSetHs(self):
        code = """
            void main(){
                uvec2 id = gl_GlobalInvocationID.xy;
                uint idx = id.x;
                uint idy = id.y;
            
                if (idx >= lens_data.y) {
                    return;
                }
                Hs[idx][0][0] = -12.3;
                Hs[idx][0][1] = float(lens_data.x);
                Hs[idx][0][2] = float(lens_data.y);
                Hs[idx][0][3] = float(lens_data.z);
                Hs[idx][0][4] = float(idx);
                Hs[idx][7][7] = -1.0;
            }
        """

        program = self.computeShader.create_shader_program(code)
        self.computeShader.setActiveProgram(program)

        points_a = np.tile(np.array([1, 1, 1]), (100, 1))
        points_b = np.tile(np.array([123, 123, 123, 123]), (200, 1)).astype(np.float32)
        origin = np.array([0, 0, 0])
        scan_lines = np.zeros((len(points_a), 4))

        self.computeShader.saveUniformInfo(points_a, points_b, origin, scan_lines, True)
        self.computeShader.setCommonUniforms(program)

        self.computeShader.bufferSubdata(points_a, self.computeShader.ssbo_points_a)
        self.computeShader.bufferSubdata(points_b, self.computeShader.ssbo_points_b)
        self.computeShader.bufferSubdata(scan_lines, self.computeShader.ssbo_scan_lines)

        self.computeShader.dispatchCurrentProgramWait(len(points_b))


        self.computeShader.getBufferSubdata(self.computeShader.ssbo_Hs)
        h1 = self.computeShader.hs_out[0]
        h2 = self.computeShader.hs_out[1]
        h3 = self.computeShader.hs_out[2]




    def test_rotationOriginTranslationSame(self):
        grid1 = self.icpContainer.getUniformGrid(10) + np.array([0, 1, 0])
        grid2 = self.icpContainer.getUniformGrid(10) + np.array([10, 0, 0])

        r = PointcloudAlignment.randomRotation(0.01)

        grid1 = (r @ grid1.T).T
        grid2 = (r @ grid2.T).T

        mean1 = np.mean(grid1, axis=0)
        grid1 = grid1 - mean1

        mean2 = np.mean(grid2, axis=0)
        grid2 = grid2 - mean2

        np.allclose(grid1, grid2, self.tolerance)



    def test_H_unrollEqualsNumpyBehaviour(self):

        point = np.array([12.23, -12.82, 0.001])

        j = self.icpContainer.getJ_i(point)

        h_numpy = j.T @ j

        h_unroll = self.icpContainer.getH_i(j)

        self.assertTrue(np.allclose(h_numpy, h_unroll, atol=self.tolerance))

    def test_B_unrollEqualsNumpyBehaviour(self):

        error = np.array([1234.123, 12, -0.001])
        point = np.array([12.23, -12.82, 0.001])

        j = self.icpContainer.getJ_i(point)

        b_numpy = j.T @ error

        b_unroll = self.icpContainer.getB(j, error)

        self.assertTrue(np.allclose(b_numpy, b_unroll, atol=self.tolerance))

    def test_uniformGridNumpyEqualLoop(self):


        grid1 = self.icpContainer.getUniformGrid(10)

        R = PointcloudAlignment.randomRotation(0.4)
        t = PointcloudAlignment.randomTranslation1() * 10
        grid2 = (R @ self.icpContainer.getUniformGrid(10).T).T + t

        Hs_loop, Bs_loop = self.icpContainer.getHsBsLoop(grid1, grid2)
        Hs_numpy, Bs_numpy = self.icpContainer.getHsBsNumpy(grid1, grid2)

        self.assertTrue(np.allclose(Hs_loop, Hs_numpy, atol=self.tolerance))
        self.assertTrue(np.allclose(Bs_loop, Bs_numpy, atol=self.tolerance))

    def test_uniformGridComputeEqualNumpy(self):

        grid1 = self.icpContainer.getUniformGrid(10)

        R = PointcloudAlignment.randomRotation(0.4)
        t = PointcloudAlignment.randomTranslation1() * 10
        grid2 = (R @ self.icpContainer.getUniformGrid(10).T).T + t


        grid1 = grid1.astype(np.float32)
        grid2 = grid2.astype(np.float32)
        Hs_numpy, Bs_numpy = self.icpContainer.getHsBsNumpy(grid1, grid2)
        H_numpy = np.sum(Hs_numpy, axis=0)
        B_numpy = np.sum(Bs_numpy, axis=0)

        self.computeShader.prepareLS(grid1, [(0, 0)], grid2, np.array([0, 0, 0]), True)

        H_cuda, B_cuda = self.computeShader.dispatchLS(grid2)


        self.assertTrue(np.allclose(H_cuda, H_numpy, atol=self.tolerance))
        self.assertTrue(np.allclose(B_cuda, B_numpy.T, atol=self.tolerance))



