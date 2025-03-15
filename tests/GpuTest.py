import time
import unittest
import numpy as np

from ComputeShader import ComputeShader
from LidarDataReader import LidarDataReader
from OxtsDataReader import OxtsDataReader
from PointcloudAlignment import PointcloudAlignment
from PointcloudIcpContainer import PointcloudIcpContainer
from tests.FunctionalTesting import FunctionalTesting


class GpuTest(unittest.TestCase):

    def setUp(self):
        path = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"
        calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"

        self.oxtsDataReader = OxtsDataReader(path)
        self.lidarDataReader = LidarDataReader(path=path, oxtsDataReader=self.oxtsDataReader, calibration=calibration,
                                          targetCamera="02")

        self.pointcloudAlignment = PointcloudAlignment(self.lidarDataReader, self.oxtsDataReader)

        self.computeShader = ComputeShader()
        self.icpContainer = PointcloudIcpContainer(self.computeShader, self.pointcloudAlignment)

        self.tolerance = 1e-5 # 0.00001

        self.baseComputeShader = "FunctionalBase.glsl"
        self.mainRegex = self.computeShader.main_shadercode_regex

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
        self.assertTrue(self.computeShader.ssbo_Bs is not None)
        self.assertTrue(self.computeShader.ssbo_Hs is not None)
        self.assertTrue(self.computeShader.ssbo_correspondence is not None)
        self.assertTrue(self.computeShader.ssbo_points_b is not None)
        self.assertTrue(self.computeShader.ssbo_points_a is not None)
        self.assertTrue(self.computeShader.ssbo_scan_lines is not None)

        self.assertTrue(self.computeShader.nearest_neighbour is not None)
        self.assertTrue(self.computeShader.least_squares_point is not None)

    def test_shouldCleanupBuffers(self):
        self.computeShader.cleanup()

        self.assertTrue(self.computeShader.ssbo_Bs is None)
        self.assertTrue(self.computeShader.ssbo_Hs is None)
        self.assertTrue(self.computeShader.ssbo_correspondence is None)
        self.assertTrue(self.computeShader.ssbo_points_b is None)
        self.assertTrue(self.computeShader.ssbo_points_a is None)
        self.assertTrue(self.computeShader.ssbo_scan_lines is None)

        self.assertTrue(self.computeShader.nearest_neighbour is None)
        self.assertTrue(self.computeShader.least_squares_point is None)


    def test_initialOutRealloc(self):
        self.assertEqual(len(self.computeShader.hs_out), self.computeShader.maxPointsPerCloud)
        self.assertEqual(len(self.computeShader.bs_out), self.computeShader.maxPointsPerCloud)
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
        normals = self.computeShader.normals_out_a

        normals_ref = FunctionalTesting.getNormals(scan_lines, pts, origin, self.icpContainer)
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

    def test_shouldSetHandB(self):
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
                
                Bs[idx][0] = -999.3;
                Bs[idx][1] = float(idx);
                Bs[idx][2] = float(lens_data.x);
                Bs[idx][3] = float(lens_data.y);
                Bs[idx][4] = float(lens_data.z);
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

        self.computeShader.getBufferSubdata(self.computeShader.ssbo_Bs)



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

        Hs_cuda, Bs_cuda = self.computeShader.dispatchLS(grid2)
        H_cuda = np.sum(Hs_cuda, axis=0)[:6, :6]
        B_cuda = np.sum(Bs_cuda, axis=0)[:6]



        self.assertTrue(np.allclose(H_cuda, H_numpy, atol=self.tolerance))
        self.assertTrue(np.allclose(B_cuda, B_numpy, atol=self.tolerance))

