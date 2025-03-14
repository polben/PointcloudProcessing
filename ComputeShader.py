import time
from ctypes import c_void_p

from OpenGL.GL import *
import glfw

import numpy as np


class ComputeShader:

    def __init__(self, maxPointsPerCloud=200000, maxScans=200):




        if not glfw.init():
            raise Exception("Failed to initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Hide the window
        self.window = glfw.create_window(100, 100, "Hidden Context", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create a hidden OpenGL context")

        glfw.make_context_current(self.window)  # Make the context current



        # max_invocations = glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
        # print(f"Max Compute Work Group Invocations: {max_invocations}")
        #print(glGetString(GL_RENDERER))  # GPU renderer info
        #print(glGetString(GL_VENDOR))  # Vendor info
        #print(glGetString(GL_VERSION))  # OpenGL version

        self.float_size = 4

        self.maxPointsPerCloud = maxPointsPerCloud
        self.maxScans = maxScans

        self.ssbo_points_a = None
        self.ssbo_points_b = None
        self.ssbo_scan_lines = None
        self.ssbo_correspondence = None
        self.ssbo_Hs = None
        self.ssbo_Bs = None
        self.ssbo_normals_a = None
        self.ssbo_normals_b = None


        self.origin = None
        self.origin_location = None
        self.debug_location = None
        self.debug_info = None
        self.lens_location = None
        self.lens_data = None

        self.Hs = None
        self.Bs = None
        self.points_a = None
        self.points_b = None
        self.scan_lines = None
        self.correspondences = None
        self.normals_a = None
        self.normals_b = None

        self.hs_out = None
        self.bs_out = None
        self.corr_out = None
        self.prev_out_a = 0
        self.prev_out_b = 0
        self.normals_out_a = None
        self.normals_out_b = None


        self.least_squares_point = None
        self.nearest_neighbour = None
        self.normal_shader = None
        self.point_plane_shader = None

        self.prepareBuffers()
        self.preparePrograms()


    def glslFile(self, name):
        path = "computeshaders//" + name
        code = ""
        try:
            with open(path, "r") as f:
                code = f.read()
        except FileNotFoundError:
            path = "..//computeshaders//" + name
            with open(path, "r") as f:
                code = f.read()

        return code

    def preparePrograms(self):
        self.least_squares_point = self.create_shader_program(self.glslFile("LeastSquaresPoint.glsl"))
        self.nearest_neighbour = self.create_shader_program(self.glslFile("NearestNeighbourScanV2.glsl"))
        self.normal_shader = self.create_shader_program(self.glslFile("NormalShader.glsl"))
        self.point_plane_shader = self.create_shader_program(self.glslFile("LeastSquaresPlane.glsl"))


    def prepareBuffers(self):
        self.ssbo_points_a = glGenBuffers(1)
        self.ssbo_points_b = glGenBuffers(1)
        self.ssbo_scan_lines = glGenBuffers(1)
        self.ssbo_correspondence = glGenBuffers(1)
        self.ssbo_Hs = glGenBuffers(1)
        self.ssbo_Bs = glGenBuffers(1)
        self.ssbo_normals_a = glGenBuffers(1)
        self.ssbo_normals_b = glGenBuffers(1)

        self.points_a = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)
        self.points_b = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)
        self.scan_lines = np.zeros((self.maxScans, 4)).astype(np.uint32)
        self.correspondences = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)
        """self.correspondences[0][0] = 1
        self.correspondences[0][1] = 2
        self.correspondences[0][2] = 3
        self.correspondences[0][3] = 4"""
        self.Hs = np.zeros((self.maxPointsPerCloud, 8, 8)).astype(np.float32)
        self.Bs = np.zeros((self.maxPointsPerCloud, 8)).astype(np.float32)
        self.normals_a = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)
        self.normals_b = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)


        self.reallocate_out(np.array([self.maxPointsPerCloud, self.maxPointsPerCloud, 0, 0]))

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_points_a)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.points_a.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_points_b)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.points_b.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_scan_lines)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.scan_lines.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_correspondence)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.correspondences.nbytes, self.correspondences, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_Hs)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.Hs.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_Bs)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.Bs.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_normals_a)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.normals_a.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_normals_b)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.normals_b.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_points_a)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo_points_b)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.ssbo_scan_lines)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ssbo_correspondence)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.ssbo_Hs)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.ssbo_Bs)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self.ssbo_normals_a)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self.ssbo_normals_b)

    def reallocate_out(self, lens_data):
        len_a = lens_data[0]
        len_b = lens_data[1]
        if len_b != self.prev_out_b:
            self.corr_out = np.zeros((len_b, 4)).astype(np.float32)
            self.hs_out = np.zeros((len_b, 8, 8)).astype(np.float32)
            self.bs_out = np.zeros((len_b, 8)).astype(np.float32)
            self.normals_out_b = np.zeros((len_b, 4)).astype(np.float32)

            self.prev_out_b = len_b
        if len_a != self.prev_out_a:
            self.normals_out_a = np.zeros((len_a, 4)).astype(np.float32)
            self.prev_out_a = len_a


    def freeBuffers(self):
        if self.ssbo_points_a is not None:
            glDeleteBuffers(1, [self.ssbo_points_a])
            self.ssbo_points_a = None

        if self.ssbo_points_b is not None:
            glDeleteBuffers(1, [self.ssbo_points_b])
            self.ssbo_points_b = None

        if self.ssbo_scan_lines is not None:
            glDeleteBuffers(1, [self.ssbo_scan_lines])
            self.ssbo_scan_lines = None

        if self.ssbo_correspondence is not None:
            glDeleteBuffers(1, [self.ssbo_correspondence])
            self.ssbo_correspondence = None

        if self.ssbo_Hs is not None:
            glDeleteBuffers(1, [self.ssbo_Hs])
            self.ssbo_Hs = None

        if self.ssbo_Bs is not None:
            glDeleteBuffers(1, [self.ssbo_Bs])
            self.ssbo_Bs = None

        if self.ssbo_normals_a is not None:
            glDeleteBuffers(1, [self.ssbo_normals_a])
            self.ssbo_normals_a = None

        if self.ssbo_normals_b is not None:
            glDeleteBuffers(1, [self.ssbo_normals_b])
            self.ssbo_normals_b = None

    def bufferSubdata(self, np_array, ssbo):
        if np_array.shape[1] == 3:
            np_array = self.pad_points(np_array)



        target_ssbo = None
        if ssbo == self.ssbo_points_a:
            target_ssbo = self.ssbo_points_a
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)

        if ssbo == self.ssbo_points_b:
            target_ssbo = self.ssbo_points_b
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)

        if ssbo == self.ssbo_scan_lines:
            target_ssbo = self.ssbo_scan_lines
            if np_array.dtype != np.uint32:
                np_array = np_array.astype(np.uint32)

        if ssbo == self.ssbo_correspondence:
            target_ssbo = self.ssbo_correspondence
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)

        if ssbo == self.ssbo_Hs:
            target_ssbo = self.ssbo_Hs
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)

        if ssbo == self.ssbo_Bs:
            target_ssbo = self.ssbo_Bs
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)

        if ssbo == self.ssbo_normals_a:
            target_ssbo = self.ssbo_normals_a
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)

        if ssbo == self.ssbo_normals_b:
            target_ssbo = self.ssbo_normals_b
            if np_array.dtype != np.float32:
                np_array = np_array.astype(np.float32)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, target_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, np_array.nbytes, np_array)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        return target_ssbo

    def getBufferFullData(self, ssbo_from):
        target_out = None
        target_size_bytes = None
        target_ssbo = None

        if ssbo_from == self.ssbo_Hs:
            hs_unit_size = self.float_size * 8 * 8

            target_out = self.Hs
            target_size_bytes = self.maxPointsPerCloud * hs_unit_size # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_Bs:
            bs_unit_size = self.float_size * 8


            target_out = self.Bs
            target_size_bytes = self.maxPointsPerCloud * bs_unit_size # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_correspondence:
            corr_unit_size = self.float_size * 4

            target_out = self.correspondences
            target_size_bytes = self.maxPointsPerCloud * corr_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_points_a:
            point_unit_size = self.float_size * 4

            target_out = self.points_a
            target_size_bytes = self.maxPointsPerCloud * point_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_points_b:
            point_unit_size = self.float_size * 4

            target_out = self.points_b
            target_size_bytes = self.maxPointsPerCloud * point_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_correspondence:
            corr_unit_size = self.float_size * 4

            target_out = self.correspondences
            target_size_bytes = self.maxPointsPerCloud * corr_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from


        if ssbo_from == self.ssbo_normals_a:
            norm_unit_size = self.float_size * 4

            target_out = self.normals_a
            target_size_bytes = self.maxPointsPerCloud * norm_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_normals_b:
            norm_unit_size = self.float_size * 4

            target_out = self.normals_b
            target_size_bytes = self.maxPointsPerCloud * norm_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from




        glBindBuffer(GL_SHADER_STORAGE_BUFFER, target_ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, target_size_bytes,
                           target_out.ctypes.data_as(c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def getBufferSubdata(self, ssbo_from):
        target_out = None
        target_size_bytes = None
        target_ssbo = None

        self.reallocate_out(self.lens_data)

        if ssbo_from == self.ssbo_Hs:
            hs_unit_size = self.float_size * 8 * 8

            target_out = self.hs_out
            target_size_bytes = self.lens_data[1] * hs_unit_size # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_Bs:
            bs_unit_size = self.float_size * 8


            target_out = self.bs_out
            target_size_bytes = self.lens_data[1] * bs_unit_size # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_correspondence:
            corr_unit_size = self.float_size * 4

            target_out = self.corr_out
            target_size_bytes = self.lens_data[1] * corr_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from


        if ssbo_from == self.ssbo_normals_a:
            norm_unit_size = self.float_size * 4

            target_out = self.normals_out_a
            target_size_bytes = self.lens_data[0] * norm_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        if ssbo_from == self.ssbo_normals_b:
            norm_unit_size = self.float_size * 4

            target_out = self.normals_out_b
            target_size_bytes = self.lens_data[1] * norm_unit_size  # lens_data1 > pts a len, lens2 > b len
            target_ssbo = ssbo_from

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, target_ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, target_size_bytes,
                           target_out.ctypes.data_as(c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        """
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_correspondence)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.correspondences.nbytes,
                           self.correspondences.ctypes.data_as(c_void_p))
       """


    def saveUniformInfo(self, points_a, points_b, origin, scan_lines, debug_mode):
        origin = np.resize(origin, (4,))
        self.origin = origin.astype(np.float32)


        self.debug_info = np.array([0, 0, 0, 0]).astype(np.float32)

        if debug_mode:
            self.debug_info[0] = 1.123

        self.lens_data = np.array([len(points_a), len(points_b), len(scan_lines), 0]).astype(np.uint32)

    def setCommonUniforms(self, shader_program):
        self.setUniform(shader_program, "origin", self.origin)

        self.setUniform(shader_program, "debug_info", self.debug_info)

        self.setUniform(shader_program, "lens_data", self.lens_data)

    def setUniform(self, shader_program, uniform_name, uniform_data):
        location = glGetUniformLocation(shader_program, uniform_name)

        if uniform_name == "lens_data":
            if uniform_data.shape != (4,):
                uniform_data = np.resize(uniform_data, (4,))
            uniform_data = uniform_data.astype(np.uint32)


            glUniform4ui(location, uniform_data[0], uniform_data[1], uniform_data[2], uniform_data[3])
            self.lens_data = uniform_data

        else:
            if uniform_data.shape != (4,):
                uniform_data = np.resize(uniform_data, (4,))
            uniform_data = uniform_data.astype(np.float32)

            glUniform4f(location, uniform_data[0], uniform_data[1], uniform_data[2], uniform_data[3])

    def setActiveProgram(self, shaderprogram):
        glUseProgram(shaderprogram)

    def dispatchCurrentProgramWait(self, threads_x):
        num_groups_x = int(np.ceil(threads_x / 64.0))
        num_groups_y = 1  # int(np.ceil(len(scan_lines) / 32.0))
        glDispatchCompute(num_groups_x, num_groups_y, 1)

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)

    def prepareLS(self, points_a, scan_lines, points_b, origin, debug_mode = False):
        self.setActiveProgram(self.least_squares_point)


        scan_lines = np.array(scan_lines).astype(np.uint32)
        padded = np.ones((scan_lines.shape[0], 4), dtype=np.uint32)
        padded[:, :2] = scan_lines.astype(np.uint32)
        scan_lines = padded

        self.bufferSubdata(points_a, self.ssbo_points_a)

        self.bufferSubdata(scan_lines, self.ssbo_scan_lines)



        self.saveUniformInfo(points_a, points_b, origin, scan_lines, debug_mode)

    def dispatchLS(self, points_b):

        self.bufferSubdata(points_b, self.ssbo_points_b)

        self.setCommonUniforms(self.least_squares_point)


        self.dispatchCurrentProgramWait(len(points_b))



        self.getBufferSubdata(self.ssbo_Hs)
        self.getBufferSubdata(self.ssbo_Bs)

        return self.hs_out, self.bs_out

    def padScanlines(self, scan_lines, points_a):
        scan_lines = np.array(scan_lines).astype(np.uint32)

        if scan_lines.shape != (len(points_a), 4):
            padded = np.ones((scan_lines.shape[0], 4), dtype=np.uint32)
            padded[:, :2] = scan_lines.astype(np.uint32)
            return padded

        return scan_lines

    def prepareDispatchNormals(self, points_a, scan_lines, origin):
        self.setActiveProgram(self.normal_shader)

        scan_lines = self.padScanlines(scan_lines, points_a)


        self.bufferSubdata(points_a, self.ssbo_points_a)

        self.bufferSubdata(scan_lines, self.ssbo_scan_lines)

        self.setUniform(self.normal_shader, "origin", origin)


        self.lens_data = np.array([len(points_a), len(points_a), len(scan_lines), 0])
        self.setUniform(self.normal_shader, "lens_data", self.lens_data)


        self.dispatchCurrentProgramWait(len(points_a))

        # self.getBufferSubdata(self.ssbo_normals_a)



    def preparePointPlane(self, points_a, scan_lines, points_b, origin, debug_mode = False):
        self.prepareDispatchNormals(points_a, scan_lines, origin)

        self.setActiveProgram(self.point_plane_shader)
        scan_lines = np.array(scan_lines).astype(np.uint32)
        padded = np.ones((scan_lines.shape[0], 4), dtype=np.uint32)
        padded[:, :2] = scan_lines.astype(np.uint32)
        scan_lines = padded

        self.bufferSubdata(points_a, self.ssbo_points_a)

        self.bufferSubdata(scan_lines, self.ssbo_scan_lines)

        self.saveUniformInfo(points_a, points_b, origin, scan_lines, debug_mode)

    def dispatchPointPlane(self, points_b):
        self.bufferSubdata(points_b, self.ssbo_points_b)

        self.setCommonUniforms(self.point_plane_shader)

        self.dispatchCurrentProgramWait(len(points_b))

        self.getBufferSubdata(self.ssbo_Hs)
        self.getBufferSubdata(self.ssbo_Bs)

        return self.hs_out, self.bs_out

    def prepareNNS(self, points_a, scan_lines, points_b, origin):
        """self.program = self.create_shader_program(self.glslFile("NearestNeighbourScanV2.glsl"))"""
        glUseProgram(self.nearest_neighbour)


        self.ssbo_points_a = glGenBuffers(1)
        self.ssbo_points_b = glGenBuffers(1)
        self.ssbo_scan_lines = glGenBuffers(1)
        self.ssbo_correspondence = glGenBuffers(1)

        self.correspondences = np.zeros((len(points_b), 4)).astype(np.float32)

        points_a = self.pad_points(points_a)
        points_b = self.pad_points(points_b)


        scan_lines = np.array(scan_lines).astype(np.uint32)
        padded = np.ones((scan_lines.shape[0], 4), dtype=np.uint32)
        padded[:, :2] = scan_lines.astype(np.uint32)
        scan_lines = padded

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_points_a)
        glBufferData(GL_SHADER_STORAGE_BUFFER, points_a.nbytes, points_a, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_points_a)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_points_b)
        glBufferData(GL_SHADER_STORAGE_BUFFER, points_b.nbytes, points_b, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo_points_b)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_scan_lines)
        glBufferData(GL_SHADER_STORAGE_BUFFER, scan_lines.nbytes, scan_lines, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.ssbo_scan_lines)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_correspondence)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.correspondences.nbytes, self.correspondences, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ssbo_correspondence)

        self.origin_location = glGetUniformLocation(self.program, "origin")
        self.origin = origin.astype(np.float32)
        glUniform4fv(self.origin_location, 1, self.origin)


    def dispatchNNS(self, points_b):



        points_b = self.pad_points(points_b)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_points_b)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, points_b.nbytes, points_b)

        glUniform4fv(self.origin_location, 1, self.origin)


        num_groups_x = int(np.ceil(len(points_b) / 64.0))
        num_groups_y = 1  # int(np.ceil(len(scan_lines) / 32.0))
        glDispatchCompute(num_groups_x, num_groups_y, 1)

        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_correspondence)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.correspondences.nbytes,
                           self.correspondences.ctypes.data_as(c_void_p))


        # print(str(time.time() - st))

        return self.correspondences



    def pad_points(self, points):
        if points.shape[1] == 4:
            return points.astype(np.float32)
        padded = np.zeros((points.shape[0], 4), dtype=np.float32)
        padded[:, :3] = points.astype(np.float32)
        return padded

    def deleteProgram(self, shader_program):
        glDeleteProgram(shader_program)

    def cleanup(self):
        print("Compute cleanup")
        self.freeBuffers()

        if self.least_squares_point is not None:
            self.deleteProgram(self.least_squares_point)
            self.least_squares_point = None

        if self.nearest_neighbour is not None:
            self.deleteProgram(self.nearest_neighbour)
            self.nearest_neighbour = None

        if self.normal_shader is not None:
            self.deleteProgram(self.normal_shader)
            self.normal_shader = None



        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None

        # glfw.terminate()

    def create_shader_program(self, shader_code):
        program = glCreateProgram()
        shader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(shader, shader_code)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader).decode())

        glAttachShader(program, shader)
        glLinkProgram(program)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program).decode())

        glDetachShader(program, shader)  # Detach shader
        glDeleteShader(shader)  # Delete shader after linking

        return program






    """def dispatchNeighbourMorton(self, all_sorted_mortons, all_sorted_points, b_mortons, b_points):
        if self.program is None:
            raise Exception("Program not set")

        all_sorted_points = self.pad_points(all_sorted_points)
        b_points = self.pad_points(b_points)

        ssbo_sorted_mortons_a = glGenBuffers(1)
        ssbo_sorted_points_a = glGenBuffers(1)
        ssbo_mortons_b = glGenBuffers(1)
        ssbo_points_b = glGenBuffers(1)

        ssbo_correspondence = glGenBuffers(1)

        correspondences = np.zeros_like(b_mortons).astype(np.uint32)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_sorted_mortons_a)
        glBufferData(GL_SHADER_STORAGE_BUFFER, all_sorted_mortons.nbytes, all_sorted_mortons, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_sorted_mortons_a)  # Binding index 0 for mortons

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_sorted_points_a)
        glBufferData(GL_SHADER_STORAGE_BUFFER, all_sorted_points.nbytes, all_sorted_points, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_sorted_points_a)  # Binding index 1 for points

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_mortons_b)
        glBufferData(GL_SHADER_STORAGE_BUFFER, b_mortons.nbytes, b_mortons, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_mortons_b)  # Binding index 0 for mortons

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_points_b)
        glBufferData(GL_SHADER_STORAGE_BUFFER, b_points.nbytes, b_points, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssbo_points_b)  # Binding index 1 for points

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_correspondence)
        glBufferData(GL_SHADER_STORAGE_BUFFER, correspondences.nbytes, correspondences, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ssbo_correspondence)  # Binding index 1 for points



        glUseProgram(self.program)
        num_groups = int(np.ceil(len(b_points) / 64.0))
        glDispatchCompute(num_groups, 1, 1)

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glFinish()

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_correspondence)

        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, correspondences.nbytes, correspondences.ctypes.data_as(c_void_p))

        glDeleteBuffers(1, [ssbo_sorted_mortons_a])
        glDeleteBuffers(1, [ssbo_sorted_points_a])
        glDeleteBuffers(1, [ssbo_points_b])
        glDeleteBuffers(1, [ssbo_mortons_b])
        glDeleteBuffers(1, [ssbo_correspondence])


        return correspondences
    """