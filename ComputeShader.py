import time
from ctypes import c_void_p

import OpenGL.error
from OpenGL.GL import *
import glfw

import numpy as np


class ComputeShader:

    def __init__(self, maxPointsPerCloud=200000, maxScans=100):




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

        self.bigint = 9999999
        self.float_size = 4
        self.max_points_per_voxel = 4096

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

        self.ssbo_voxel_index = None
        self.ssbo_voxel_data = None
        self.ssbo_unknown_points = None

        self.ssbo_debug_buffer = None

        self.ssbo_voxel_stage = None
        self.ssbo_single_counter = None
        self.ssbo_voxel_stat = None

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

        self.voxel_stage_out = None

        self.least_squares_point = None
        self.nearest_neighbour = None
        self.normal_shader = None
        self.point_plane_shader = None
        self.voxel_shader = None
        self.voxel_stage = None

        self.main_shadercode_regex = "void main(){}"

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
        self.nearest_neighbour = self.create_shader_program(self.glslFile("NearestNeighbourScan.glsl"))
        self.normal_shader = self.create_shader_program(self.glslFile("NormalShader.glsl"))
        self.point_plane_shader = self.create_shader_program(self.glslFile("LeastSquaresPlane.glsl"))
        self.voxel_shader = self.create_shader_program(self.glslFile("VoxelShader.glsl"), extend=False)
        self.voxel_stage = self.create_shader_program(self.glslFile("VoxelStage.glsl"), extend=False)

    def prepareBuffers(self):
        self.ssbo_points_a = glGenBuffers(1)
        self.ssbo_points_b = glGenBuffers(1)
        self.ssbo_scan_lines = glGenBuffers(1)
        self.ssbo_correspondence = glGenBuffers(1)
        self.ssbo_Hs = glGenBuffers(1)
        self.ssbo_Bs = glGenBuffers(1)
        self.ssbo_normals_a = glGenBuffers(1)
        self.ssbo_normals_b = glGenBuffers(1)

        self.ssbo_voxel_data = glGenBuffers(1)
        self.ssbo_voxel_index = glGenBuffers(1)
        self.ssbo_unknown_points = glGenBuffers(1)

        self.ssbo_debug_buffer = glGenBuffers(1)

        self.ssbo_single_counter = glGenBuffers(1)
        self.ssbo_voxel_stat = glGenBuffers(1)
        self.ssbo_voxel_stage = glGenBuffers(1)



        self.points_a = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)
        self.points_b = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)
        self.scan_lines = np.zeros((self.maxScans, 4)).astype(np.uint32)
        self.correspondences = np.zeros((self.maxPointsPerCloud, 4)).astype(np.float32)
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

        # voxel buffers are more dynamic, have to allocate on the spot

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo_points_a)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo_points_b)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.ssbo_scan_lines)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ssbo_correspondence)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.ssbo_Hs)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.ssbo_Bs)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self.ssbo_normals_a)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self.ssbo_normals_b)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, self.ssbo_voxel_index)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, self.ssbo_voxel_data)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, self.ssbo_unknown_points)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, self.ssbo_debug_buffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, self.ssbo_voxel_stage)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, self.ssbo_single_counter)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, self.ssbo_voxel_stat)


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

        if self.ssbo_voxel_index is not None:
            glDeleteBuffers(1, [self.ssbo_voxel_index])
            self.ssbo_voxel_index = None

        if self.ssbo_voxel_data is not None:
            glDeleteBuffers(1, [self.ssbo_voxel_data])
            self.ssbo_voxel_data = None

        if self.ssbo_unknown_points is not None:
            glDeleteBuffers(1, [self.ssbo_unknown_points])
            self.ssbo_unknown_points = None

        if self.ssbo_debug_buffer is not None:
            glDeleteBuffers(1, [self.ssbo_debug_buffer])
            self.ssbo_debug_buffer = None

        if self.ssbo_voxel_stage is not None:
            glDeleteBuffers(1, [self.ssbo_voxel_stage])
            self.ssbo_voxel_stage = None

        if self.ssbo_single_counter is not None:
            glDeleteBuffers(1, [self.ssbo_single_counter])
            self.ssbo_single_counter = None

        if self.ssbo_voxel_stat is not None:
            glDeleteBuffers(1, [self.ssbo_voxel_stat])
            self.ssbo_voxel_stat = None

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
            # print(len(np_array))

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

    def setInitialScanGuess(self):
        self.correspondences[:, 3] = -1.0

    def prepareLS(self, points_a, scan_lines, points_b, origin, debug_mode = False):
        self.setActiveProgram(self.least_squares_point)


        scan_lines = np.array(scan_lines).astype(np.uint32)
        padded = np.ones((scan_lines.shape[0], 4), dtype=np.uint32)
        padded[:, :2] = scan_lines.astype(np.uint32)
        scan_lines = padded

        self.bufferSubdata(points_a, self.ssbo_points_a)

        self.setInitialScanGuess()
        self.bufferSubdata(self.correspondences, self.ssbo_correspondence)


        try: # this is a bug, scan for kitti, no more than 64 scan lines
            self.bufferSubdata(scan_lines, self.ssbo_scan_lines)
        except OpenGL.error.GLError as e:
            a = scan_lines

            print("ERRORZ")
            raise e


        self.saveUniformInfo(points_a, points_b, origin, scan_lines, debug_mode)

    def dispatchLS(self, points_b):

        self.bufferSubdata(points_b, self.ssbo_points_b)

        self.setCommonUniforms(self.least_squares_point)


        self.dispatchCurrentProgramWait(len(points_b))



        self.getBufferSubdata(self.ssbo_Hs)
        self.getBufferSubdata(self.ssbo_Bs)

        return self.hs_out, self.bs_out

    def padScanlines(self, scan_lines):
        scan_lines = np.array(scan_lines).astype(np.uint32)

        if scan_lines.shape != (self.maxScans, 4):
            padded = np.ones((self.maxScans, 4), dtype=np.uint32)
            padded[:scan_lines.shape[0], :scan_lines.shape[1]] = scan_lines.astype(np.uint32)
            return padded

        return scan_lines

    def prepareDispatchNormals(self, points_a, scan_lines, origin):
        self.setActiveProgram(self.normal_shader)

        scan_lines = self.padScanlines(scan_lines)


        self.bufferSubdata(points_a, self.ssbo_points_a)

        try:
            self.bufferSubdata(scan_lines, self.ssbo_scan_lines)
        except OpenGL.error.GLError as e:
            a = scan_lines

            print("ERRORZ: scan line count:" + str(len(scan_lines)))
            raise e

        self.setUniform(self.normal_shader, "origin", origin)


        self.lens_data = np.array([len(points_a), len(points_a), len(scan_lines), 0])
        self.setUniform(self.normal_shader, "lens_data", self.lens_data)


        self.dispatchCurrentProgramWait(len(points_a))

        self.getBufferSubdata(self.ssbo_normals_a)



    def preparePointPlane(self, points_a, scan_lines, points_b, origin, debug_mode = False):
        self.prepareDispatchNormals(points_a, scan_lines, origin)

        self.setActiveProgram(self.point_plane_shader)

        scan_lines = self.padScanlines(scan_lines)

        self.bufferSubdata(points_a, self.ssbo_points_a)

        self.bufferSubdata(scan_lines, self.ssbo_scan_lines)

        self.setInitialScanGuess()
        self.bufferSubdata(self.correspondences, self.ssbo_correspondence)

        self.saveUniformInfo(points_a, points_b, origin, scan_lines, debug_mode)

    def dispatchPointPlane(self, points_b):
        self.bufferSubdata(points_b, self.ssbo_points_b)

        self.setCommonUniforms(self.point_plane_shader)

        self.dispatchCurrentProgramWait(len(points_b))

        self.getBufferSubdata(self.ssbo_Hs)
        self.getBufferSubdata(self.ssbo_Bs)

        return self.hs_out, self.bs_out

    def prepareNNS(self, points_a, scan_lines, points_b, origin):
        self.setActiveProgram(self.nearest_neighbour)

        scan_lines = self.padScanlines(scan_lines)

        self.bufferSubdata(points_a, self.ssbo_points_a)

        self.bufferSubdata(scan_lines, self.ssbo_scan_lines)

        self.setInitialScanGuess()
        self.bufferSubdata(self.correspondences, self.ssbo_correspondence)

        self.saveUniformInfo(points_a, points_b, origin, scan_lines, False)


    def dispatchNNS(self, points_b):
        self.bufferSubdata(points_b, self.ssbo_points_b)

        self.setCommonUniforms(self.nearest_neighbour)

        self.dispatchCurrentProgramWait(len(points_b))

        self.getBufferSubdata(self.ssbo_correspondence)

        corr_indexes = self.corr_out[:, 0].astype(np.uint32)
        dists = self.corr_out[:, 1].astype(np.float32)

        return corr_indexes, dists

    def prepareDispatchVoxelStager(self, stored_voxel_num, max_points, voxel_statistics, stage_everything=False):
        self.setActiveProgram(self.voxel_stage)

        max_staging_area = 512
        if self.voxel_stage_out is None: # allocate voxel_stage once
            self.voxel_stage_out = np.empty((max_staging_area, max_points)).astype(np.int32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_stage)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.voxel_stage_out.nbytes, None, GL_DYNAMIC_DRAW) # None data ptr
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_stat)
        glBufferData(GL_SHADER_STORAGE_BUFFER, voxel_statistics.nbytes, voxel_statistics, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        single_counter_buffer = np.zeros((1, 16)).astype(np.int32)
        single_counter_buffer[0][3] = stored_voxel_num
        single_counter_buffer[0][2] = max_points
        single_counter_buffer[0][1] = max_staging_area

        if stage_everything:
            single_counter_buffer[0][4] = 1

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_single_counter)
        glBufferData(GL_SHADER_STORAGE_BUFFER, single_counter_buffer.nbytes, single_counter_buffer, GL_DYNAMIC_DRAW)  # None data ptr
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)



        self.dispatchCurrentProgramWait(stored_voxel_num)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_single_counter)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, single_counter_buffer.nbytes,
                           single_counter_buffer.ctypes.data_as(c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_stat)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, voxel_statistics.nbytes,
                           voxel_statistics.ctypes.data_as(c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        probs = voxel_statistics[voxel_statistics[:, 0] > 0.9]
        if len(probs) > 0:
            a = 0

        if single_counter_buffer[0][0] > 0:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_stage)
            new_data_len = single_counter_buffer[0][0] * max_points * 4

            # !!!!
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.voxel_stage_out.nbytes,
                               self.voxel_stage_out.ctypes.data_as(c_void_p)) # buffering into a bigger than needed np array!!!
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            return voxel_statistics, self.voxel_stage_out[:single_counter_buffer[0][0]]

        return voxel_statistics, []


    def prepareDispatchVoxelizer(self, np_points, voxel_index, voxel_data, voxel_size, stored_voxel_num, begin_index, max_points_to_store, realloc_needed, prev_stored_voxels, debug=False):
        self.setActiveProgram(self.voxel_shader)

        st = time.time()
        unknown_len = 1 + len(np_points)
        modd = 16 - unknown_len % 16
        unknown_len = unknown_len + modd

        unknown_points = np.zeros(unknown_len).astype(np.int32)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_index)
        glBufferData(GL_SHADER_STORAGE_BUFFER, voxel_index.nbytes, voxel_index, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        if realloc_needed:
            print("realloced voxdat")
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_data)
            glBufferData(GL_SHADER_STORAGE_BUFFER, voxel_data.nbytes, voxel_data, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        else:
            print("subbuffered voxdat")
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_data)
            byte_offset = prev_stored_voxels * max_points_to_store * 4
            new_data_len = (stored_voxel_num - prev_stored_voxels) * max_points_to_store * 4

            new_data = voxel_data[prev_stored_voxels:stored_voxel_num]

            # test sizes
            """a = new_data.nbytes
            old_data = voxel_data[:prev_stored_voxels]
            b = old_data.nbytes"""

            data_ptr = new_data.ctypes.data_as(ctypes.POINTER(ctypes.c_char))

            glBufferSubData(GL_SHADER_STORAGE_BUFFER, byte_offset, new_data_len, data_ptr)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_unknown_points)
        glBufferData(GL_SHADER_STORAGE_BUFFER, unknown_points.nbytes, unknown_points, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        if debug:
            debug_data = np.zeros_like(np_points)
            debug_data = self.pad_points(debug_data)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_debug_buffer)
            glBufferData(GL_SHADER_STORAGE_BUFFER, debug_data.nbytes, debug_data, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


        self.bufferSubdata(np_points, self.ssbo_points_a)

        location = glGetUniformLocation(self.voxel_shader, "voxel_lens_data")
        lens_data = np.array([len(np_points), stored_voxel_num, begin_index, max_points_to_store]).astype(np.uint32)
        glUniform4ui(location, lens_data[0], lens_data[1], lens_data[2], lens_data[3])

        location = glGetUniformLocation(self.voxel_shader, "voxel_size")
        glUniform1f(location, voxel_size[0])
        # print("prep time: " + str(time.time()-st))


        st = time.time()
        self.dispatchCurrentProgramWait(len(np_points))
        # print(str(time.time()-st))


        st = time.time()
        # out_vdat = np.zeros_like(voxel_data).astype(np.int32)


        """glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_data)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, out_vdat.nbytes,
                           out_vdat.ctypes.data_as(c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)"""

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_unknown_points)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, unknown_points.nbytes,
                           unknown_points.ctypes.data_as(c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        if debug:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_debug_buffer)
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, debug_data.nbytes,
                               debug_data.ctypes.data_as(c_void_p))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        if debug:
            return unknown_points, debug_data
        else:
            return unknown_points, None

    def getFullVoxelData(self, prev_stored_voxels, max_points, voxelizer_voxel_data):
        temp_data_buffer = np.empty((prev_stored_voxels, max_points)).astype(np.int32)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo_voxel_data)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, temp_data_buffer.nbytes,
                           temp_data_buffer.ctypes.data_as(c_void_p))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        voxelizer_voxel_data[:prev_stored_voxels] = temp_data_buffer

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

        if self.point_plane_shader is not None:
            self.deleteProgram(self.point_plane_shader)
            self.point_plane_shader = None

        if self.voxel_shader is not None:
            self.deleteProgram(self.voxel_shader)
            self.voxel_shader = None

        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None

        # glfw.terminate()

    def extend_functional_code(self, shader_code):
        base_code = self.glslFile("FunctionalBase.glsl")

        return base_code.replace(self.main_shadercode_regex, shader_code)

    def create_shader_program(self, shader_code, extend=True):
        # test if base compiles
        shader = glCreateShader(GL_COMPUTE_SHADER)

        if extend:
            base_code = self.glslFile("FunctionalBase.glsl")
            glShaderSource(shader, base_code)
            glCompileShader(shader)
            if not glGetShaderiv(shader, GL_COMPILE_STATUS):
                print("Unable to compile base shader")
                raise RuntimeError(glGetShaderInfoLog(shader).decode())




            shader_code = self.extend_functional_code(shader_code=shader_code)


        program = glCreateProgram()


        glShaderSource(shader, shader_code)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            print("Unable to compile complimentary shader")
            raise RuntimeError(glGetShaderInfoLog(shader).decode())

        glAttachShader(program, shader)
        glLinkProgram(program)

        if not glGetProgramiv(program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(program).decode())

        glDetachShader(program, shader)  # Detach shader
        glDeleteShader(shader)  # Delete shader after linking

        return program
