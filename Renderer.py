import math

import glfw
from OpenGL.GL import *
import numpy as np
from dash.exceptions import InvalidIndexException

from GpuMemoryManager import GpuMemoryManager
from InputListener import InputListener
from ShaderPrograms import TEST_VERTEX_SHADER, TEST_FRAGMENT_SHADER, TEST_GEOMETRY_SHADER, DEFAULT_VERTEX_SHADER, \
    DEFAULT_FRAGMENT_SHADER, CUBE_GEOMETRY_SHADER, CUBE_FRAGMENT_SHADER
from pyglm import glm

import time
import threading

class Renderer:

    TARGET_FPS = 60
    frame_time = 1.0 / TARGET_FPS

    mousens = 0.1
    speed = 0.1

    float_size = 4

    point_size = 3 * float_size + 3 * float_size # pos, color

    defaultColor = np.array([64, 224, 208]) / 255.0

    def __init__(self, voxelSize, maxNumberOfPoints=10000000):
        self.buffer_capacity = 1024
        self.inited = False

        self.voxelSize = voxelSize

        self.position = glm.vec3(0, 0, 0)
        self.front = glm.vec3(0, 0, 0)
        self.up = glm.vec3(0, 1, 0)

        self.yaw = -90.0
        self.pitch = 0.0

        self.width = 1600
        self.height = 800

        self.view = None
        self.projection = None
        self.model = None

        self.shader_program = None
        self.lineShader = None

        self.window = None
        self.inputListener = None

        self.VAO = None
        self.VBO = None

        self.lineVAO = None
        self.lineVBO = None

        self.vbo_update_needed = False


        self.lineVertices = None



        self.maxNumPts = maxNumberOfPoints

        self.inputListener = InputListener()
        self.inputListener.listen()

        self.MemoryManager = GpuMemoryManager(maxNumberOfPoints=self.maxNumPts)

        self.lock = threading.Lock()

        self.rendering_thread = threading.Thread(target=self.startRender, daemon=True)
        self.rendering_thread.start()

        self.maxlinecount = 0
        self.line_update_needed = False

        while not self.getInited():
            a = 0

    def getRenderingThread(self):
        return self.rendering_thread

    def setupOpenGL(self):
        if not glfw.init():
            print("Error creating glfw context")
            return

        self.window = glfw.create_window(self.width, self.height, "opengl", None, None)
        if not self.window:
            print("Error creating window")
            glfw.terminate()

        glfw.make_context_current(self.window)

        self.inited = True

        # self.shader_program = self.createShaderProgram(TEST_VERTEX_SHADER, CUBE_GEOMETRY_SHADER, CUBE_FRAGMENT_SHADER)
        self.shader_program = self.createShaderProgram(TEST_VERTEX_SHADER, TEST_GEOMETRY_SHADER, TEST_FRAGMENT_SHADER)
        self.lineShader = self.createLineShaderProgram(DEFAULT_VERTEX_SHADER, DEFAULT_FRAGMENT_SHADER)
        self.setupVAO()
        self.setupLineVAO()

        max_width = glGetFloatv(GL_LINE_WIDTH_RANGE)
        print("Supported line width range:", max_width)

    def getInited(self):
        return self.inited

    def startRender(self):

        self.setupOpenGL()






        self.model = glm.mat4(1.0)  # Identity matrix (no transformation)
        self.updateCamera()
        self.projection = glm.perspective(glm.radians(45.0), self.width / self.height, 0.1,
                                          100.0)  # Perspective projection



        glEnable(GL_DEPTH_TEST)


        frame_counter = 0
        while True:
            render_start = time.time()

            glfw.poll_events()
            if glfw.window_should_close(self.window) or self.inputListener.esc:
                glfw.set_window_should_close(self.window, True)
                break


            # Clear the screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUseProgram(self.shader_program)
            glBindVertexArray(self.VAO)

            self.updateCamera()
            self.setMVP(self.shader_program)

            glUniform1f(glGetUniformLocation(self.shader_program, "voxelSize"), self.voxelSize)

            with self.lock:
                self.updateVBO()

            glDrawArrays(GL_POINTS, 0, self.MemoryManager.getMaxPointIndex())
            glBindVertexArray(0)



            if self.lineVertices is not None:
                glUseProgram(self.lineShader)
                self.setMVP(self.lineShader)

                glBindVertexArray(self.lineVAO)
                self.updateLineVBO()
                glDrawArrays(GL_LINES, 0, len(self.lineVertices))
                glBindVertexArray(0)  # Unbind VAO


            glfw.swap_buffers(self.window)

            render_time = time.time() - render_start

            sleep_time = self.frame_time - render_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            frame_counter += 1

            if frame_counter % 60 == 0:
                # print(f"Frame: {frame_counter}, FPS: {1.0/sleep_time:.2f}, Points: {self.MemoryManager.getMaxPointIndex()/1000000:.1f}M")  # Display FPS
                a = 0

        print("cleanup")
        self.cleanup()

    def setMVP(self, shader_program):
        modelLoc = glGetUniformLocation(shader_program, "model")
        viewLoc = glGetUniformLocation(shader_program, "view")
        projLoc = glGetUniformLocation(shader_program, "projection")

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm.value_ptr(self.model))
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm.value_ptr(self.view))
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm.value_ptr(self.projection))

    def handleColors(self, np_points, colors):
        if colors is not None:
            colors = np.array(colors)

        if colors is None or colors.shape == (3,):
            if colors is None:
                colors = np.tile(np.array([1, 1, 1]), (len(np_points), 1)).astype(np.float32)
            else:
                colors = np.tile(colors, (len(np_points), 1)).astype(np.float32)


        if len(colors) != len(np_points):
            print("Error! Colors and points are not in pair")

        if len(colors.shape) == 1:
            if colors[0] > 1:
                colors = colors.astype(np.float32) / 255.0
                colors = np.tile(colors, (3, 1)).T
        else:
            if colors[0][0] > 1 or colors[0][1] > 1 or colors[0][2] > 1:
                colors = colors / 255.0
                colors = colors.astype(np.float32)

        return colors
    def addPoints(self, np_points, colors=None):
        colors = self.handleColors(np_points, colors)

        with self.lock:
            self.vbo_update_needed = True

            points = np.hstack((np_points, colors))
            return self.MemoryManager.addPoints(points)

    def freePoints(self, pointer):
        with self.lock:
            self.MemoryManager.freeSpace(pointer)


    def setLines(self, np_line_points, color=None):
        if len(np_line_points) % 2 != 0:
            raise InvalidIndexException("Invalid line points provided")


        line_points = np.array(np_line_points).astype(np.float32)

        colors = self.handleColors(np_line_points, color)

        self.lineVertices = np.hstack([line_points, colors])


    def createShaderProgram(self, V_shader, G_shader, F_shader): #V_ertex shader, F_ragment shader
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, V_shader)
        glCompileShader(vertex_shader)

        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER)
        glShaderSource(geometry_shader, G_shader)
        glCompileShader(geometry_shader)

        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, F_shader)
        glCompileShader(fragment_shader)

        # Create shader program
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, geometry_shader)
        glAttachShader(shader_program, fragment_shader)

        glBindAttribLocation(shader_program, 0, "inPosition")
        glBindAttribLocation(shader_program, 1, "inColor")

        glLinkProgram(shader_program)

        # Clean up shaders (no longer needed after linking)
        glDeleteShader(vertex_shader)
        glDeleteShader(geometry_shader)
        glDeleteShader(fragment_shader)

        return shader_program

    def createLineShaderProgram(self, V_shader, F_shader): #V_ertex shader, F_ragment shader
        # Compile vertex shader
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex_shader, V_shader)
        glCompileShader(vertex_shader)

        # Compile fragment shader
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fragment_shader, F_shader)
        glCompileShader(fragment_shader)

        # Create shader program
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)

        glBindAttribLocation(shader_program, 0, "inPosition")
        glBindAttribLocation(shader_program, 1, "inColor")

        glLinkProgram(shader_program)

        # Clean up shaders (no longer needed after linking)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return shader_program

    def setupVAO(self):
        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

    def setupLineVAO(self):
        self.lineVAO = glGenVertexArrays(1)
        glBindVertexArray(self.lineVAO)

    def updateVBO(self):
        if self.VBO is None:
            self.VBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferData(GL_ARRAY_BUFFER, self.MemoryManager.maxNumPoints * self.point_size, self.MemoryManager.points, GL_DYNAMIC_DRAW)

        if self.vbo_update_needed:
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.MemoryManager.getMaxPointIndex() * self.point_size, self.MemoryManager.points)


            self.vbo_update_needed = False

            # Position attribute (location 0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * self.float_size, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            # Color attribute (location 1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * self.float_size, ctypes.c_void_p(3 * self.float_size))
            glEnableVertexAttribArray(1)



    def updateLineVBO(self):

        if self.lineVBO is None:  # First-time initialization
            self.lineVBO = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.lineVBO)
            self.buffer_capacity = max(self.lineVertices.nbytes, 1024)  # Start with at least 1KB
            glBufferData(GL_ARRAY_BUFFER, self.buffer_capacity, None, GL_DYNAMIC_DRAW)  # Allocate but don't upload
        else:
            glBindBuffer(GL_ARRAY_BUFFER, self.lineVBO)

        # Check if buffer size needs to be increased
        current_size = glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)

        if self.lineVertices.nbytes > current_size:
            self.buffer_capacity = max(self.buffer_capacity * 2, self.lineVertices.nbytes)  # Double size or fit data
            glBufferData(GL_ARRAY_BUFFER, self.buffer_capacity, None, GL_DYNAMIC_DRAW)  # Reallocate with new size
            print(f"Resized line buffer to {self.buffer_capacity} bytes")  # Debug print
        else:
            # Simply update the existing buffer
            glBufferSubData(GL_ARRAY_BUFFER, 0, self.lineVertices.nbytes, self.lineVertices)

        # Position attribute (location 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * self.float_size, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Color attribute (location 1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * self.float_size, ctypes.c_void_p(3 * self.float_size))
        glEnableVertexAttribArray(1)




    def updateCamera(self):
        if self.inputListener.getLeft():
            delta = self.inputListener.getMouseDelta()
            x = float(delta[0])
            y = -float(delta[1])

            self.yaw += x * self.mousens
            self.pitch += y * self.mousens

        if self.pitch > 89.0:
            self.pitch = 89.0

        if self.pitch < -89.0:
            self.pitch = -89.0

        x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        y = math.sin(glm.radians(self.pitch))
        z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front = glm.normalize(glm.vec3(x,y,z))

        # handle camera movement

        if self.inputListener.left:
            self.position -= glm.normalize(glm.cross(front, self.up)) * self.speed
        if self.inputListener.right:  # captures depth xd
            self.position += glm.normalize(glm.cross(front, self.up)) * self.speed
        if self.inputListener.forward:
            self.position += front * self.speed
        if self.inputListener.backward:
            self.position -= front * self.speed

        if self.inputListener.up:
            self.position += glm.vec3(0, 1, 0) * self.speed

        if self.inputListener.down:
            self.position += glm.vec3(0, -1, 0) * self.speed

        # handle mouse movement

        self.view = glm.lookAt(self.position, self.position + front,
                      self.up)


    def cleanup(self):
        # Unbind VAO & VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        # Delete VBO & VAO
        if hasattr(self, 'VBO') and self.VBO is not None:
            glDeleteBuffers(1, [self.VBO])
            self.VBO = None

        if hasattr(self, 'lineVBO') and self.lineVBO is not None:
            glDeleteBuffers(1, [self.lineVBO])
            self.lineVBO = None

        if hasattr(self, 'VAO') and self.VAO is not None:
            glDeleteVertexArrays(1, [self.VAO])
            self.VAO = None

        if hasattr(self, 'lineVAO') and self.lineVAO is not None:
            glDeleteVertexArrays(1, [self.lineVAO])
            self.lineVAO = None

        # Delete Shader Program
        if hasattr(self, 'shader_program') and self.shader_program is not None:
            glDeleteProgram(self.shader_program)
            self.shader_program = None

        if hasattr(self, 'lineShader') and self.lineShader is not None:
            glDeleteProgram(self.lineShader)
            self.lineShader = None

        # Destroy Window & Terminate GLFW
        glfw.destroy_window(self.window)
        glfw.terminate()

    @staticmethod
    def unitCube():
        return [
            (0.5, 0.5, 0.5),
            (0.5, 0.5, -0.5),
            (0.5, -0.5, 0.5),
            (0.5, -0.5, -0.5),
            (-0.5, 0.5, 0.5),
            (-0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (-0.5, -0.5, -0.5),
        ]

    @staticmethod
    def unitCubeEdges(offset = np.array([0, 0, 0])):
        vertices = Renderer.unitCube()
        edges = [
            vertices[0], vertices[1], vertices[0], vertices[2], vertices[0], vertices[4],
            vertices[1], vertices[3], vertices[1], vertices[5],
            vertices[2], vertices[3], vertices[2], vertices[6],
            vertices[3], vertices[7],
            vertices[4], vertices[5], vertices[4], vertices[6],
            vertices[5], vertices[7],
            vertices[6], vertices[7]
        ]
        return np.array(edges) + offset