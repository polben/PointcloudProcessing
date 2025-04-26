import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ComputeShader import ComputeShader
from EnvironmentConstructor import EnvironmentConstructor
from LidarDataReader import LidarDataReader
from OxtsDataReader import OxtsDataReader
from PointcloudIcpContainer import PointcloudIcpContainer
from Renderer import Renderer
from PointcloudHandler import PointcloudHandler


class UI:

    def __init__(self, lauch=True):
        self.prev_pointclouds = None
        self.button_stop_build = None
        self.stop_build = False
        self.prev_kitti = None
        self.point_counter_var = None
        self.root = None
        self.title = "Pointcloud Visualizer"
        self.w = 400
        self.h = 520
        self.x = 10
        self.y = 10
        self.resolution = f"{self.w}x{self.h}+{self.x}+{self.y}"
        self.testing = not lauch

        self.button_load = None
        self.button_save = None
        self.button_calib = None
        self.button_next = None
        self.button_prev = None
        self.label_frame_counter = None
        self.label_color_mode = None
        self.radio_color = None
        self.radio_intensity = None

        self.button_mode_switch = None
        self.button_build = None

        self.label_env_mode = None
        self.radio_imu = None
        self.radio_icp = None
        self.radio_filter = None
        self.label_main_text = None

        self.info_frame = None
        self.frame_counter = None

        self.color_mode = None

        self.imu_mode = None
        self.icp_mode = None
        self.filter_mode = None

        self.text_output = None


        self.frame_counter_var = None

        self.renderer = None
        self.compute = None
        self.environment = None
        self.oxts = None
        self.lidar = None
        self.alignment = None
        self.icp = None
        self.pointcloudhandler = None

        self.renderingThread = None
        self.calibration = None
        self.calibrated = False

        self.RENDER_FRAME = 0
        self.RENDER_BUILD = 1
        self.RENDER_POINTCLOUDS = 2

        # variables
        self.last_rendered_pointcloud = None
        self.current_lidar_frame = 0
        self.current_pointcloud_frame = 0



        self.loadedKitti = False
        self.loadedPointclouds = False



        self.rendering_mode = self.RENDER_FRAME

        if lauch:
            self.intializeComponents()

        self.createUiLayout()








    def choseFolder(self, prompt):
        path = filedialog.askdirectory(title=prompt)

        if path == '':
            self.appendConsole("Data load cancelled!")
            return None

        return path

    def promptReadPointcloudsFolder(self):
        path = self.choseFolder("Please select a folder containing .ply files")
        if path is None:
            return
        if self.prev_pointclouds == path:
            self.appendConsole("Previous pointclouds loaded")
            return

        pointcloud_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".ply"):
                    path = root + "/" + f
                    pointcloud_files.append(path)

        if len(pointcloud_files) == 0:

            self.appendConsole("No files found to load!")

        else:
            self.pointcloudhandler.loaded_pointclouds = []

            counter = 1
            for f in pointcloud_files:
                path, filename = os.path.split(f)

                self.appendConsole("Loading " + filename + "... [" +  str(counter) + "/" + str(len(pointcloud_files)) + "]")
                if self.pointcloudhandler.loadPointcloud(path, filename, self):
                    self.appendConsole("Loaded " + filename + "!")
                else:
                    self.appendConsole("Couldn't load " + filename + "...")

                counter += 1

            self.loadedPointclouds = True

    def comm_button_load(self):

        path = None
        if self.rendering_mode == self.RENDER_FRAME or self.rendering_mode == self.RENDER_BUILD:
            if not self.calibrated:
                self.calibrated = self.loadCalibPath()
                if not self.calibrated:
                    return

            path = self.choseFolder("Please select a folder with a KITTI dataset")
            if path is None:
                return

            target_dirs = ["image_00", "image_01", "image_02", "image_03", "oxts", "velodyne_points"]

            kitti_path = None
            for root, dirs, files in os.walk(path):
                if set(target_dirs).issubset(set(dirs)):
                    kitti_path = root
                    break

            if kitti_path is not None:
                self.appendConsole("Kitti dataset folders found!")
                if self.prev_kitti != kitti_path:
                    self.prev_kitti = kitti_path

                    self.lidar.cleanup()
                    self.oxts.cleanup()

                    self.lidar.init(kitti_path, self.calibration, targetCamera="02", max_read=500, ui=self)
                    self.oxts.init(kitti_path)

                    self.setFrameCounter(0, self.lidar.count)
                else:
                    self.appendConsole("Previous KITTI folder path was selected")
                    return

                self.loadedKitti = True




        if self.rendering_mode == self.RENDER_POINTCLOUDS:
            self.promptReadPointcloudsFolder()

        self.initCurrentMode()

    def initCurrentMode(self):
        if self.rendering_mode == self.RENDER_FRAME:
            self.initFrameMode()

        if self.rendering_mode == self.RENDER_BUILD:
            self.initBuildMode()

        if self.rendering_mode == self.RENDER_POINTCLOUDS:
            self.initPointcloudMode()

        self.toggleSaveButton()

    def initPointcloudMode(self):
        if self.toggleUiRenderMode():
            self.current_pointcloud_frame = 0
            self.renderConsecutivePointcloud(self.current_pointcloud_frame)

    def initFrameMode(self):
        if self.toggleUiRenderMode():
            self.current_lidar_frame = 0
            self.renderConsecutiveFrame(self.current_lidar_frame)

    def initBuildMode(self):
        if self.toggleUiRenderMode():
            pass


    def toggleInitialLayoutState(self):
        self.button_save.config(state="disabled")
        self.button_prev.config(state="disabled")
        self.button_next.config(state="disabled")
        self.button_build.config(state="disabled")
        self.radio_filter.config(state="disabled")
        self.radio_icp.config(state="disabled")
        self.button_mode_switch.config(state="normal")
        self.radio_color.config(state="disabled")
        self.button_stop_build.config(state="disabled")

    def beginBuildUi(self):
        self.button_stop_build.config(state="normal")
        self.button_build.config(state="disabled")

        self.radio_icp.config(state="disabled")
        self.radio_filter.config(state="disabled")
        self.button_mode_switch.config(state="disabled")
        self.radio_color.config(state="disabled")
        self.button_load.config(state="disabled")
        self.button_calib.config(state="disabled")
        self.button_save.config(state="disabled")


    def endBuildUi(self):
        self.button_stop_build.config(state="disabled")
        self.button_build.config(state="normal")

        self.radio_icp.config(state="normal")
        self.radio_filter.config(state="normal")
        self.button_mode_switch.config(state="normal")
        self.radio_color.config(state="normal")
        self.button_load.config(state="normal")
        self.button_calib.config(state="normal")

        self.toggleSaveButton()

    def cycle_modes(self):
        if self.rendering_mode == self.RENDER_FRAME:
            self.rendering_mode = self.RENDER_BUILD
        elif self.rendering_mode == self.RENDER_BUILD:
            self.rendering_mode = self.RENDER_POINTCLOUDS
        elif self.rendering_mode == self.RENDER_POINTCLOUDS:
            self.rendering_mode = self.RENDER_FRAME

    def comm_button_mode(self):

        self.cycle_modes()
        self.renderer.reset()
        self.last_rendered_pointcloud = None
        self.setPointCounter(self.renderer.MemoryManager.getMaxPointIndex())



        self.appendConsole("[MODE] ", False)
        if self.rendering_mode == self.RENDER_FRAME:
            self.appendConsole("Frame walk")
        if self.rendering_mode == self.RENDER_BUILD:
            self.appendConsole("Build")
        if self.rendering_mode == self.RENDER_POINTCLOUDS:
            self.appendConsole("Point cloud")


        self.initCurrentMode()



    def toggleSaveButton(self):
        if self.renderer.MemoryManager.getMaxPointIndex() != 0:
            self.button_save.config(state="normal")
        else:
            self.button_save.config(state="disabled")

    def toggleUiRenderMode(self):

        if self.rendering_mode == self.RENDER_FRAME and self.loadedKitti:
            self.button_build.config(state='disabled')
            self.radio_filter.config(state='disabled')
            self.radio_icp.config(state='disabled')

            self.button_prev.config(state="normal")
            self.button_next.config(state="normal")
            self.radio_color.config(state="normal")

            return True
        elif self.rendering_mode == self.RENDER_FRAME and not self.loadedKitti:
                self.toggleInitialLayoutState()
                self.appendConsole("[WARN] Please load KITTI files")

        if self.rendering_mode == self.RENDER_BUILD and self.loadedKitti and self.calibrated:
            self.button_build.config(state='normal')
            self.radio_filter.config(state='normal')
            self.radio_icp.config(state='normal')

            self.button_prev.config(state="disabled")
            self.button_next.config(state="disabled")

            return True
        elif self.rendering_mode == self.RENDER_BUILD and (not self.loadedKitti or not self.calibrated):
                self.toggleInitialLayoutState()
                self.appendConsole("[WARN] Please load KITTI and CALIBRATION files")

        if self.rendering_mode == self.RENDER_POINTCLOUDS and self.loadedPointclouds:
            self.button_build.config(state='disabled')
            self.radio_filter.config(state='disabled')
            self.radio_icp.config(state='disabled')

            self.button_prev.config(state="normal")
            self.button_next.config(state="normal")
            self.radio_color.config(state="disabled")

            return True
        elif self.rendering_mode == self.RENDER_POINTCLOUDS and not self.loadedPointclouds:
                self.toggleInitialLayoutState()
                self.appendConsole("[WARN] Please load pointclouds")



        return False

    # FUNCTIONALS









    def comm_button_stop_build(self):
        self.stop_build = True


    def comm_button_build(self):
        self.beginBuildUi()

        self.renderer.reset()
        self.environment.cleanup()
        self.environment.init(0.5)


        start_from = 0
        until = self.lidar.count

        separate_colors = self.color_mode.get()
        remove_outliers = self.filter_mode.get()
        pure_imu = not self.icp_mode.get()

        ind = start_from
        while not self.stop_build and ind < until:
            lidardata, oxts = self.environment.getNextFrameData(0)
            self.environment.calculateTransition_imu(lidardata, oxts, point_to_plane=True, debug=False,
                                                           iterations=10, separate_colors=separate_colors, removeOutliers=remove_outliers, pure_imu=pure_imu )
            self.setPointCounter(self.renderer.MemoryManager.getMaxPointIndex())
            self.setFrameCounter(ind + 1, until)
            ind += 1

        if self.stop_build:
            self.appendConsole("Build process stopped")


        self.stop_build = False

        self.environment.voxelizer.stageAllRemaningVoxels(self)

        self.endBuildUi()

    def renderConsecutiveFrame(self, offset):
        filenames = self.lidar.getfilenames()

        self.current_lidar_frame += offset
        if self.current_lidar_frame < 0:
            self.current_lidar_frame = len(filenames) - 1

        if self.current_lidar_frame >= len(filenames):
            self.current_lidar_frame = 0


        points, ints, colors = self.lidar.getPointsWithIntensities(filenames[self.current_lidar_frame])

        if self.last_rendered_pointcloud is not None:
            self.renderer.freePoints(self.last_rendered_pointcloud)

        if not self.color_mode.get():
            self.last_rendered_pointcloud = self.renderer.addPoints(points, colors)
        else:
            self.last_rendered_pointcloud = self.renderer.addPoints(points, ints)


        self.setFrameCounter(self.current_lidar_frame + 1, len(filenames))

    def renderConsecutivePointcloud(self, offset):
        self.current_pointcloud_frame += offset
        if self.current_pointcloud_frame < 0:
            self.current_pointcloud_frame = len(self.pointcloudhandler.loaded_pointclouds) - 1

        if self.current_pointcloud_frame >= len(self.pointcloudhandler.loaded_pointclouds):
            self.current_pointcloud_frame = 0

        if self.last_rendered_pointcloud is not None:
            self.renderer.freePoints(self.last_rendered_pointcloud)

        points, colors = self.pointcloudhandler.getCloud(self.current_pointcloud_frame)

        if points is not None:
            self.last_rendered_pointcloud = self.renderer.addPoints(points, colors)

            self.setFrameCounter(self.current_pointcloud_frame + 1, len(self.pointcloudhandler.loaded_pointclouds))

    def comm_button_save(self):
        save_path = filedialog.asksaveasfilename(
            title="Please select a path to save the rendered points",
            defaultextension=".ply",
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
        )
        self.pointcloudhandler.savePointcloudPly(save_path, self)

    def comm_button_next(self):
        if self.rendering_mode == self.RENDER_FRAME:
            self.renderConsecutiveFrame(1)

        if self.rendering_mode == self.RENDER_POINTCLOUDS:
            self.renderConsecutivePointcloud(1)



    def comm_button_prev(self):
        if self.rendering_mode == self.RENDER_FRAME:
            self.renderConsecutiveFrame(-1)

        if self.rendering_mode == self.RENDER_POINTCLOUDS:
            self.renderConsecutivePointcloud(-1)







    def yesNoPopup(self, prompt, message):
        result = messagebox.askyesno(
            title=prompt,
            message=message
        )
        if result:
            return True
        else:
            return False

    def on_closing(self):
        self.appendConsole("Releasing resourcess...")
        if not self.testing:
            self.appendConsole("shaders...", False)
            self.compute.cleanup()
            self.appendConsole("released!")

            self.root.update()

            self.appendConsole("renderer...", False)
            self.renderer.close()
            self.appendConsole("released!")
            self.root.update()

            self.appendConsole("oxts data...", False)
            self.oxts.cleanup()
            self.appendConsole("released!")
            self.root.update()

            self.appendConsole("lidar data...", False)
            self.lidar.cleanup()
            self.appendConsole("released!")
            self.root.update()

            self.appendConsole("environment data...", False)
            self.environment.cleanup()
            self.appendConsole("released!")
            self.root.update()

            self.renderingThread.join()

            time.sleep(1)

        self.root.destroy()  # This actually closes the window

    def appendConsole(self, string, newline=True):
        self.text_output.config(state="normal")
        if newline:
            self.text_output.insert(tk.END, string + "\n")
        else:
            self.text_output.insert(tk.END, string)

        self.text_output.config(state="disabled")
        self.root.update()

    def setFrameCounter(self, current, mmax):
        self.frame_counter_var.set("Frame: " + str(current) + "/" + str(mmax))
        self.root.update()


    def setPointCounter(self, current):
        mil = current / 1000000.0
        mils = f"{mil:.2f}"
        self.point_counter_var.set("PC: " + mils + " M")
        self.root.update()



    def loadCalibPath(self):
        target_calib_files = ["calib_cam_to_cam.txt", "calib_imu_to_velo.txt", "calib_velo_to_cam.txt"]

        while True:
            file_path = filedialog.askdirectory(title="Please select a path to the calibration folder.")
            if file_path == '': # cancel
                self.appendConsole("Calibration cancelled!")
                return False

            for root, dirs, files in os.walk(file_path):
                if set(target_calib_files).issubset(set(files)):
                    self.appendConsole("Calibration file loaded!")
                    self.calibration = root + "/"
                    return True

    def intializeComponents(self):



        self.oxts = OxtsDataReader()
        self.lidar = LidarDataReader()


        self.renderer = Renderer(0.1, maxNumberOfPoints=20000000, anim=True)
        self.renderingThread = self.renderer.getRenderingThread()

        self.compute = ComputeShader() # instantiate after the renderer
        self.icp = PointcloudIcpContainer(self.compute, self.alignment)
        self.environment = EnvironmentConstructor(self.renderer, self.oxts, self.lidar, self.icp,
                                                        self.compute)
        self.pointcloudhandler = PointcloudHandler(self.renderer)

    def comm_button_calib(self):
        self.calibrated = self.loadCalibPath()


    def createUiLayout(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(self.resolution)
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.attributes("-topmost", True)

        # Make grid layout responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(5, weight=1)

        # === First Row: Load & Save Buttons (Expands with Window) ===
        row1 = tk.Frame(self.root, bg="black")
        row1.pack(fill="x", pady=5, padx=5)

        row1.columnconfigure(0, weight=1)
        row1.columnconfigure(1, weight=1)

        self.button_load = tk.Button(row1, text="Load Data", command=self.comm_button_load,
                                     bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_load.grid(row=0, column=0, sticky="ew", padx=5)

        self.button_save = tk.Button(row1, text="Save Rendered", command=self.comm_button_save,
                                     bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_save.grid(row=0, column=1, sticky="ew", padx=5)

        self.button_calib = tk.Button(row1, text="Calibration", command=self.comm_button_calib,
                                     bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_calib.grid(row=0, column=2, sticky="ew", padx=5)

        # === Second Row: Navigation Buttons & Frame Counter (Expands) ===
        row2 = tk.Frame(self.root, bg="black")
        row2.pack(fill="x", pady=5, padx=5)

        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)
        row2.columnconfigure(2, weight=1)

        self.button_prev = tk.Button(row2, text="Prev", command=self.comm_button_prev,
                                     bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_prev.grid(row=0, column=0, sticky="ew", padx=5)

        self.button_next = tk.Button(row2, text="Next", command=self.comm_button_next,
                                     bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_next.grid(row=0, column=1, sticky="ew", padx=5)

        self.frame_counter_var = tk.StringVar()
        self.frame_counter = tk.Entry(row2, textvariable=self.frame_counter_var, bg="white", fg="black", font=("Arial", 10, "bold"), state="readonly",justify="center")
        self.frame_counter.grid(row=0, column=2, sticky="ew", padx=5)
        self.setFrameCounter(0, 0)
        # === Color Mode Section ===
        color_mode_frame = tk.Frame(self.info_frame, bg="black", highlightbackground="white", highlightthickness=1)
        color_mode_frame.pack(fill="x", pady=10, padx=5)

        tk.Label(color_mode_frame, text="Color Mode", bg="black", fg="white").pack(anchor="w", padx=5)

        self.color_mode = tk.BooleanVar()

        self.radio_color = (tk.Checkbutton(color_mode_frame, text="Color", variable=self.color_mode, bg="black", fg="white",
                       selectcolor="gray"))
        self.radio_color.pack(side="left", padx=10, expand=True)

        env_buttons_frame = tk.Frame(self.info_frame, bg="black")
        env_buttons_frame.pack(fill="x", pady=(0, 10), padx=5)
        self.button_mode_switch = tk.Button(env_buttons_frame, text="Switch modes", command=self.comm_button_mode,
                                            bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_mode_switch.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.button_build = tk.Button(env_buttons_frame, text="Build", command=self.comm_button_build,
                                        bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_build.pack(side="left", expand=True, fill="x", padx=(5, 0))

        self.button_stop_build = tk.Button(env_buttons_frame, text="Stop", command=self.comm_button_stop_build,
                                      bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_stop_build.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # === Environment Modes Section ===
        env_mode_frame = tk.Frame(self.info_frame, bg="black", highlightbackground="white", highlightthickness=1)
        env_mode_frame.pack(fill="x", pady=10, padx=5)

        tk.Label(env_mode_frame, text="Environment Modes", bg="black", fg="white").pack(anchor="w", padx=5)


        self.icp_mode = tk.BooleanVar()
        self.filter_mode = tk.BooleanVar()

        self.radio_icp = (tk.Checkbutton(env_mode_frame, text="ICP", variable=self.icp_mode, bg="black", fg="white",
                       selectcolor="gray"))
        self.radio_icp.pack(side="left", padx=5, expand=True)

        self.radio_filter = (tk.Checkbutton(env_mode_frame, text="Filter", variable=self.filter_mode, bg="black", fg="white",
                       selectcolor="gray"))
        self.radio_filter.pack(side="left", padx=5, expand=True)

        self.point_counter_var = tk.StringVar()
        tk.Entry(env_mode_frame, textvariable=self.point_counter_var, bg="white", fg="black",
                                      font=("Arial", 10, "bold"), state="readonly", justify="center").pack(side="left", padx=5, expand=True)
        self.setPointCounter(0)

        # === Text Output Box ===
        output_frame = tk.Frame(self.info_frame, bg="black", highlightbackground="white", highlightthickness=1)
        output_frame.pack(fill="both", expand=True, pady=10, padx=5)

        tk.Label(output_frame, text="Console", bg="black", fg="white").pack(anchor="w", padx=5)

        self.text_output = tk.Text(output_frame, height=5, bg="white", fg="black")
        self.text_output.pack(fill="both", padx=5, pady=5, expand=True)
        self.text_output.config(state="disabled")

        self.toggleInitialLayoutState()

        self.appendConsole("[MODE] ", False)
        self.appendConsole("Frame walk")

        self.root.mainloop()