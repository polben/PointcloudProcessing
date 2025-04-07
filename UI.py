import os
import time
import tkinter as tk
from tkinter import ttk, filedialog


from ComputeShader import ComputeShader
from EnvironmentConstructor import EnvironmentConstructor
from LidarDataReader import LidarDataReader
from OxtsDataReader import OxtsDataReader
from PointcloudIcpContainer import PointcloudIcpContainer
from Renderer import Renderer


class UI:

    def __init__(self):
        self.point_counter_var = None
        self.root = None
        self.title = "Pointcloud Visualizer"
        self.resolution = "400x520"

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

        self.renderingThread = None
        self.calibration = None
        self.calibrated = False

        self.RENDER_SINGULAR = True

        # variables
        self.last_rendered_pointcloud = None
        self.current_lidar_frame = 0







        self.rendering_mode = self.RENDER_SINGULAR

        self.intializeComponents()

        self.createUiLayout()





    def initKittiMode(self, path):
        if not self.calibrated:
            self.calibrated = self.loadCalibPath()
            if not self.calibrated:
                return

        self.lidar.init(path, self.calibration, targetCamera="02", max_read=500, ui=self)
        self.oxts.init(path)

        self.setFrameCounter(0, self.lidar.count)

        if self.RENDER_SINGULAR:
            self.renderConsecutiveFrame(0)



    def comm_button_load(self):



        file_path = filedialog.askdirectory(title="Please select a folder.")

        target_dirs = ["image_00", "image_01", "image_02", "image_03", "oxts", "velodyne_points"]

        kitti_path = None
        for root, dirs, files in os.walk(file_path):
            if set(target_dirs).issubset(set(dirs)):
                kitti_path = root
                break

        if kitti_path is not None:
            self.appendConsole("Kitti dataset folders found!")
            self.initKittiMode(kitti_path)
            return

        pointcloud_files = []
        for root, dirs, files in os.walk(file_path):
            for f in files:
                if f.endswith(".ply"):
                    path = root + "/" + f
                    pointcloud_files.append(path)

        if not pointcloud_files and kitti_path is None:
            self.appendConsole("No files found to load!")

        else:
            self.appendConsole("The following pointcloud files are selected")
            for f in pointcloud_files:
                folder, filename = os.path.split(f)
                self.appendConsole(filename)
            self.pointcloud_paths = pointcloud_files


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

    def comm_button_save(self):
        print("Save")

    def comm_button_next(self):
        self.renderConsecutiveFrame(1)



    def comm_button_prev(self):
        self.renderConsecutiveFrame(-1)



    def createUiLayout(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(self.resolution)
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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

        tk.Checkbutton(color_mode_frame, text="Color", variable=self.color_mode, bg="black", fg="white",
                       selectcolor="gray").pack(side="left", padx=10, expand=True)

        env_buttons_frame = tk.Frame(self.info_frame, bg="black")
        env_buttons_frame.pack(fill="x", pady=(0, 10), padx=5)
        self.button_mode_switch = tk.Button(env_buttons_frame, text="Switch modes", command=self.comm_button_mode,
                                            bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_mode_switch.pack(side="left", expand=True, fill="x", padx=(0, 5))

        self.button_build = tk.Button(env_buttons_frame, text="Build", command=self.comm_button_build,
                                        bg="gray", fg="black", font=("Arial", 10, "bold"))
        self.button_build.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # === Environment Modes Section ===
        env_mode_frame = tk.Frame(self.info_frame, bg="black", highlightbackground="white", highlightthickness=1)
        env_mode_frame.pack(fill="x", pady=10, padx=5)

        tk.Label(env_mode_frame, text="Environment Modes", bg="black", fg="white").pack(anchor="w", padx=5)


        self.icp_mode = tk.BooleanVar()
        self.filter_mode = tk.BooleanVar()

        tk.Checkbutton(env_mode_frame, text="ICP", variable=self.icp_mode, bg="black", fg="white",
                       selectcolor="gray").pack(side="left", padx=5, expand=True)
        tk.Checkbutton(env_mode_frame, text="Filter", variable=self.filter_mode, bg="black", fg="white",
                       selectcolor="gray").pack(side="left", padx=5, expand=True)

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

        self.root.mainloop()

    def comm_button_build(self):
        self.renderer.reset()
        self.setPointCounter(self.renderer.MemoryManager.getMaxPointIndex())


        self.environment.cleanup()
        self.environment.init(0.5)

        start_from = 0
        until = self.lidar.count

        separate_colors = self.color_mode.get()
        remove_outliers = self.filter_mode.get()
        pure_imu = not self.icp_mode.get()
        for i in range(until - start_from):
            lidardata, oxts = self.environment.getNextFrameData(start_from)
            self.environment.calculateTransition_imu(lidardata, oxts, point_to_plane=True, debug=False,
                                                           iterations=20, separate_colors=separate_colors, removeOutliers=remove_outliers, pure_imu=pure_imu )
            self.setPointCounter(self.renderer.MemoryManager.getMaxPointIndex())
        self.environment.voxelizer.stageAllRemaningVoxels(self)



    def comm_button_mode(self):
        self.appendConsole("Switching render modes")
        self.RENDER_SINGULAR = not self.RENDER_SINGULAR

    def on_closing(self):
        self.appendConsole("Releasing resourcess...")

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


        self.renderer = Renderer(0.1, maxNumberOfPoints=20000000)
        self.renderingThread = self.renderer.getRenderingThread()

        self.compute = ComputeShader() # instantiate after the renderer
        self.icp = PointcloudIcpContainer(self.compute, self.alignment)
        self.environment = EnvironmentConstructor(self.renderer, self.oxts, self.lidar, self.icp,
                                                        self.compute)


    def comm_button_calib(self):
        self.calibrated = self.loadCalibPath()
