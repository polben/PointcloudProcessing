from ComputeShader import ComputeShader
from EnvironmentConstructor import EnvironmentConstructor
from LidarDataReader import LidarDataReader
from PointcloudAlignment import PointcloudAlignment
from OxtsDataReader import OxtsDataReader
from PointcloudIcpContainer import PointcloudIcpContainer
from Renderer import Renderer




path1 = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"
pathOxts = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"

path2 = "F://uni//3d-pointcloud//samle1"
path3 = "F://uni//3d-pointcloud//sample2"

calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"


display = path3

oxtsDataReader = OxtsDataReader(display)
lidarDataReader = LidarDataReader(path=display, oxtsDataReader=oxtsDataReader, calibration=calibration, targetCamera="02", max_read=500)

pointcloudAlignment = PointcloudAlignment(lidarDataReader, oxtsDataReader)

VOXEL_SIZE = 0.1

filenames = lidarDataReader.getfilenames()


renderer = Renderer(VOXEL_SIZE)
renderingThread = renderer.getRenderingThread()

computeShader = ComputeShader() # this has to be instantiated after the renderer!!! e_e
icpContainer = PointcloudIcpContainer(computeShader, pointcloudAlignment)
environmentConstructor = EnvironmentConstructor(renderer, oxtsDataReader, lidarDataReader, icpContainer)

start_from = 200
until = 400 # lidarDataReader.count
for i in range(until - start_from):
    lidardata, oxts = environmentConstructor.getNextFrameData(start_from)
    environmentConstructor.calculateTransition_imu(lidardata, oxts, point_to_plane=True, debug=False, iterations=15, cullColors=True, removeOutliers=False, pure_imu=False)
    # time.sleep(0.1)







computeShader.cleanup()
renderingThread.join()
