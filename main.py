from ComputeShader import ComputeShader
from EnvironmentConstructor import EnvironmentConstructor
from LidarDataReader import LidarDataReader
from LidarFilter import LidarFilter
from PointcloudAlignment import PointcloudAlignment
from OxtsDataReader import OxtsDataReader
from PointcloudIcpContainer import PointcloudIcpContainer
from Renderer import Renderer

import numpy as np

from Voxelizer import VoxelData


path = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"
pathOxts = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"

path2 = "F://uni//3d-pointcloud//samle1"
path3 = "F://uni//3d-pointcloud//sample2"

calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"


display = path

oxtsDataReader = OxtsDataReader(display)
lidarDataReader = LidarDataReader(path=display, oxtsDataReader=oxtsDataReader, calibration=calibration, targetCamera="02", max_read=100)

pointcloudAlignment = PointcloudAlignment(lidarDataReader, oxtsDataReader)

VOXEL_SIZE = 0.1
voxelizer = VoxelData(LidarFilter(maxRange=100, minHeight=0.1, voxelSize=VOXEL_SIZE, minPoints=2))


filenames = lidarDataReader.getfilenames()


renderer = Renderer(VOXEL_SIZE)
renderingThread = renderer.getRenderingThread()

computeShader = ComputeShader() # this has to be instantiated after the renderer!!! e_e
icpContainer = PointcloudIcpContainer(computeShader, pointcloudAlignment)
environmentConstructor = EnvironmentConstructor(renderer, oxtsDataReader, lidarDataReader, icpContainer)

"""start_from = 0
for i in range(5):
    lidardata = environmentConstructor.getNextFrameData(start_from)
    environmentConstructor.calculateTransition(lidardata, point_to_plane=True)
    # time.sleep(1)"""







computeShader.cleanup()
renderingThread.join()
