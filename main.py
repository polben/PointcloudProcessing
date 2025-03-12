import math
import time
from math import atan2
from random import randint

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

red = np.array([255,0,0])
green = np.array([0,255,0])
blue = np.array([0,0,255])
white = np.array([255,255,255])






def binaryScanLineSearch(points, scanline, reference_point, origin):


    def cos(index, refvec, origin):
        lidar_tall = 0.254
        lidarOffset = np.array([0, lidar_tall / 2.0, 0])

        v = points[index] - origin - lidarOffset
        # v[1] = 0
        v = v / np.linalg.norm(v)
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



    l = []
    l.extend([origin, reference_point])

    refvec = reference_point - origin
    refvec[1] = 0
    refvec /= np.linalg.norm(refvec)  # Normalize

    begin, end = scanline
    circlen = end - begin + 1


    index = (begin + end) // 2
    half = circlen // 2

    runs = int(math.ceil(math.log2(circlen)))

    counter = 0
    for i in range(runs):
        counter += 1
        d, scale = dir(index=index, circlen=circlen, begin=begin, refVec=refvec, origin=origin)
        index = indmod(index + int(half * d), circlen, begin)
        l.extend([origin, points[begin + index]])

        half = int(math.ceil(half / 2.0))
        renderer.setLines(l)
        time.sleep(1)
        l.pop()
        l.pop()

    # print(counter)
    return begin + index




def getClosestPointNaive(np_points, reference_point):
    return icpContainer.__findClosestPointNaiveNumpy__(reference_point, np_points)

def getClosestPointScan(np_points, reference_point, scan_lines):
    mindist = 99999.0
    minind = 0

    for sl in scan_lines:

        closest = binaryScanLineSearch(np_points, sl, reference_point)


        d = icpContainer.dist(np_points[closest], reference_point)
        if d < mindist:
            mindist = d
            minind = closest




    return minind


def fullScanCheck(renderer):
    points1, cols = lidarDataReader.getPoints(filenames[0])
    points1 = pointcloudAlignment.align(filenames[0], points1)
    points2, cols = lidarDataReader.getPoints(filenames[1])
    points2 = pointcloudAlignment.align(filenames[1], points2)
    renderer.addPoints(points1, np.array([255, 0, 0]))
    renderer.addPoints(points2, np.array([0, 255, 0]))

    scan_lines = icpContainer.getScanLines(points1, np.array([0,0,0]))

    lines = []

    for i in range(60000, len(points2)):
        refp = points2[i]


        closest = getClosestPointScan(points1, refp, scan_lines)
        lines.extend([refp, points1[closest]])
        if len(lines) % 1000 == 0:
            renderer.setLines(lines)


def scanCorrespondenceTest(renderer):
    points1, cols = lidarDataReader.getPoints(filenames[0])
    points1 = pointcloudAlignment.align(filenames[0], points1)
    points2, cols = lidarDataReader.getPoints(filenames[1])
    points2 = pointcloudAlignment.align(filenames[1], points2)
    renderer.addPoints(points1)
    renderer.addPoints(points2)

    scan_lines = icpContainer.getScanLines(points1, np.array([0,0,0]))

    lines = []

    for i in range(10000):
        ri = randint(1, len(points2))
        refp = points2[ri]

        st = time.time()
        closest = getClosestPointScan(points1, refp, scan_lines)
        # print("scan time: " + str(time.time()- st))

        st = time.time()
        closest_naive = getClosestPointNaive(points1, refp)
        # print("naive time: " + str(time.time()- st))


        if closest_naive != closest:

            p_naive = points1[closest_naive]
            p_scan = points1[closest]
            if abs(icpContainer.dist(refp, p_scan) - icpContainer.dist(refp, p_naive)) < 0.1:
                continue


            print("error!: " + str(ri))
            lines.extend([refp, p_scan, refp, p_naive, refp, refp + np.array([0, 10, 10])])
            renderer.setLines(lines)
        else:



            if i % 100 == 0:
                print(i)

def testClosestPoint(renderer, point_to_test):
    pts = getLidarPoints()

    refp = pts[point_to_test] + np.array([0, 0.5, 0])
    projs = icpContainer.sphereProjectPoints(pts)


    renderer.addPoints(pts)
    scan_lines = icpContainer.getScanLines(projs,np.array([0,0,0]))

    closest_scan = getClosestPointScan(pts, refp, scan_lines)
    closest_naive = getClosestPointNaive(pts, refp)
    if closest_naive == closest_scan:
        print("actually ok, lol")

    renderer.setLines([refp, pts[closest_scan], refp, pts[closest_naive], refp, refp + np.array([0, 10, 10])])

def debugBinScan(renderer):
    pts_a, cols = lidarDataReader.getPoints(filenames[0])
    pts_a = pointcloudAlignment.align(filenames[0], pts_a)
    origin, rot = pointcloudAlignment.getPose(filenames[0])

    scan_lines = icpContainer.getScanLines(pts_a, origin)

    pts_b, col = lidarDataReader.getPoints(filenames[1])
    pts_b = pointcloudAlignment.align(filenames[0], pts_b)


    lines = []



    renderer.addPoints(pts_a, None)
    renderer.addPoints(pts_b, None)
    #for i in range(len(pts_b)):
    if True:
        i = 20000
        refp = pts_b[i]


        height = (refp / np.linalg.norm(refp))[1]
        low = 0
        high = len(scan_lines) - 1
        closest = 0

        minheightdiff = 1.23e20
        minheightind = 0
        while low < high:

            mid = (low + high) // 2
            scanindex = mid
            sl = scan_lines[scanindex]

            closest = binaryScanLineSearch(pts_a, sl, refp, np.array([0,0,0]))
            renderer.setLines([pts_a[closest], refp])
            time.sleep(1)

            pc = pts_a[closest] - origin
            pc = (pc / np.linalg.norm(pc))

            ht = abs(pc[1] - height)
            if ht < minheightdiff:
                minheightdiff = ht
                minheightind = closest

            if pc[1] > height:
                low = mid + 1

            else:
                high = mid




        closest = minheightind

        lines.append(pts_a[closest])
        lines.append(pts_b[i])

        if i % 1000 == 0:
            renderer.setLines(lines, None)





def visAngleScan_rawScanLineHeight(renderer):
    origin = np.array([0,0,0])


    pts = getLidarPoints() + origin


    randind = randint(1, len(pts))
    #randind = 8 * 2000 -800
    randind = 20 * 2000 # 7 two scan line miss
    print(randind)
    refp = pts[randind]

    lidar_tall = 0.254
    lidar_offset = np.array([0, lidar_tall / 2, 0])

    # renderer.addPoints(projs)
    renderer.addPoints(pts)
    scan_lines = icpContainer.getScanLines(pts, origin)
    #pts = icpContainer.sphereProjectPoints(pts, origin)
    #renderer.addPoints(pts) break

    lines = []
    line_colors = []

    lines.append(refp) # add origin line to refp
    lines.append(np.array([0,0,0]))

    line_colors.extend([np.array([255,0,0]), np.array([255,0,0])])


    def addScanLineRender(sl):
        s, e = sl
        lps = pts[s:e+1]
        lps = icpContainer.connectPointsInOrder(lps)
        color = np.array([randint(0, 255), randint(0, 255), randint(0, 255)])
        for p in lps:
            lines.append(p)
            line_colors.append(color)

    def indmod(index, clen, cbegin):
        return (index + clen) % clen


    renderer.addPoints([refp / np.linalg.norm(refp)], None)
    height = icpContainer.sphereProjectPont(refp, origin)[1]
    low = 0
    high = len(scan_lines) - 1
    closest = 0

    count = 0

    minheightdiff = 1.23e20
    minheightind = 0
    while low < high:

        mid = (low + high) // 2
        scanindex = mid
        sl = scan_lines[scanindex]
        addScanLineRender(sl)
        begin, end = sl

        closest = begin

        pc = pts[closest]

        lines.extend([pc, np.array([0,0,0])])
        line_colors.extend([green, blue])


        pc = icpContainer.sphereProjectPont(pc, origin)
        renderer.addPoints([pc], None)

        ht = abs(pc[1] - height)
        if ht < minheightdiff:
            minheightdiff = ht
            minheightind = closest

        if pc[1] > height:
            low = mid + 1

        else:
            high = mid

        renderer.setLines(lines, line_colors)

        time.sleep(1)

    closest = minheightind

    lines.append(pts[closest])
    lines.append(pts[closest] + np.array([0, 1, 0]))
    line_colors.extend([np.array([255, 255, 255]), np.array([255, 255, 255])])
    renderer.setLines(lines, line_colors)
def visAngleScan(renderer):
    pts = getLidarPoints()


    randind = randint(1, len(pts))
    #randind = 8 * 2000 -800
    randind = 42 * 2000 # 7 two scan line miss
    print(randind)
    refp = pts[randind]


    # renderer.addPoints(projs)
    renderer.addPoints(pts)
    scan_lines = icpContainer.getScanLines(pts, np.array([0,0,0]))
    lines = []
    line_colors = []

    lines.append(refp) # add origin line to refp
    lines.append(np.array([0,0,0]))

    line_colors.extend([np.array([255,0,0]), np.array([255,0,0])])


    def addScanLineRender(sl):
        s, e = sl
        lps = pts[s:e+1]
        lps = icpContainer.connectPointsInOrder(lps)
        color = np.array([randint(0, 255), randint(0, 255), randint(0, 255)])
        for p in lps:
            lines.append(p)
            line_colors.append(color)

    height = (refp / np.linalg.norm(refp))[1]
    low = 0
    high = len(scan_lines) - 1
    closest = 0

    count = 0

    minheightdiff = 1.23e20
    minheightind = 0
    while low < high:
        count += 1
        if count >= 0: # 10
            mid = (low + high) // 2
            scanindex = mid
            sl = scan_lines[scanindex]


            addScanLineRender(sl)

            prev_best = closest
            closest = binaryScanLineSearch(pts, sl, refp, np.array([0,0,0]))
            pc = pts[closest]
            pc = (pc / np.linalg.norm(pc))

            ht = abs(pc[1] - height)
            if ht < minheightdiff:
                minheightdiff = ht
                minheightind = closest

            if pc[1] > height:
                low = mid + 1

            else:
                high = mid

            renderer.setLines(lines, line_colors)

            #time.sleep(1)

    closest = minheightind

    lines.append(pts[closest])
    lines.append(pts[closest] + np.array([0, 1, 0]))
    line_colors.extend([np.array([255, 255, 255]), np.array([255, 255, 255])])
    renderer.setLines(lines, line_colors)


def getLidarPoints():
    pts, cols = lidarDataReader.getPoints(filenames[0])
    return pts

def getRandomLidarPoint(lidar_points):
    return lidar_points[randint(1, len(lidar_points))] + np.array([0, 1, 0])




def visScanLines(renderer):
    pts_l = getLidarPoints()
    pts = icpContainer.sphereProjectPoints(pts_l, np.array([0,0,0]))

    sphere = icpContainer.sphereProjectPoints(pts_l, np.array([0,0,0])) * 1.01
    renderer.addPoints(pts_l, np.array([0.5, 0.5, 0.5]))



    stt = time.time()
    scan_lines = icpContainer.getScanLines(pts_l, np.array([0,0,0]))

    print("scans: " + str(time.time() - stt))

    for start, end in scan_lines:
        scan_points = pts_l[start:end]
        renderer.setLines(icpContainer.connectPointsInOrder(scan_points), None)
        time.sleep(0.1)

def showScanLines(renderer, points, scan_lines):
    for start, end in scan_lines:
        scan_points = points[start:end]
        renderer.setLines(icpContainer.connectPointsInOrder(scan_points), None)
        time.sleep(0.5)


def sphereProjectLidarPoints(renderer):
    pts1, cols1 = lidarDataReader.getPoints(filenames[0])
    pts2, cols2 = lidarDataReader.getPoints(filenames[1])
    pc = pts1 - np.array([0, 0, 0])

    magnitudes = np.linalg.norm(pc, axis=1)
    pc = pc / magnitudes[:, np.newaxis]

    lines = icpContainer.connectPointsInOrder(pc)
    renderer.addPoints(pc, np.array([1, 1, 1]))
    renderer.setLines(lines)

def lidarMortonCorrespondenceTest(renderer, icpContainer, lidarDataReader):
    filenames = lidarDataReader.getfilenames()
    pc1, cols1 = lidarDataReader.getPoints(filenames[0])
    pc2, cols2 = lidarDataReader.getPoints(filenames[1])
    pc1 = pointcloudAlignment.align(filenames[0], pc1)
    pc2 = pointcloudAlignment.align(filenames[1], pc2)

    renderer.addPoints(pc1, cols1)
    renderer.addPoints(pc2, cols2)

    align_to_ref_morton = icpContainer.getCorrespondencesComputeMorton(pc1, pc2)
    lines = []
    for i in range(len(align_to_ref_morton)):
        corr_ind = align_to_ref_morton[i]
        lines.append(pc1[corr_ind])
        lines.append(pc2[i])
    renderer.setLines(lines, None)

def uniformGridMortonCorrespondeceTest(renderer, icpContainer):
    points_ref = icpContainer.getUniformGrid(10)
    points_to_align = icpContainer.getUniformGrid(10) + np.array([0, 0.1, 0])

    renderer.addPoints(points_ref, None)
    renderer.addPoints(points_to_align, None)

    align_to_ref_morton = icpContainer.getCorrespondencesComputeMorton(points_ref, points_to_align)
    lines = []
    for i in range(len(align_to_ref_morton)):
        corr_ind = align_to_ref_morton[i]
        lines.append(points_ref[corr_ind])
        lines.append(points_to_align[i])
    renderer.setLines(lines, None)

def uniformGridMortonCurve(renderer, icpContainer):
    points_ref = icpContainer.getUniformGrid(10)

    renderer.addPoints(points_ref, None)

    ext, min = icpContainer.__getExtentOfPointcloud__(points_ref)
    norm = icpContainer.__normalizePointcloud__(points_ref, ext, min)
    norm_grid = icpContainer.__getMortonGrid__(norm)
    mortonCodes = icpContainer.__getMortonCodes_vectorized__(norm_grid)
    sorted_inds = np.argsort(mortonCodes, axis=0)
    sorted_points = points_ref[sorted_inds]

    lines = icpContainer.connectPointsInOrder(sorted_points)
    for i in range(int(len(lines) / 2)):
        renderer.setLines(lines[:i * 2], None)
        time.sleep(0.1)

def uniformGridICP_LS(renderer, icpContainer):

    grid1 = icpContainer.getUniformShape()
    grid2 = (PointcloudAlignment.randomRotation(0.1) @ icpContainer.getUniformShape().T).T + PointcloudAlignment.randomTranslation1() * 2

    red = np.array([1, 0, 0])
    green = np.array([0, 1, 0])
    p1 = renderer.addPoints(grid1, green)
    p2t = renderer.addPoints(grid2, red)

    time.sleep(5)
    renderer.freePoints(p2t)

    """tree1 = icpContainer.getKDTree(grid1)
    for i in range(100):
        R, t = icpContainer.icp_step(tree1, grid1, grid2)
        grid2 = icpContainer.applyIcpStep(grid2, R, t)
        p2 = renderer.addPoints(grid2, red)
        time.sleep(0.1)
        renderer.freePoints(p2)
    """
    p2 = renderer.addPoints(grid2)

def display_points(renderer, icpContainer):
    key1 = filenames[5]
    key2 = filenames[6]

    pts1, cols = lidarDataReader.getPoints(key1)


    pts2, cols = lidarDataReader.getPoints(key2)


    renderer.addPoints(pts1, green)
    renderer.addPoints(pts2, red)

def incremental_ls_test(renderer, icpContainer):




def uniformGridICP_LS(renderer, icpContainer):

    grid1 = icpContainer.getUniformShape()
    R, (x,y,z) = PointcloudAlignment.randomRotation(0.4)
    t = PointcloudAlignment.randomTranslation1() * 10
    grid2 = ( R @ icpContainer.getUniformShape().T).T + t

    red = np.array([1, 0, 0])
    green = np.array([0, 1, 0])
    p1 = renderer.addPoints(grid1, green)
    p2t = renderer.addPoints(grid2, red)


    time.sleep(3)






    for i in range(10):
        t, R = icpContainer.icp_step_LS_vector(grid1, grid2)
        #mean = np.mean(grid2, axis=0)
        #grid2 = grid2 - mean
        grid2 = (PointcloudAlignment.rotation(R[0], R[1], R[2]) @ grid2.T).T
        grid2 -= t
        #grid2 += mean


        renderer.freePoints(p2t)
        p2t = renderer.addPoints(grid2)

        if p1 is None:
            p1 = renderer.addPoints(grid1, green)
        else:
            renderer.freePoints(p1)
            p1 = None
        time.sleep(1)

def groundDetectionIcpTest(renderer, icpcontainer):
    points = getLidarPoints()
    origin = np.array([0, 1, 0])

    randrot = PointcloudAlignment.randomRotation(0.01)

    # points = (randrot @ points.T).T
    points += origin

    renderer.addPoints(points, blue)

    start = time.time()
    norm, point = icpcontainer.estimateGroundIcp(points, origin, None)
    print(str(time.time() - start))

    not_ground = points[icpContainer.remove_points_near_plane(points, norm, point, 0.2) ]+ np.array([0, 0.01, 0])

    renderer.addPoints(not_ground, red)

def groundDetectionClosestPointTest(renderer, icpContainer):
    points = getLidarPoints()
    origin = np.array([0,1,0])

    points += origin
    renderer.addPoints(points)

    lidar_height = 1.73
    plane = icpContainer.get_plane_grid(100, -lidar_height)
    renderer.addPoints(plane)

    sls = icpContainer.getScanLines(points, origin)  # origin has to be taken into cons


    corrs, dists = icpContainer.getCorrespondencesComputeScan(points, sls, plane, origin)  # ~0.02 sec with raw shader

    lines = []
    for i in range(len(corrs)):
        c = corrs[i]
        lines.extend([points[c], plane[i]])
    renderer.setLines(lines)

def testNNSComputeVisPerf(renderer):
    origin = np.array([0,0,0])

    frame1 = 0
    frame2 = 1

    pts1, cols = lidarDataReader.getPoints(filenames[frame1])
    pts1 = pointcloudAlignment.align(filenames[frame1], pts1)

    pos1, rot1 = pointcloudAlignment.getPose(filenames[frame1])


    pts2, cols = lidarDataReader.getPoints(filenames[frame2])
    pts2 = pointcloudAlignment.align(filenames[frame2], pts2) + np.array([0.01,0,0])

    pos2, rot2 = pointcloudAlignment.getPose(filenames[frame2])

    #proj = icpContainer.sphereProjectPoints(pts2, origin2)
    #renderer.addPoints(proj)


    renderer.addPoints([pos1, pos2], [red, green])

    sls = icpContainer.getScanLines(pts1, pos1) # origin has to be taken into cons

    renderer.addPoints(pts1, np.array([255, 0, 0]))

    # showScanLines(renderer, pts1, sls)

    renderer.addPoints(pts2, np.array([0, 255, 0]))

    st = time.time()
    corrs, dists = icpContainer.getCorrespondencesComputeScan(pts1, sls, pts2, pos1) # ~0.02 sec with raw shader
    print("that closest point took: "+ str(time.time() - st))


    lines = []
    for i in range(len(corrs)):
        c = corrs[i]
        lines.extend([pts2[i], pts1[c]])
    renderer.setLines(lines)

def ICPtest(renderer, icpcontainer):
    for i in range(0, 20):
        filename = filenames[i]
        points, colors = lidarDataReader.getPoints(filename)
        points_aligned = pointcloudAlignment.align(filename, points)
        # features = voxelizer.getFilteredPoints(points_aligned)

        icpContainer.addAndAlign(filename, points_aligned)
        renderer.addPoints(icpContainer.getAlignedOriginal(filename), colors)
        # time.sleep(0.1)

path = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"
pathOxts = "F://uni//3d-pointcloud//2011_09_26_drive_0005_sync"

path2 = "F://uni//3d-pointcloud//samle1"
path3 = "F://uni//3d-pointcloud//sample2"

calibration = "F://uni//3d-pointcloud//2011_09_26_calib//2011_09_26"


display = path

oxtsDataReader = OxtsDataReader(display)
lidarDataReader = LidarDataReader(path=display, oxtsDataReader=oxtsDataReader, calibration=calibration, targetCamera="02")

pointcloudAlignment = PointcloudAlignment(lidarDataReader, oxtsDataReader)

VOXEL_SIZE = 0.1
voxelizer = VoxelData(LidarFilter(maxRange=100, minHeight=0.1, voxelSize=VOXEL_SIZE, minPoints=2))






filenames = lidarDataReader.getfilenames()





renderer = Renderer(VOXEL_SIZE)
renderingThread = renderer.getRenderingThread()

computeShader = ComputeShader() # this has to be instantiated after the renderer!!! e_e
icpContainer = PointcloudIcpContainer(computeShader, pointcloudAlignment)


environmentConstructor = EnvironmentConstructor(renderer, oxtsDataReader, lidarDataReader, icpContainer)

start_from = 0
for i in range(3):
    lidardata = environmentConstructor.getNextFrameData(start_from)
    environmentConstructor.calculateTransition(lidardata)
    # time.sleep(1)



# display_points(renderer, icpContainer)



# groundDetectionIcpTest(renderer, icpContainer)
# groundDetectionClosestPointTest(renderer, icpContainer)

incremental_ls_test(renderer, icpContainer)

# full_ls_test(renderer, icpContainer)

# uniformGridICP_LS(renderer, icpContainer)

# ICPtest(renderer, icpContainer)

# visScanLines(renderer)

# testNNSComputeVisPerf(renderer)




# visAngleScan_rawScanLineHeight(renderer)

# testClosestPoint(renderer, 105970)
# debugBinScan(renderer)
# fullScanCheck(renderer)
# scanCorrespondenceTest(renderer)
# lidarMortonCorrespondenceTest(renderer, icpContainer, lidarDataReader)
# visAngleScan(renderer)
# uniformGridMortonCorrespondeceTest(renderer, icpContainer)
# uniformGridMortonCurve(renderer, icpContainer)
# uniformGridICP(renderer, icpContainer)





computeShader.cleanup()
renderingThread.join()




















"""
for f in filenames:
    points = lidarDataReader.getPoints(f)
    aligned = pointcloudAlignment.align(f, points)
    filtered = voxelizer.getFilteredPoints(aligned)

    icpContainer.addAndAlign(f, aligned, filtered)
    renderer.addPoints(icpContainer.getAlignedFiltered(f))
    #renderer.addPoints(filtered)
    # time.sleep(1)
"""

"""
grid1 = icpContainer.getUniformGrid(10)
grid2 = icpContainer.getUniformGrid(10)

rotation = PointcloudAlignment.randomRotation(0.1)
translation = PointcloudAlignment.randomTranslation1()
grid1 = (rotation @ grid1.T).T + translation * 5


rotation = PointcloudAlignment.randomRotation(0.1)
translation = PointcloudAlignment.randomTranslation1()
grid2 = (rotation @ grid2.T).T + translation * 5


renderer.addPoints([np.array([0,0,0])])
g1p = renderer.addPoints(grid1)
g2p = renderer.addPoints(grid2)

time.sleep(5)

aligned, R_opt, t_opt = icpContainer.align(grid1, grid2)



galignedp = renderer.addPoints(aligned)

time.sleep(1)

renderer.freePoints(g2p)
renderer.freePoints(galignedp)
time.sleep(1)
g2p = renderer.addPoints(icpContainer.applyIcpStep(grid2, R_opt, t_opt))
"""
