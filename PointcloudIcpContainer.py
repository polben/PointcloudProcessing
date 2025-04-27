import time
from math import atan2

import numpy as np
import numpy.linalg

from scipy.spatial import KDTree

from PointcloudAlignment import PointcloudAlignment


class PointcloudIcpContainer:

    def __init__(self, computeShader, pointcloudAlignment):
        self.compute = computeShader
        self.pointcloudAlignment = pointcloudAlignment



    def preparePointToPoint(self, points_a, origin, points_b):
        scan_lines = self.getScanLines(points_a, origin)
        self.compute.prepareLS(points_a, scan_lines, points_b, origin, False)

    def dispatchPointToPoint(self, points_b):
        H, b = self.compute.dispatchLS(points_b)

        delta_x = np.linalg.solve(H, -b)
        t, R = delta_x[:3], delta_x[3:]
        return t, PointcloudAlignment.rotation(R[0], R[1], R[2])

    def preparePointToPlane(self, points_a, origin, points_b):

        st = time.time()
        scan_lines = self.getScanLines(points_a, origin)
        # print("scan line time: " + str(time.time()-st))

        st = time.time()
        self.compute.preparePointPlane(points_a, scan_lines, points_b, origin, False)
        print("prep time: " + str(time.time() - st)) # ~0.1s







    def dispatchPointToPlane(self, points_b):

        # Hs, Bs = self.compute.dispatchPointPlane(points_b)
        H, b = self.compute.dispatchPointPlane(points_b)




        """H = np.nansum(Hs, axis=0)[:6, :6]
        b = np.nansum(Bs, axis=0)[:6]"""

        try:
            delta_x = np.linalg.solve(H, -b)
        except numpy.linalg.LinAlgError as e:
            print("failed to solve for transition")
            print(H)
            print(b)
            a = 0
        t, R = delta_x[:3], delta_x[3:]
        # print(delta_x)
        delta_transf = np.mean(np.abs(delta_x))
        # print(delta_transf)
        return t, PointcloudAlignment.rotation(R[0], R[1], R[2]), delta_transf

    def full_pt2pt(self, aligned, to_align, origin, iterations = 20, renderer = None):

        self.preparePointToPoint(aligned, origin, to_align)

        reference_grid = self.getUniformGrid(10)


        prev_pp = None
        if renderer is not None:
            prev_pp = renderer.addPoints(to_align, np.array([0,0,255]))
            time.sleep(1)

        start = time.time()
        for i in range(iterations):
            t, R = self.dispatchPointToPoint(to_align)
            to_align = self.apply_iteration_of_ls(to_align, t, R)

            if renderer is not None:
                if prev_pp is not None:
                    renderer.freePoints(prev_pp)
                prev_pp = renderer.addPoints(to_align)
                time.sleep(0.1)

            reference_grid = self.applyIcpStep(reference_grid, R, t)

        print("icp ls: " + str(time.time()-start))

        R_opt, t_opt = self.uniform_optimal_icp(reference_grid, self.getUniformGrid(10))

        return t_opt, R_opt


    def full_pt2pl(self, aligned, to_align, origin, iterations = 20, renderer = None, full_iter = False):

        self.preparePointToPlane(aligned, origin, to_align)

        reference_grid = self.getUniformGrid(10)


        prev_pp = None
        if renderer is not None:
            prev_pp = renderer.addPoints(to_align, np.array([0,0,255]))
            time.sleep(1)


        start = time.time()
        c = 0
        for i in range(iterations):
            c += 1
            t, R, delta_transf = self.dispatchPointToPlane(to_align)
            if delta_transf < 1e-6 and not full_iter:
                break

            to_align = self.apply_iteration_of_ls(to_align, t, R)

            if renderer is not None:
                print(i)
                if prev_pp is not None:
                    renderer.freePoints(prev_pp)
                prev_pp = renderer.addPoints(to_align)
                time.sleep(0.1)

            reference_grid = self.applyIcpStep(reference_grid, R, t)
        # print(c)
        if renderer is not None:
            renderer.freePoints(prev_pp)
        # print("icp plane: " + str(time.time()-start))

        start_opt = time.time()
        R_opt, t_opt = self.uniform_optimal_icp(reference_grid, self.getUniformGrid(10))
        # print("opt trans: " + str(time.time()-start_opt))


        return t_opt, R_opt

    def initLS(self, aligned, to_align, origin):

        scan_lines = self.getScanLines(aligned, origin)
        self.compute.prepareLS(aligned, scan_lines, to_align, origin)

    def prepareNNS(self, points_a, origin, points_b):
        scan_lines = self.getScanLines(points_a, origin)
        self.compute.prepareNNS(points_a, scan_lines, points_b, origin)

    def dispatchNNS(self, points_b):
        return self.compute.dispatchNNS(points_b)

    def iteration_of_ls(self, to_align):
        #st = time.time()
        Hs, Bs = self.compute.dispatchLS(to_align)
        #print("compute time: " + str(time.time()-st))

        #st = time.time()
        H = np.sum(Hs, axis=0)[:6, :6]
        b = np.sum(Bs, axis=0)[:6]
        #print("numpy time: " + str(time.time()-st))

        delta_x = np.linalg.solve(H, -b)

        return delta_x[:3], delta_x[3:]

    def apply_iteration_of_ls(self, to_align, t, R):

        return (R @ to_align.T).T - t



    def getHsBsCompute(self, to_align):
        return self.compute.dispatchLS(to_align)

    def getHsBsNumpy(self, aligned, to_align):
        J = self.getJ(to_align)
        J_T = J.swapaxes(1, 2)
        Hs = np.matmul(J_T, J)
        e = (aligned - to_align)[:, :, np.newaxis]
        Bs = np.matmul(J_T, e).squeeze(-1)
        return Hs, Bs


    def sumHsBsNumpy(self, Hs, Bs):
        H = np.sum(Hs, axis=0)
        b = np.sum(Bs, axis=0)
        return H, b


    def icp_step_LS_vector(self, aligned, to_align):

        Hs, Bs = self.getHsBsNumpy(aligned, to_align)

        H, b = self.sumHsBsNumpy(Hs, Bs)


        delta_x = np.linalg.solve(H, -b)

        return delta_x[:3], delta_x[3:]

    def getHsBsLoop(self, aligned, to_align):
        Hs, Bs = [], []
        for i in range(len(to_align)):
            p1 = aligned[i]
            p2 = to_align[i]

            e_i = p1 - p2
            J_i = self.getJ_i(to_align[i])
            H_i = self.getH_i(J_i)
            b_i = self.getB(J_i, e_i)

            Hs.append(H_i)
            Bs.append(b_i)

        return np.array(Hs), np.array(Bs)

    def sumHsBsLoop(self, Hs, Bs):

        H = np.zeros((6, 6))
        b = np.zeros(6)
        for i in range(len(Bs)):
            H = self.addH(H, Hs[i])
            b = self.addB(b, Bs[i])

        return H, b




    def addH(self, H, H_i):
        H_ret = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                H_ret[i][j] = H[i][j] + H_i[i][j]

        return H_ret

    def addB(self, B, b_i):
        B_ret = np.zeros((6,))
        for i in range(6):
            B_ret[i] = B[i] + b_i[i]

        return B_ret

    def getH_i_vector(self, J_i):
        return J_i.T @ J_i

    def getH_i(self, J_i):
        H_i = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                sum_val = 0.0
                for k in range(3):
                    sum_val += J_i[k][i] * J_i[k][j]

                H_i[i][j] = sum_val


        return H_i

    def getJ(self, points):
        N = points.shape[0]
        J = np.zeros((N, 3, 6)).astype(np.float32)

        x = points[:, 0]  # Shape (N,)
        y = points[:, 1]  # Shape (N,)
        z = points[:, 2]  # Shape (N,)

        ones = np.ones(N).astype(np.float32)
        zeros = np.zeros(N).astype(np.float32)

        J[:, 0, :] = np.stack([ones, zeros, zeros,
                               zeros, -z, y], axis=1)

        J[:, 1, :] = np.stack([zeros, ones, zeros,
                               z, zeros, -x], axis=1)

        J[:, 2, :] = np.stack([zeros, zeros, ones,
                               -y, x, zeros], axis=1)

        return J

    def getJ_i(self, point):
        J_i = np.zeros((3, 6))
        x = point[0]
        y = point[1]
        z = point[2]
        J_i[0] = np.array([1, 0, 0,   0, -z,  y])
        J_i[1] = np.array([0, 1, 0,   z,  0, -x])
        J_i[2] = np.array([0, 0, 1,  -y,  x,  0])


        return J_i

    def getB(self, J_i, e_i):
        # return J_i.T @ e_i
        b_i = np.zeros((6, ))
        for i in range(6):
            sum_val = 0.0
            for k in range(3):
                sum_val += J_i[k][i] * e_i[k]
            b_i[i] = sum_val


        """b_i_test = J_i.T @ e_i
        if not np.array_equal(b_i_test, b_i):
            print("B ERROR")"""
        return b_i





    def applyIcpStep(self, np_points, R, t):
        # np_points = np_points - mean# ??!
        return (R @ np_points.T).T + t # + mean


    def zipPointsToLines(self, points1, points2):
        minl = min(len(points1), len(points2)) - 1
        zipped_arr = np.empty((2 * minl, 3), dtype=points1.dtype)
        zipped_arr[0::2] = points1[:minl]  # Even indices
        zipped_arr[1::2] = points2[:minl]  # Odd indices
        return zipped_arr


    def uniform_optimal_icp(self, grid1, grid2):
        mean_ref = np.mean(grid1, axis=0)
        reference = grid1 - mean_ref

        mean_other = np.mean(grid2, axis=0)
        other = grid2 - mean_other

        W = reference.T @ other

        U, S, Vh = np.linalg.svd(W)

        R = U @ Vh
        t = mean_ref - R @ mean_other # mean_other # t = mean_ref - R @ mean_other without rotating the mean, points have to be normalized to origin before rotation in apply icp

        return R, t

    def sphereProjectPont(self, point, origin):
        lidar_tall = 0.254
        lidar_off = np.array([0, lidar_tall / 2, 0])

        magnitude = np.linalg.norm(point - origin - lidar_off)
        return (point - origin - lidar_off) / magnitude

    def sphereProjectPoints(self, np_points, origin):
        lidar_tall = 0.254
        lidar_off = np.array([0, lidar_tall / 2, 0])
        magnitudes = np.linalg.norm(np_points - origin - lidar_off, axis=1)
        return (np_points - origin - lidar_off) / magnitudes[:, np.newaxis]

    def getScanLines(self, pts, origin):  # on the projected points

        pts = self.sphereProjectPoints(pts, origin)


        angs = np.arctan2(pts[:,2], pts[:, 0])
        mask = angs < 0
        angs[mask] += np.pi * 2
        prev = angs[:len(angs) - 1]
        next = angs[1:]
        diffs = np.abs(next - prev)
        wraps = diffs > 6
        inds = np.arange(len(pts) - 1)
        lines = inds[wraps]
        scan_lines = []

        begin = 0
        for l in lines:
            scan_lines.append((begin, l))
            begin = l + 1

        """index = 0
        end_index = 0

        while end_index < len(pts) - 1:
            end_index = self.getScanLineFrom(pts, index)
            scan_lines.append((index, end_index))
            index = end_index + 1

        # print("scans out of algo: " + str(len(scan_lines)))"""
        return scan_lines



    def dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def getScanLineFrom(self, np_points, index):
        def ang(np_point):
            an = atan2(np_point[2], np_point[0])
            return an if an >= 0 else an + 2 * np.pi

        lp = len(np_points) - 1

        check = index
        prev_ang = ang(np_points[check])

        ang_sum = 0
        while ang_sum < np.pi * 2:
            if check == lp:
                return check

            check += 1
            check_ang = ang(np_points[check])
            if check_ang < 0.5:
                a = 0

            diff = abs(prev_ang - check_ang)
            if diff > 6.1:  # ~2pi wrap around
                diff = abs(check_ang - np.pi * 2 - prev_ang)
            prev_ang = check_ang

            ang_sum += diff

        return check - 1


    def getUniformGrid(self, pointsPerAxis):
        num_points = pointsPerAxis  # Adjust as needed

        x = np.linspace(-1, 1, num_points)
        y = np.linspace(-1, 1, num_points)
        z = np.linspace(-1, 1, num_points)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        return np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    def getUniformGridSurface(self, pointsPerAxis):
        num_points = pointsPerAxis

        x = np.linspace(-1, 1, num_points)
        y = np.linspace(-1, 1, num_points)
        z = np.linspace(-1, 1, num_points)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()

        mask = (X == -1) | (X == 1) | (Y == -1) | (Y == 1) | (Z == -1) | (Z == 1)

        return np.vstack([X[mask], Y[mask], Z[mask]]).T

    def getUniformShape(self):
        g1 = self.getUniformGrid(10)
        g2 = self.getUniformGrid(10) + np.array([1, 0, 0]) * 5
        g3 = self.getUniformGrid(10) + np.array([0, 1, 0]) * 5
        g4 = self.getUniformGrid(10) + np.array([1, 1, 1]) * 5

        return np.vstack([g1, g2, g3, g4])



##################################################### legacy KDtree

    def getNearestNeighbours(self, kdTree_pc1, pc1, pc2):
        distances, closestIndexes = kdTree_pc1.query(pc2)
        return pc1[closestIndexes], pc2, distances

    def getKDTree(self, np_points):
        return KDTree(np_points)

    def icp_step_KD(self, kdTree_pc1, pc1, pc2):

        reference, other, dists = self.getNearestNeighbours(kdTree_pc1, pc1, pc2) # this takes the longest of all the operations


        percent_of_distances = 0.8
        taken = int(len(dists) * percent_of_distances)

        sorted_indexes = np.argsort(dists)
        reference, other, distances = reference[sorted_indexes][:taken], other[sorted_indexes][:taken], dists[sorted_indexes][:taken]

        mean_ref = np.mean(reference, axis=0)
        reference = reference - mean_ref

        mean_other = np.mean(other, axis=0)
        other = other - mean_other

        W = reference.T @ other

        U, S, Vh = np.linalg.svd(W)

        R = U @ Vh
        t = mean_ref - R @ mean_other

        return R, t

    def align_KD(self, reference_points, points_to_align):

        kdTreeRef = self.getKDTree(reference_points)
        reference_grid = self.getUniformGrid(10)

        for i in range(20):
            start = time.time()

            R, t = self.icp_step_KD(kdTreeRef, reference_points, points_to_align)
            print(time.time() - start)
            points_to_align = self.applyIcpStep(points_to_align, R, t)
            reference_grid = self.applyIcpStep(reference_grid, R, t)

        R_opt, t_opt = self.uniform_optimal_icp(reference_grid, self.getUniformGrid(10))

        return points_to_align, R_opt, t_opt

    def connectPointsInOrder(self, np_points):
        repeated = np.empty((2 * (len(np_points) - 1), 3), dtype=np_points.dtype)
        repeated[0::2] = np_points[:-1]  # p1, p2, p3, ...
        repeated[1::2] = np_points[1:]  # p2, p3, p4, ...

        return repeated

    def get_plane_grid(self, points_per_axis, y_value=0, gridsize = 10.0):
        num_points = points_per_axis  # Adjust as needed

        x = np.linspace(-1, 1, num_points) * gridsize
        z = np.linspace(-1, 1, num_points) * gridsize

        X, Z = np.meshgrid(x, z, indexing='ij')

        Y = np.full_like(X, y_value)

        return np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T