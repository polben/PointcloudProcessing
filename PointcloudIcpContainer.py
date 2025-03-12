import time
from math import atan2

import numpy as np

from scipy.spatial import KDTree

from PointcloudAlignment import PointcloudAlignment


class PointcloudIcpContainer:

    def __init__(self, computeShader, pointcloudAlignment):
        self.data = {} # filename: original, objects, aligned
        self.filenames = []
        self.compute = computeShader
        self.pointcloudAlignment = pointcloudAlignment


    def addAndAlign(self, filename, original_lidar_points):
        self.filenames.append(filename)
        # print(self.pointcloudAlignment.getPose(filename)[1])

        if len(self.filenames) == 1:
            self.data[filename] = original_lidar_points, self.pointcloudAlignment.getPose(filename)
        else:
            previous_filename = self.filenames[len(self.filenames) - 2]
            prev_aligned_original, prev_translations = self.data[previous_filename]
            previous_pose = self.pointcloudAlignment.getPose(previous_filename)


            # aligned, R_opt, t_opt = self.align_NNS(prev_aligned_original, original_lidar_points, previous_pose)
            aligned, R_opt, t_opt = self.align_NNS_ls(prev_aligned_original, original_lidar_points, previous_pose)

            self.data[filename] = aligned, (R_opt, t_opt)



    def filter_points(self, np_points):

        norm, point = self.icp.estimateGroundIcp(np_points, np.array([0, 0.3, 0]))
        mask = self.remove_points_near_plane(np_points, norm, point, 0.2)
        return mask

        distances = np.linalg.norm(np_points, axis = 1)
        max = np.max(distances) / 4.0
        probabilities = (distances / max) ** 2
        randoms = np.random.rand(len(probabilities))
        indexes = probabilities > randoms

        return np_points[indexes]

        a= 0

    def remove_points_near_plane(self, points, normal, point_on_plane, threshold):

        a, b, c = normal
        x0, y0, z0 = point_on_plane

        # Compute the plane's d constant
        d = -(a * x0 + b * y0 + c * z0)

        # Compute distances of all points to the plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.linalg.norm(normal)

        # Filter points that are farther than the threshold
        return distances > threshold


    def full_ls(self, aligned, to_align, origin, iterations = 20, renderer = None):

        self.initLS(aligned, to_align, origin)

        reference_grid = self.getUniformGrid(10)


        prev_pp = None
        if renderer is not None:
            prev_pp = renderer.addPoints(to_align, np.array([0,0,255]))
            time.sleep(1)

        start = time.time()
        for i in range(iterations):
            t, R = self.iteration_of_ls(to_align)
            R = PointcloudAlignment.rotation(R[0], R[1], R[2])
            to_align = self.apply_iteration_of_ls(to_align, t, R)

            if renderer is not None:
                if prev_pp is not None:
                    renderer.freePoints(prev_pp)
                prev_pp = renderer.addPoints(to_align)
                time.sleep(0.1)

            reference_grid = self.applyIcpStep(reference_grid, R, t)

        print("icp ls: " + str(time.time()-start))

        self.release_ls()
        R_opt, t_opt = self.uniform_optimal_icp(reference_grid, self.getUniformGrid(10))

        return t_opt, R_opt

    def initLS(self, aligned, to_align, origin):

        scan_lines = self.getScanLines(aligned, origin)
        self.compute.prepareLS(aligned, scan_lines, to_align, origin)

    def iteration_of_ls(self, to_align):
        #st = time.time()
        Hs, Bs = self.compute.dispatchLS(to_align)
        #print("compute time: " + str(time.time()-st))

        #st = time.time()
        H = np.sum(Hs, axis=0)
        b = np.sum(Bs, axis=0)
        #print("numpy time: " + str(time.time()-st))

        delta_x = np.linalg.solve(H, -b)

        return delta_x[:3], delta_x[3:]

    def apply_iteration_of_ls(self, to_align, t, R):

        return (R @ to_align.T).T - t

    def release_ls(self):
        self.compute.releaseLS()


    def estimateGroundIcp(self, lidar_points, origin, renderer = None):
        lidar_height = 1.73

        gridsize = 100
        plane = self.get_plane_grid(gridsize, -lidar_height, 30.0)

        if renderer is not None:
            plp = renderer.addPoints(plane)
            time.sleep(2)


        t_opt, R_opt = self.full_ls(lidar_points, plane, origin, 10, renderer)
        plane = (R_opt @ plane.T).T - t_opt

        if renderer is not None:

            renderer.freePoints(plp)
            renderer.addPoints(plane)

        v1 = plane[0] - plane[10]
        v2 = plane[0] - plane[gridsize * 10]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        normal = np.linalg.cross(v1, v2)

        half = int(gridsize / 2)
        point = plane[half * half - half]

        if renderer is not None:
            renderer.setLines([plane[0], plane[10], plane[0], plane[gridsize * 10], point, point + normal * 2])

        return normal, point



    def align_NNS_ls(self, aligned, to_align, current_pose):
        reference_grid = self.getUniformGrid(10)
        pose_t, pose_R = current_pose
        origin = pose_t

        scan_lines = self.getScanLines(aligned, origin)
        self.compute.prepareLS(aligned, scan_lines, to_align, origin)


        for i in range(10):
            start = time.time()
            shader = time.time()
            Hs, Bs = self.compute.dispatchLS(to_align)
            print("shader time:" + str(time.time()-shader))

            H = np.sum(Hs, axis=0)
            b = np.sum(Bs, axis=0)

            delta_x = np.linalg.solve(H, -b)

            t, R = delta_x[:3], delta_x[3:]
            print("compute ls: " + str(time.time()-start))

            R = PointcloudAlignment.rotation(R[0], R[1], R[2])
            to_align = (R @ to_align.T).T - t

            # to_align = self.applyIcpStep(to_align, R, t)
            # reference_grid = self.applyIcpStep(reference_grid, R, t)

        self.compute.releaseLS()

        R_opt, t_opt = self.uniform_optimal_icp(reference_grid, self.getUniformGrid(10))
        return to_align, R_opt, t_opt

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


    def icp_step_NNS_LS(self, aligned, to_align):

        Hs, Bs = self.getHsBsLoop(aligned, to_align)
        H, b = self.sumHsBsLoop(Hs, Bs)



        delta_x = np.linalg.solve(H, -b)

        return delta_x[:3], delta_x[3:]



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

    """def compute_icp_jacobian(self, points):

        Js = []  # List to store individual Jacobian matrices

        for x, y, z in points:
            # Create the Jacobian for this point
            J_i = np.zeros((3, 6))

            # Translation Jacobian (Identity)
            J_i[:, :3] = np.eye(3)

            # Rotation Jacobian (Skew-symmetric matrix)
            J_i[:, 3:] = np.array([
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0]
            ])

            Js.append(J_i)  # Store in the list

        return Js"""

    def align_NNS(self, reference_points, points_to_align, prev_pose):
        reference_grid = self.getUniformGrid(10)
        t, R = prev_pose
        origin = t

        scan_lines = self.getScanLines(reference_points, origin)

        self.compute.prepareNNS(reference_points, scan_lines, points_to_align, origin)


        for i in range(30):
            start = time.time()

            R, t = self.icp_step_NNS(reference_points, points_to_align)
            # print(time.time() - start)

            points_to_align = self.applyIcpStep(points_to_align, R, t)
            reference_grid = self.applyIcpStep(reference_grid, R, t)


        self.compute.releaseNNS()


        R_opt, t_opt = self.uniform_optimal_icp(reference_grid, self.getUniformGrid(10))

        return points_to_align, R_opt, t_opt

    def icp_step_NNS(self, pc1, pc2):

        corresps = self.compute.dispatchNNS(pc2)

        corr_indexes = corresps[:, 0].astype(np.uint32)
        dists = corresps[:, 1].astype(np.float32)
        other = pc2
        reference = pc1[corr_indexes]


        # reference, other, dists = self.getNearestNeighbours(tree, pc1, pc2)

        percent_of_distances = 0.4
        taken = int(len(dists) * percent_of_distances)


        """
        reference, other, dists = self.getNearestNeighbours(kdTree_pc1, pc1, pc2) # this takes the longest of all the operations


        percent_of_distances = 0.8
        taken = int(len(dists) * percent_of_distances)

        sorted_indexes = np.argsort(dists)
        reference, other, distances = reference[sorted_indexes][:taken], other[sorted_indexes][:taken], dists[sorted_indexes][:taken]

        mean_ref = np.mean(reference, axis=0)
        reference = reference - mean_ref

        mean_other = np.mean(other, axis=0)
        other = other - mean_other
        """
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

    def applyIcpStep(self, np_points, R, t):
        # np_points = np_points - mean# ??!
        return (R @ np_points.T).T + t # + mean





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

        scan_lines = []
        index = 0
        end_index = 0

        while end_index < len(pts) - 1:
            end_index = self.getScanLineFrom(pts, index)
            scan_lines.append((index, end_index))
            index = end_index + 1

        # print(len(scan_lines))
        return scan_lines

    def ang(self, np_point):
        an = atan2(np_point[2], np_point[0])
        return an if an >= 0 else an + 2 * np.pi

    def dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def getScanLineFrom(self, np_points, index):

        lp = len(np_points) - 1

        check = index
        prev_ang = self.ang(np_points[check])

        ang_sum = 0
        while ang_sum < np.pi * 2:
            if check == lp:
                return check

            check += 1
            check_ang = self.ang(np_points[check])
            if check_ang < 0.5:
                a = 0

            diff = abs(prev_ang - check_ang)
            if diff > 6.1:  # ~2pi wrap around
                diff = abs(check_ang - np.pi * 2 - prev_ang)
            prev_ang = check_ang

            ang_sum += diff

        return check - 1

    def getNearestNeighbours(self, kdTree_pc1, pc1, pc2):
        distances, closestIndexes = kdTree_pc1.query(pc2)
        return pc1[closestIndexes], pc2, distances

    def getAlignedOriginal(self, filename):

        original, translations = self.data[filename]
        return original

    def getLS(self, points_a, scan_lines, points_b, origin):

        st = time.time()
        self.compute.prepareLS(points_a, scan_lines, points_b, origin)
        print("prep: " + str(time.time()-st))

        st = time.time()
        Hs, Bs = self.compute.dispatchLS(points_b)
        print("dispatch: " + str(time.time()-st))

        self.compute.releaseLS()



        return Hs, Bs



    def getCorrespondencesComputeScan(self, points_a, scan_lines, points_b, origin):

        st = time.time()
        self.compute.prepareNNS(points_a, scan_lines, points_b, origin)
        print("prep: " + str(time.time()-st))

        st = time.time()
        r = 1
        for i in range(r):
            correspondences = self.compute.dispatchNNS(points_b)
        print("dispatches " + str(r) + ": " + str(time.time()-st))

        self.compute.releaseNNS()

        st = time.time()

        corr_indexes = correspondences[:, 0].astype(np.uint32)
        distances = correspondences[:, 1].astype(np.float32)
        print("skinning data: " + str(time.time()-st))
        return corr_indexes, distances



    def getUniformGrid(self, pointsPerAxis):
        num_points = pointsPerAxis  # Adjust as needed

        # Generate linearly spaced values along each axis
        x = np.linspace(-1, 1, num_points)
        y = np.linspace(-1, 1, num_points)
        z = np.linspace(-1, 1, num_points)

        # Create a meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Stack the coordinates into a (N, 3) array
        return np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    def getUniformGridSurface(self, pointsPerAxis):
        num_points = pointsPerAxis

        # Generate linearly spaced values along each axis
        x = np.linspace(-1, 1, num_points)
        y = np.linspace(-1, 1, num_points)
        z = np.linspace(-1, 1, num_points)

        # Create a meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Flatten the arrays
        X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()

        # Select only points on the surface (at least one coordinate must be at an extreme)
        mask = (X == -1) | (X == 1) | (Y == -1) | (Y == 1) | (Z == -1) | (Z == 1)

        # Apply the mask
        return np.vstack([X[mask], Y[mask], Z[mask]]).T

    def getUniformShape(self):
        g1 = self.getUniformGrid(10)
        g2 = self.getUniformGrid(10) + np.array([1, 0, 0]) * 5
        g3 = self.getUniformGrid(10) + np.array([0, 1, 0]) * 5
        g4 = self.getUniformGrid(10) + np.array([1, 1, 1]) * 5

        return np.vstack([g1, g2, g3, g4])



##################################################### legacy KDtree
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

        # Generate linearly spaced values along the x and z axes
        x = np.linspace(-1, 1, num_points) * gridsize
        z = np.linspace(-1, 1, num_points) * gridsize

        # Create a meshgrid for the plane
        X, Z = np.meshgrid(x, z, indexing='ij')

        # The y-coordinate is constant
        Y = np.full_like(X, y_value)

        # Stack the coordinates into a (N, 3) array
        return np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    """
        def getCorrespondecesNaive(self, reference_pc, other_pc):

        other_to_reference = []
        for i in range(len(other_pc)):
            point = other_pc[i]
            closest = self.__findClosestPointNaiveNumpy__(point, reference_pc)
            other_to_reference.append(closest)

        return other_to_reference
    
    
    def getCorrespondencesComputeMorton(self, reference_pc, other_pc):
            ######
        compute_start = time.time()
            ######

        sorted_mortons_ref, sorted_indexes, other_mortons = self.__generateMortonCodesForPointclouds__(reference_pc,
                                                                                                       other_pc)
        original_indexes = np.arange(len(reference_pc))

        sorted_points = reference_pc[sorted_indexes]
        sorted_original_indexes = original_indexes[sorted_indexes]

            ######
        print("morton time: " + str(time.time() - compute_start))
            ######



            ######
        compute_start = time.time()
            ######

        self.compute.setActiveProgram(self.compute.PROGRAM_NEIGHBOUR_MORTON)
        correspondences = self.compute.dispatchNeighbourMorton(sorted_mortons_ref, sorted_points, other_mortons, other_pc)

            ######
        print("compute time: " + str(time.time() - compute_start))
            ######

        return sorted_original_indexes[correspondences]

    def getCorrespondencesMorton(self, reference_pc, other_pc):
        start = time.time()
        sorted_mortons_ref, sorted_indexes, other_mortons = self.__generateMortonCodesForPointclouds__(reference_pc, other_pc)
        print("morton time: " + str(time.time()-start))

        start = time.time()
        original_indexes = np.arange(len(reference_pc))

        sorted_points = reference_pc[sorted_indexes]
        sorted_original_indexes = original_indexes[sorted_indexes]

        asd = True

        other_to_reference = []
        for i in range(len(other_mortons)):
            code = other_mortons[i]
            other_point = other_pc[i]

            start_ref1 = time.time()

            close_index = self.__findClosestPointMorton__(code, sorted_mortons_ref, other_point, sorted_points)

            if asd:
                print("single ref time: " + str(time.time() - start_ref1))

            asd = False


            other_to_reference.append(sorted_original_indexes[close_index])

        print("reference time: " + str(time.time() - start))

        return other_to_reference

    #findClosestPointMorton__(code, sorted_mortons_ref, other_point, sorted_points)
    def __findClosestPointMorton__(self, morton_code, other_codes, point, reference_points):
        length = len(other_codes) - 1
        low = 0
        high = length

        close_index = -1
        while low <= high:
            mid = int(low + (high - low) / 2)

            if other_codes[mid] == morton_code:
                close_index = mid
                break

            if other_codes[mid] < morton_code:
                low = mid + 1
            else:
                high = mid - 1

        if close_index == -1:
            close_index = low
        close_index = np.clip(close_index, 0, len(reference_points) - 1)

        PERCENT_OF_POINTS_TO_CHECK = 5 / 100.
        numPointsCheck = int(len(reference_points) * PERCENT_OF_POINTS_TO_CHECK)

        start = max(0, close_index - numPointsCheck)
        end = min(length, close_index + numPointsCheck)

        dist = np.linalg.norm(point - reference_points[close_index])
        for i in range(start, end):
            reference_point = reference_points[i]

            ddist = np.linalg.norm(point - reference_point)
            if ddist < dist:
                close_index = i
                dist = ddist

        return close_index

    def __getMortonGrid__(self, pc, resolution=1024.0):
        return np.clip(pc * resolution, 0.0, resolution - 1.0)

    def __generateMortonCodesForPointclouds__(self, pc1, pc2):
        n_pc1, n_pc2 = self.__normalizeTwoPointclouds__(pc1, pc2)

        grid1_query = self.__getMortonGrid__(n_pc1)
        grid2_other = self.__getMortonGrid__(n_pc2)



        morton_codes_vector_query = self.__getMortonCodes_vectorized__(grid1_query)
        morton_codes_vector_other = self.__getMortonCodes_vectorized__(grid2_other)

        sorted_indexes = np.argsort(morton_codes_vector_query, axis=0)
        return morton_codes_vector_query[sorted_indexes], sorted_indexes, morton_codes_vector_other

    def __getMortonCodes_vectorized__(self, points: np.ndarray) -> np.ndarray:


        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        x_interleaved = self.__interleaveBits_vectorized__(x)
        y_interleaved = self.__interleaveBits_vectorized__(y) << 1
        z_interleaved = self.__interleaveBits_vectorized__(z) << 2

        return x_interleaved | y_interleaved | z_interleaved

    def __interleaveBits_vectorized__(self, values: np.ndarray) -> np.ndarray:

        values = values.astype(np.uint32)
        values &= np.uint32(0x000003FF)

        values = (values | (values << 16)) & np.uint32(0x030000FF)
        values = (values | (values << 8)) & np.uint32(0x0300F00F)
        values = (values | (values << 4)) & np.uint32(0x030C30C3)
        values = (values | (values << 2)) & np.uint32(0x09249249)

        return values


    

    def __getExtentOfPointcloud__(self, pc):
        min1 = np.min(pc, axis=0)
        max1 = np.max(pc, axis=0)
        return max1 - min1, min1


    def __getExtentOfPointclouds__(self, pc1, pc2):
        min1, max1 = self.__getExtentOfPointcloud__(pc1)
        min2, max2 = self.__getExtentOfPointcloud__(pc2)

        minmin = np.min([min1, min2], axis=0)
        maxmax = np.max([max1, max2], axis = 0)

        return maxmax - minmin, minmin

    def __normalizeTwoPointclouds__(self, pc1, pc2):
        maxextent, minmin = self.__getExtentOfPointclouds__(pc1, pc2)

        return self.__normalizePointcloud__(pc1, maxextent, minmin), self.__normalizePointcloud__(pc2, maxextent, minmin)

    def __findClosestPointNaiveNumpy__(self, point_p, other_points):
        distances = np.sum((other_points - point_p) ** 2, axis=1)
        return np.argmin(distances)

    def __normalizePointcloud__(self, np_points, extent, minmin):
        normalized = ((np_points - minmin) / extent)

        return normalized
    """
"""
    def generateMortonCodesForNormalizedPointcloud(self, pc_normalized, resolution = 1024.0):
        pc_normalized = pc_normalized * resolution
        grid = np.clip(pc_normalized, 0.0, resolution - 1.0)

        morton_codes_vector = self.getMortonCodes_vectorized(grid)

        sorted_indexes = np.argsort(morton_codes_vector, axis=0)
        return morton_codes_vector[sorted_indexes], sorted_indexes

    
    def printBinary(self, morton_codes):
        for morton_code in morton_codes:
            binary_code = format(morton_code, '032b')
            print(binary_code)
            
            
    




def plotDistances(randind, pointcloud, naive_found, morton_found):

    random_point = pointcloud[randind]
    distances = pointcloud - random_point
    distances = np.linalg.norm(distances, axis=1)

    x = np.arange(0, len(pointcloud))

    # Replace this with your actual y-values
    y = distances

    # Create scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, s=10, alpha=0.5, color='blue')  # Adjust size and transparency

    plt.axvline(x=randind, color='red', linestyle='--', linewidth=2, label=f'Split at x={randind}')
    plt.axvline(x=naive_found, color='green', linestyle='-.', linewidth=2, label=f'naive at x={naive_found}')
    plt.axvline(x=morton_found, color='blue', linestyle='--', linewidth=2, label=f'morton at x={morton_found}')

    plt.xlabel('X Values')
    plt.ylabel('Y Values (Your Data)')
    plt.title('Scatter Plot of Given Distribution')
    plt.grid(True)

    # Show the plot
    plt.show()
"""
