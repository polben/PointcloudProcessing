import threading
import time
from queue import Queue

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from PointcloudAlignment import PointcloudAlignment
from Voxelizer import Voxelizer


class OutlierFilter:


    def __init__(self, icpContainer, dbscan_eps=0.2, dbscan_minpt=20, min_cluster_points = 50, do_threading=False, do_print=False):
        self.eps = dbscan_eps
        self.minpt = dbscan_minpt
        if do_threading:
            self.scan = DBSCAN(eps=self.eps, min_samples=self.minpt, n_jobs=4)
        else:
            self.scan = DBSCAN(eps=self.eps, min_samples=self.minpt)

        self.icpContainer = icpContainer
        self.current_noise = None
        self.min_cluster_size = min_cluster_points
        self.do_print = do_print


    def getOutlierIndexes(self, reference_pointcloud, prev_frame, renderer=None, use_threads=True):

        total_outlier_time = time.time()

        outliers_ref = Voxelizer.voxelGroundFilter(reference_pointcloud) # reference_pointcloud[~self.get_ground_mask(reference_pointcloud, 0.2)] # self.temporalClosestPointOutliers(reference_pointcloud, next_frame)
        prev_frame = Voxelizer.voxelGroundFilter(prev_frame, 100, 100)

        # outliers_next = Voxelizer.voxelGroundFilter(next_frame) # next_frame[~self.get_ground_mask(next_frame, 0.2)] # self.temporalClosestPointOutliers(next_frame, reference_pointcloud)
        # cluster once, do aabbs on next frame
        clusters_ref = self.getClusters(outliers_ref, baseline=True)
        # clusters_next = self.getClusters(outliers_next)
        if renderer is not None:
            vis = []
            for c in clusters_ref:
                vis.append(renderer.addPoints(c, self.randcolor()))

            time.sleep(1)
            for v in vis:
                renderer.freePoints(v)


        start = time.time()
        cluster_pairs = []
        for c in clusters_ref:
            if len(c) > self.min_cluster_size:
                minc, maxc = self.getAABBofCluster(c, down_extension=1.0, side_expansion=0.3)
                corresponding_section = prev_frame[self.insideAABB(prev_frame, minc, maxc)]
                # corresponding_section = Voxelizer.voxelGroundFilter(corresponding_section, 20, 20)
                correspondence_tree = KDTree(corresponding_section)
                if len(corresponding_section) > self.min_cluster_size:
                    cluster_pairs.append((correspondence_tree, corresponding_section, c))

        if self.do_print:
            print("cross section time: " + str(time.time() - start))
            print("total clusters: " + str(len(cluster_pairs)))

        detect_above_kmh = 3
        mps = detect_above_kmh / 3.6
        lidar_timeframe = 0.1

        detection_transition = mps * lidar_timeframe

        start = time.time()

        outlier_clusters = Queue()




        if renderer is not None:
            vis = []
            for c in cluster_pairs:
                vis.append(renderer.addPoints(c[1], self.randcolor()))
                vis.append(renderer.addPoints(c[2], np.array([255,0,0])))

            time.sleep(2)
            for v in vis: renderer.freePoints(v)
            vis = []


        if not use_threads:
            for i in range(0, len(cluster_pairs)):
                tree, corr_section, to_align = cluster_pairs[i]

                transition = self.tryAlignClusters(tree, corr_section, to_align, renderer)
                if transition > detection_transition:
                    if renderer is not None:
                        renderer.addPoints(to_align)
                    outlier_clusters.put(to_align)

        else: # paralell clutser alignment
            threads = []

            for i in range(len(cluster_pairs)):
                tree, corr_section, to_align = cluster_pairs[i]
                thread = threading.Thread(target=self.clusterProcessingTask,
                                          args=(tree, corr_section, to_align, detection_transition, outlier_clusters))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        outlier_clusters = list(outlier_clusters.queue)

        if self.do_print:
            print("cluster alignment time: " + str(time.time() - start))


        start = time.time()
        outlier_mask = np.full(len(reference_pointcloud), False)

        # Compute inside-AABB masks for all clusters
        inside_masks = [
            self.insideAABB(reference_pointcloud, *self.getAABBofCluster(cluster))
            for cluster in outlier_clusters
        ]

        # Combine all masks efficiently
        if inside_masks:
            outlier_mask = np.logical_or.reduce(inside_masks)

        if self.do_print:
            print("final masking" + str(time.time() - start))

        print("filter: " + str(time.time() - total_outlier_time))

        return outlier_mask

    def clusterProcessingTask(self, tree, corr_section, to_align, detection_transition, outlier_clusters):
        transition = self.tryAlignClusters(tree, corr_section, to_align, None)
        if transition > detection_transition:
            outlier_clusters.put(to_align)

    def getAABBofCluster(self, cluster, down_extension=1.0, side_expansion=0.2):
        min_corner, max_corner = np.min(cluster, axis=0), np.max(cluster, axis=0)

        # Extend downwards in Z
        min_corner[1] -= down_extension
        max_corner[1] += down_extension

        # Expand X and Y by 1%
        expansion = (max_corner - min_corner)[[0, 2]] * side_expansion
        min_corner[[0,2]] -= expansion
        max_corner[[0,2]] += expansion

        return min_corner, max_corner


    def insideAABB(self, points, min_corner, max_corner):

        return np.all((points >= min_corner) & (points <= max_corner), axis=1)


    def temporalClosestPointOutliers(self, reference_cloud, next_cloud):

        ref_objs = reference_cloud[~self.get_ground_mask(reference_cloud, 0.2)]
        next_objs = next_cloud[~self.get_ground_mask(next_cloud, 0.2)]

        tree = KDTree(next_objs)
        dists, corrs = tree.query(ref_objs, k = 5)
        dists = np.sum(dists, axis=1)

        p_dists = np.linalg.norm(ref_objs, axis=1)
        maxd = np.max(p_dists)
        weights = 1.0 / np.pow(p_dists / maxd, 2.0)

        dists = dists * weights

        mean = np.mean(dists)
        medi = np.median(dists)

        outmask = dists > (mean + medi) / 2

        return ref_objs[outmask]

    def getClusters(self, points, baseline = False):



        start = time.time()
        labels = self.scan.fit_predict(points)

        if self.do_print:
            print("clustering time: " + str(time.time() - start))



        indexes = np.arange(len(points))

        unique_labels = np.unique(labels)
        clusters = []

        for i in range(len(unique_labels)):
            label = unique_labels[i]
            if label == -1:  # noise
                if baseline:
                    matching_labels = labels == label
                    point_indexes = indexes[matching_labels]
                    cluster = points[point_indexes]
                    self.current_noise = cluster
                continue

            matching_labels = labels == label
            point_indexes = indexes[matching_labels]
            cluster = points[point_indexes]


            clusters.append(cluster)





        return clusters

    def getClusterCenters(self, clusters):
        centers = []
        for c in clusters:
            centers.append(np.mean(c, axis=0))

        return centers


    def getAverageDisplacement(self, points, same_points):
        diff = points - same_points
        diff[:, 1] = 0
        d = np.linalg.norm(diff, axis=1)
        return np.mean(d)

    def alignPoint(self, tree_ref, reference_section, cluster_to_check, renderer=None):
        try:
            vis = None
            r_vis = None
            if renderer is not None:
                r_vis = renderer.addPoints(reference_section, np.array([255, 0, 0]))


            original_others = cluster_to_check



            for i in range(10):
                t, R = OutlierFilter.mini_ls_icp(tree_ref, reference_section, cluster_to_check)
                cluster_to_check = (R @ cluster_to_check.T).T - t


                if renderer is not None:
                    if vis is not None:
                        renderer.freePoints(vis)
                    vis = renderer.addPoints(cluster_to_check)
                    time.sleep(0.1)

            if renderer is not None:
                renderer.freePoints(vis)
                renderer.freePoints(r_vis)

            return self.getAverageDisplacement(original_others, cluster_to_check)

        except np.linalg.LinAlgError as e:
            return 2e20


    def tryAlignClusters(self, tree, reference_section, cluster_to_align, renderer=None):



        if renderer is not None:
            v1 = renderer.addPoints(reference_section, self.randcolor())
            time.sleep(2)
            renderer.freePoints(v1)




        return self.alignPoint(tree, reference_section, cluster_to_align, renderer)

    def getClusterPairs(self, centers_ref, centers_other, k=5):
        tree_other = KDTree(centers_other)

        dists, corrs = tree_other.query(centers_ref, k=k)

        pairs = []
        for i in range(len(centers_ref)):
            pairs.append((i, corrs[i]))


        return pairs




    def get_ground_mask(self, points, threshold):
        plane_normal = np.array([0, 1, 0])
        lidar_height = 1.73
        lidar_offset = np.array([0, -lidar_height, 0])

        a, b, c = plane_normal
        x0, y0, z0 = lidar_offset

        # Compute the plane's d constant
        d = -(a * x0 + b * y0 + c * z0)

        # Compute distances of all points to the plane
        distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.linalg.norm(plane_normal)

        # Filter points that are farther than the threshold
        return distances < threshold



    @staticmethod
    def mini_ls_icp(tree_ref, ref_points, other_points):

        dists, corrs = tree_ref.query(other_points)
        aligned = ref_points[corrs]
        to_align = other_points


        Hs, Bs = OutlierFilter.getHsBsNumpy(aligned, to_align)

        H, b = OutlierFilter.sumHsBsNumpy(Hs, Bs)


        delta_x = np.linalg.solve(H, -b)


        return delta_x[:3], PointcloudAlignment.rotation(delta_x[3:])

    @staticmethod
    def getHsBsNumpy(aligned, to_align):
        J = OutlierFilter.getJ(to_align)
        J_T = J.swapaxes(1, 2)
        Hs = np.matmul(J_T, J)
        e = (aligned - to_align)[:, :, np.newaxis]
        Bs = np.matmul(J_T, e).squeeze(-1)
        return Hs, Bs

    @staticmethod
    def getJ(points):
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

    @staticmethod
    def sumHsBsNumpy(Hs, Bs):
        H = np.sum(Hs, axis=0)
        b = np.sum(Bs, axis=0)
        return H, b

    @staticmethod
    def mini_ls_icp_plane(tree_ref, ref_points, ref_normals, other_points):
        dists, corrs = tree_ref.query(other_points)
        aligned = ref_points[corrs]
        normals = ref_normals[corrs]  # Get the reference normals for each point
        to_align = other_points

        # Compute Hs and Bs for point-to-plane ICP
        H, B = OutlierFilter.getHsBsNumpyPointToPlane(aligned, normals, to_align)


        delta_x = np.linalg.solve(H, -B)

        return delta_x[:3], PointcloudAlignment.rotation(delta_x[3:])


    @staticmethod
    def getHsBsNumpyPointToPlane(aligned, normals, to_align):
        J = OutlierFilter.getJ_plane(to_align, normals)  # Modify to include normals

        Hs = J[:, :, np.newaxis] * J[:, np.newaxis, :] #, then sum should be same?

        e = np.sum((aligned - to_align) * normals, axis=1)  # Shape (N,)

        Bs = J.T * e

        return np.nansum(Hs, axis=0), np.nansum(Bs, axis=1)

    @staticmethod
    def getJ_plane(points, normals):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]  # Extract coordinates
        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]  # Extract normal components

        # Construct the (N, 6) Jacobian matrix row-wise
        J = np.column_stack([
            nx,  # Translation X
            ny,  # Translation Y
            nz,  # Translation Z
            ny * z - nz * y,  # Rotation X
            nz * x - nx * z,  # Rotation Y
            nx * y - ny * x  # Rotation Z
        ])

        return J  # Shape (N, 6)


    def randcolor(self):
        return np.random.rand(3, )