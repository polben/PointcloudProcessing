import numpy as np
import pandas as pd
from plyfile import PlyData

class PointcloudHandler:


    def __init__(self, renderer):
        self.renderer = renderer
        self.loaded_pointclouds = []
        self.current_cloud = 0

    def savePointcloudPly(self, path_to_save_to, ui=None):
        self.renderer.MemoryManager.fullDefragment()
        memory_data = self.renderer.MemoryManager.points

        num_points = self.renderer.MemoryManager.getMaxPointIndex()
        if ui is not None:
            ui.appendConsole("Saving...")
        # Create header
        header = '\n'.join([
            'ply',
            'format ascii 1.0',
            f'element vertex {num_points}',
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header\n'
        ])

        rgb_scaled = (memory_data[:, 3:6][:num_points] * 255).clip(0, 255).astype(np.uint8)

        xyz = memory_data[:, 0:3][:num_points]
        final_data = np.hstack((xyz, rgb_scaled))

        with open(path_to_save_to + ".ply", 'w') as f:
            f.write(header)

        CHUNK_SIZE = 1000000
        for i in range(0, num_points, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, num_points)

            xyz = memory_data[i:end, 0:3]
            rgb = (memory_data[i:end, 3:6] * 255).clip(0, 255).astype(np.uint8)
            chunk_data = np.hstack((xyz, rgb))

            with open(path_to_save_to + ".ply", 'a') as fa:
                np.savetxt(fa, chunk_data, fmt="%.6f %.6f %.6f %d %d %d")

            if ui is not None:
                ui.appendConsole("Saving... [" + str(int(i/float(num_points) * 100)) + "%]" )

        if ui is not None:
            ui.appendConsole("Saved!")


    def tryReadAsciPly(self, path, filename, ui=None):
        try:
            num_points = None
            with open(path + "/" + filename, 'r') as f:
                header_lines = 0
                for line in f:
                    header_lines += 1
                    line = line.strip()
                    if line.startswith('element vertex'):
                        # e.g. "element vertex 1234567"
                        num_points = int(line.split()[2])
                    if line == 'end_header':
                        break

                if num_points is None:
                    raise ValueError("PLY header is missing 'element vertex' line")
        except UnicodeDecodeError as e:
            return None

        col_names = ['x', 'y', 'z', 'r', 'g', 'b']

        # Read data in chunks
        pointcount = 0
        last = 0
        chunk_size = 200000
        points = np.empty((num_points, 6))
        for chunk in pd.read_csv(
                path + "/" + filename,
                sep='\s+',
                skiprows=header_lines,
                names=col_names,
                dtype={'x': float, 'y': float, 'z': float, 'r': int, 'g': int, 'b': int},
                chunksize=chunk_size
        ):
            pointcount += len(chunk)
            data = chunk[['x', 'y', 'z', 'r', 'g', 'b']].values
            points[last:pointcount] = data
            last = pointcount
            if ui is not None:
                ui.appendConsole(f"Loading... [" + str(int(pointcount / float(num_points) * 100)) + "%]")

            """# XYZ
            #points = chunk[['x', 'y', 'z']].values

            # Normalize RGB to [0,1]
            rgb_norm = chunk[['r', 'g', 'b']].clip(0, 255).astype(np.float32) / 255.0
            colors = rgb_norm.values


            self.renderer.addPoints(points, colors)"""

        points[:, 3:] = points[:, 3:].clip(0, 255).astype(np.float32) / 255.0
        return points

    def tryReadBinPly(self, path, filename, ui=None):



        plydata = PlyData.read(path + "/" + filename)


        vertex_data = plydata.elements[0].data
        num_points = len(vertex_data)

        xyz = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)

        if 'red' in vertex_data.dtype.names:
            rgb = np.stack([
                vertex_data['red'],
                vertex_data['green'],
                vertex_data['blue']
            ], axis=-1).astype(np.float32) / 255.0
        else:
            rgb = np.zeros_like(xyz)

        return np.hstack((xyz, rgb)).astype(np.float32)


    def normalizeBigPointcloud(self, points, filename, ui=None):

        if ui is not None:

            xyz = points[:, :3]
            mmin, mmax = np.min(xyz, axis=0), np.max(xyz, axis=0)
            extent = mmax - mmin
            if extent[0] > 200 or extent[1] > 200 or extent[2] > 200:
                if ui.yesNoPopup("This pointcloud is huge! (" + filename + ")", "Would you like to normalize?"):
                    maxwidth = np.max(extent)
                    scale = 10.0 / maxwidth
                    points[:, :3] *= scale


    def loadPointcloud(self, path, filename, ui=None):


        points = None

        points = self.tryReadAsciPly(path, filename, ui)
        if points is None:
            points = self.tryReadBinPly(path, filename, ui)

        if points is not None:
            self.normalizeBigPointcloud(points, filename, ui)
            self.loaded_pointclouds.append(points)
            return True

        return False


    def getCloud(self, ind):
        if len(self.loaded_pointclouds) == 0:
            return None, None

        return self.loaded_pointclouds[ind][:, :3], self.loaded_pointclouds[ind][:, 3:]





