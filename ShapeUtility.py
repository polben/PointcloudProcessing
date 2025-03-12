import numpy as np


class ShapeUtility:



    @staticmethod
    def sphere(pos, radius):

        half_side = radius
        num_points_per_axis = 5  # Adjust this value for more or fewer points
        focus = pos

        # Generate linearly spaced points for each axis
        x_coords = np.linspace(focus[0] - half_side, focus[0] + half_side, num_points_per_axis)
        y_coords = np.linspace(focus[1] - half_side, focus[1] + half_side, num_points_per_axis)
        z_coords = np.linspace(focus[2] - half_side, focus[2] + half_side, num_points_per_axis)

        # Create a meshgrid for the points in 3D
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords)

        # Combine into a single array of points (N x 3)
        spoints = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        spoints = spoints[np.linalg.norm(spoints - focus, axis=1) < radius]
        # spoints = spoints[np.linalg.norm(spoints - focus, axis=1) > radius * 0.95]

        return spoints

    @staticmethod
    def createSpheres(positions, radius):
        spheres = np.array([])
        for p in positions:
            sphere = ShapeUtility.sphere(p, radius)
            if spheres.any():
                spheres = ShapeUtility.np_append(spheres, sphere)
            else:
                spheres = sphere

        return spheres

    @staticmethod
    def np_append(points, to_append):
        points = np.concatenate((points, to_append), axis=0)
        return points

    @staticmethod
    def line(point1, point2, density):
        dir = point2 - point1
        dist = np.linalg.norm(dir)
        dir = dir / dist

        inds = np.linspace(0, dist, density)

        points = point1 + np.outer(dir, inds).T

        return points

    @staticmethod
    def square(center, sidelen, density):
        h2 = sidelen / 2

        x2 = np.array([h2, 0, 0])
        y2 = np.array([0, h2, 0])

        v1 = center - x2 - y2
        v2 = center + x2 - y2
        v3 = center - x2 + y2
        v4 = center + x2 + y2

        # lines along x
        plane = ShapeUtility.line(v1, v2, density)
        plane = ShapeUtility.np_append(plane, ShapeUtility.line(v3, v4, density))
        plane = ShapeUtility.np_append(plane, ShapeUtility.line(v1, v3, density))
        plane = ShapeUtility.np_append(plane, ShapeUtility.line(v2, v4, density))

        return plane

    @staticmethod
    def box(center, sidelen, density):

        h2 = sidelen / 2
        x2 = np.array([h2, 0, 0])
        y2 = np.array([0, h2, 0])
        z2 = np.array([0, 0, h2])

        # verticies along x
        v1 = center - x2 - y2 - z2
        v2 = center + x2 - y2 - z2
        v3 = center - x2 + y2 - z2
        v4 = center + x2 + y2 - z2

        v5 = center - x2 - y2 + z2
        v6 = center + x2 - y2 + z2
        v7 = center - x2 + y2 + z2
        v8 = center + x2 + y2 + z2


        # lines along x
        box = ShapeUtility.line(v1, v2, density)
        box = ShapeUtility.np_append(box, ShapeUtility.line(v3, v4, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v5, v6, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v7, v8, density))

        # lines along y
        box = ShapeUtility.np_append(box, ShapeUtility.line(v1, v3, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v2, v4, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v5, v7, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v6, v8, density))

        # lines along z
        box = ShapeUtility.np_append(box, ShapeUtility.line(v1, v5, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v2, v6, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v3, v7, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v4, v8, density))


        return box

    @staticmethod
    def tallbox(center, sidelen, height, density):

        h2 = sidelen / 2
        x2 = np.array([h2, 0, 0])
        y2 = np.array([0, h2, 0])
        z = np.array([0, 0, height])

        # verticies along x
        v1 = center - x2 - y2
        v2 = center + x2 - y2
        v3 = center - x2 + y2
        v4 = center + x2 + y2

        v5 = center - x2 - y2 + z
        v6 = center + x2 - y2 + z
        v7 = center - x2 + y2 + z
        v8 = center + x2 + y2 + z

        # lines along x
        box = ShapeUtility.line(v1, v2, density)
        box = ShapeUtility.np_append(box, ShapeUtility.line(v3, v4, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v5, v6, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v7, v8, density))

        # lines along y
        box = ShapeUtility.np_append(box, ShapeUtility.line(v1, v3, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v2, v4, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v5, v7, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v6, v8, density))

        # lines along z
        box = ShapeUtility.np_append(box, ShapeUtility.line(v1, v5, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v2, v6, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v3, v7, density))
        box = ShapeUtility.np_append(box, ShapeUtility.line(v4, v8, density))

        return box

    @staticmethod
    def concat(listOfShapes):
        totalLen = 0
        for s in listOfShapes:
            totalLen += len(s)

        arr_result = np.zeros((totalLen, 3))
        last = 0
        for shape in listOfShapes:  # Suppose `pointclouds` is a list of arrays
            len_current = len(shape)
            arr_result[last:last + len_current] = shape
            last += len_current

        return arr_result

