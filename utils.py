import open3d as o3d
import numpy as np
from tenacity import sleep
import time

UNDEFINED = "UNDEFINED"
EARTH_RADIUS = 6371000  # Radius of Earth in meters
METERS_PER_DEGREE_LAT = 111139  # Approx meters per degree latitude

def lat_lon_alt_to_meters(lat1, lon1, alt1, lat2, lon2, alt2):
    # Calculate latitude and longitude distances in meters
    delta_lat = (lat2 - lat1) * METERS_PER_DEGREE_LAT
    delta_lon = (lon2 - lon1) * METERS_PER_DEGREE_LAT * np.cos(np.radians(lat1))
    delta_alt = alt2 - alt1  # Altitude is already in meters

    return np.array([delta_lon, delta_lat, delta_alt])

def translateToWorldCoords(np_points, oxtsDataReader, framekey):
    frameox = oxtsDataReader.getOx(framekey)
    origin = oxtsDataReader.getWorldOrigin()
    currentCoord = frameox.getWorldPos()

    delta = lat_lon_alt_to_meters(currentCoord[0], currentCoord[1], currentCoord[2], origin[0], origin[1], origin[2])

    np_points = np_points + np.array([delta[0], delta[2], delta[1]])
    return np_points

def alignToCamera(np_points, rotation):
    points = rotation @ np_points.T
    return points.T

def visPoints(lidarDataReader):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=800, width=1280)



    np_points = lidarDataReader.getPoints(lidarDataReader.getCurrentName())
    colors2 = lidarDataReader.getDepth(lidarDataReader.getCurrentName())


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(colors2)


    vis.get_render_option().load_from_json("renderopt.json")

    vis.add_geometry(pcd)


    fps = 30

    while vis.poll_events():
        vis.update_geometry(pcd)
        vis.update_renderer()
        sleep(1 / fps)

    vis.destroy_window()

def visFrame(lidarDataReader, camera):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=800, width=1280)


    pcd = o3d.geometry.PointCloud()


    points, depth = camera.project(lidarDataReader, np.array([0.5, 0, 0.5]))  # y roll, x jaw, z pitch

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(depth)
    # vis.get_render_option().save_to_json("renderopt.json")
    vis.get_render_option().load_from_json("renderopt.json")

    vis.add_geometry(pcd)

    fps = 30

    while vis.poll_events():
        vis.update_geometry(pcd)
        vis.update_renderer()
        sleep(1 / fps)

    vis.destroy_window()


def projectFrames(lidarDataReader, camera):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=800, width=1280)


    pcd = o3d.geometry.PointCloud()

    firstname = lidarDataReader.getNextName()

    points, depth = camera.project(lidarDataReader, np.array([0.5, 0, 0.5]))
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(depth)



    # vis.get_render_option().save_to_json("renderopt.json")
    vis.get_render_option().load_from_json("renderopt.json")

    vis.add_geometry(pcd)

    while vis.poll_events():

        render_start = time.time()

        nextFileName = lidarDataReader.getNextName()
        print(nextFileName)


        points, depth = camera.project(lidarDataReader, np.array([0.5, 0, 0.5]))
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(depth)

        vis.update_geometry(pcd)
        vis.update_renderer()

        render_end = time.time()
        elapsed_time_ms = int(render_end - render_start) * 1000


        for i in range(max(0, int(lidarDataReader.getNextWait()) - elapsed_time_ms)):
            sleep(1 / 1500.0)
            if not vis.poll_events():
                break

    vis.destroy_window()
