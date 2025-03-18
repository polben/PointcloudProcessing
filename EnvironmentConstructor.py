import numpy as np



class EnvironmentConstructor:


    def __init__(self, renderer, oxtsdatareader, lidardatareader, icpcontainer):
        self.renderer = renderer
        self.oxts = oxtsdatareader
        self.lidar = lidardatareader
        self.icp = icpcontainer

        self.framekeys = self.lidar.getfilenames()
        self.frameindex = 0

        self.rotation = np.eye(3)
        self.time = 0
        self.reference_points = None
        self.prev_position = None
        self.prev_velocity = None
        self.prev_rotation = None
        self.angular_velocity = None
        self.prev_time = None

        self.red = np.array([255, 0, 0])
        self.green = np.array([0, 255, 0])
        self.blue = np.array([0, 0, 255])
        self.white = np.array([255, 255, 255])
        self.colors = [self.red, self.green, self.blue, self.white]

        self.base_horizontal_rotation = np.eye(3)
        self.base_vertical_rotation = np.eye(3)

        self.count = 0
        self.reset_origin_treshold = 5

        self.curr_r_opt = None
        self.curr_t_opt = None



    def getNextFrameData(self, offset):
        key = self.framekeys[self.frameindex + offset]

        lidar_points = self.lidar.getPoints(key)
        oxts = self.oxts.getOx(key)

        self.frameindex += 1
        if self.frameindex == len(self.framekeys):
            self.frameindex = 0

        return lidar_points, oxts

    def calculateTransition_icp(self, current_lidar, oxts=None, point_to_plane=True, debug=True, iterations=20):

        points, colors = current_lidar
        if self.reference_points is None:

            self.prev_position = np.array([0, 0, 0])
            self.prev_velocity = np.array([0, 0, 0])
            self.angular_velocity = np.eye(3)
            self.prev_rotation = np.eye(3)

            self.reference_points = points
            self.renderer.addPoints(points, colors)
            self.renderer.addPoints([self.prev_position], self.red)


        else:

            self.count+=1
            # estimated_rotation = self.angular_velocity @ self.prev_rotation
            estimated_rotation = self.angular_velocity @ self.prev_rotation
            translated_points = (estimated_rotation @ points.T).T + self.prev_position + self.prev_velocity


            if debug:
                if point_to_plane:
                    t_opt, R_opt = self.icp.full_pt2pl(self.reference_points, translated_points, self.prev_position, iterations=iterations, renderer=self.renderer)
                else:
                    t_opt, R_opt = self.icp.full_pt2pt(self.reference_points, translated_points, self.prev_position, iterations=iterations, renderer=self.renderer)
            else:
                if point_to_plane:
                    t_opt, R_opt = self.icp.full_pt2pl(self.reference_points, translated_points, self.prev_position,
                                                       iterations=iterations)
                else:
                    t_opt, R_opt = self.icp.full_pt2pt(self.reference_points, translated_points, self.prev_position,
                                                       iterations=iterations)

            corrected_lidar_points = (R_opt @ translated_points.T).T - t_opt

            if self.count > self.reset_origin_treshold:
                print("reference update")
                self.reference_points = corrected_lidar_points
                self.count = 0

            translation_delta = ((R_opt @ np.array([0, 0, 0]).T).T - t_opt) - np.array([0, 0, 0])


            curr_pos = self.prev_position
            self.prev_position = self.prev_position + self.prev_velocity + translation_delta
            self.prev_velocity = self.prev_position - curr_pos

            curr_rot = self.prev_rotation
            self.prev_rotation = R_opt @ self.angular_velocity @ self.prev_rotation
            self.angular_velocity = self.prev_rotation @ np.transpose(curr_rot)

            self.renderer.addPoints([self.prev_position], self.red)
            self.renderer.addPoints(corrected_lidar_points, colors)
            # time.sleep(1)

    def getDeltaVelocity(self, current_velocity, current_rotation, current_time):
        delta_time = (current_time - self.prev_time).total_seconds()
        current_velocity *= delta_time
        delta_current_velocity = (current_rotation @ current_velocity.T).T

        return delta_current_velocity

    def getTranslationDelta(self, r_opt, t_opt):
        translated_to = (r_opt @ np.array([0,0,0]).T).T - t_opt
        return translated_to - np.array([0, 0, 0])

    def applyOptimalTranslation(self, r_opt, t_opt, points):
        return (r_opt @ points.T).T - t_opt

    def getCurrentOxtsData(self, current_oxts):
        return -current_oxts.getVelocity(), current_oxts.getTrueRotation(np.eye(3)), current_oxts.getYawRotation(np.eye(3)), current_oxts.getTime()

    def cullColorPoints(self, points, colors):
        defaultColor = self.lidar.DEFAULT_COLOR
        mask = np.all(colors != defaultColor, axis=1)
        return points[mask], colors[mask]

    def setupTransitions(self, current_oxts):
        self.prev_position = np.array([0, 0, 0])
        self.prev_velocity = np.array([0, 0, 0])
        self.angular_velocity = np.eye(3)
        self.prev_rotation = np.eye(3)

        self.prev_time = current_oxts.getTime()


        self.curr_r_opt = np.array([0, 0, 0])
        self.curr_r_opt = np.eye(3)

    def calculateTransition_imu(self, current_lidar, current_oxts, point_to_plane=True, debug=False, iterations=20, cullColors=False):

        points, colors = current_lidar
        current_velocity, current_horz_rot, current_vert_rot, current_time = self.getCurrentOxtsData(current_oxts=current_oxts)
        current_horz_rot = np.linalg.inv(current_horz_rot)

        current_rotation = current_vert_rot @ current_horz_rot

        if self.reference_points is None:

            self.setupTransitions(current_oxts=current_oxts)
            self.reference_points = (current_rotation @ points.T).T

            self.renderer.addPoints([self.prev_position], self.red)

            if cullColors:
                culled_points, culled_colors = self.cullColorPoints(self.reference_points, colors)
                self.renderer.addPoints(culled_points, culled_colors)
            else:
                self.renderer.addPoints(self.reference_points, colors)

        else:
            self.count += 1

            delta_velocity = self.getDeltaVelocity(current_velocity=current_velocity,
                                                      current_time=current_time,
                                                      current_rotation=current_rotation)
            estimated_position = self.prev_position + delta_velocity

            aligned_points = (current_rotation @ points.T).T + estimated_position


            if debug:
                if point_to_plane:
                    t_opt, R_opt = self.icp.full_pt2pl(self.reference_points, aligned_points, self.prev_position, iterations=iterations, renderer=self.renderer)
                else:
                    t_opt, R_opt = self.icp.full_pt2pt(self.reference_points, aligned_points, self.prev_position, iterations=iterations, renderer=self.renderer)
            else:
                if point_to_plane:
                    t_opt, R_opt = self.icp.full_pt2pl(self.reference_points, aligned_points, self.prev_position,
                                                       iterations=iterations)
                else:
                    t_opt, R_opt = self.icp.full_pt2pt(self.reference_points, aligned_points, self.prev_position,
                                                       iterations=iterations)

            self.curr_r_opt, self.curr_t_opt = R_opt, t_opt

            refined_points = self.applyOptimalTranslation(r_opt=R_opt, t_opt=t_opt, points=aligned_points)
            translation_delta = self.getTranslationDelta(r_opt=R_opt, t_opt=t_opt)

            self.renderer.addPoints([estimated_position], self.red)
            self.renderer.addPoints([estimated_position + translation_delta], self.green)


            if self.count > self.reset_origin_treshold:
                print("reference update")
                self.reference_points = refined_points
                self.count = 0

            self.prev_position = estimated_position + translation_delta
            self.prev_time = current_time

            if cullColors:
                refined_points, colors = self.cullColorPoints(refined_points, colors)


            self.renderer.addPoints(refined_points, colors)