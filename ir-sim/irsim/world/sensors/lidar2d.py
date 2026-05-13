from math import pi, cos, sin
import numpy as np
from shapely import MultiLineString, Point, is_valid
from irsim.util.util import (
    geometry_transform,
    transform_point_with_state,
    get_transform,
    xy_to_coord,
    coord_to_xy
)
from irsim.global_param import env_param
from shapely import get_coordinates
from matplotlib.collections import LineCollection
from shapely.strtree import STRtree
from shapely.ops import unary_union
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from irsim.world.sensors.grid_map import GridMap
import matplotlib.pyplot as plt

class Lidar2D:
    """
    Simulates a 2D Lidar sensor for detecting obstacles in the environment.

    Args:
        state (np.ndarray): Initial state of the sensor.
        obj_id (int): ID of the associated object.
        range_min (float): Minimum detection range.
        range_max (float): Maximum detection range.
        angle_range (float): Total angle range of the sensor.
        number (int): Number of laser beams.
        scan_time (float): Time taken for one complete scan.
        noise (bool): Whether noise is added to measurements.
        std (float): Standard deviation for range noise.
        angle_std (float): Standard deviation for angle noise.
        offset (list): Offset of the sensor from the object's position.
        alpha (float): Transparency for plotting.
        has_velocity (bool): Whether the sensor measures velocity.
        **kwargs: Additional arguments.
            color (str): Color of the sensor.

    Attr:
        - sensor_type (str): Type of sensor ("lidar2d"). Default is "lidar2d". 
        - range_min (float): Minimum detection range in meters. Default is 0.
        - range_max (float): Maximum detection range in meters. Default is 10.
        - angle_range (float): Total angle range of the sensor in radians. Default is pi.
        - angle_min (float): Starting angle of the sensor's scan relative to the forward direction in radians. Calculated as -angle_range / 2.
        - angle_max (float): Ending angle of the sensor's scan relative to the forward direction in radians. Calculated as angle_range / 2.
        - angle_inc (float): Angular increment between each laser beam in radians. Calculated as angle_range / number.
        - number (int): Number of laser beams. Default is 100.
        - scan_time (float): Time taken to complete one full scan in seconds. Default is 0.1.
        - noise (bool): Whether to add noise to the measurements. Default is False.
        - std (float): Standard deviation for range noise in meters. Effective only if `noise` is True. Default is 0.2.
        - angle_std (float): Standard deviation for angle noise in radians. Effective only if `noise` is True. Default is 0.02.
        - offset (np.ndarray): Offset of the sensor relative to the object's position, formatted as [x, y, theta]. Default is [0, 0, 0].
        - lidar_origin (np.ndarray): Origin position of the Lidar sensor, considering offset and the object's state.
        - alpha (float): Transparency level for plotting the laser beams. Default is 0.3.
        - has_velocity (bool): Whether the sensor measures the velocity of detected points. Default is False.
        - velocity (np.ndarray): Velocity data for each laser beam, formatted as (2, number) array. Effective only if `has_velocity` is True. Initialized to zeros.
        - time_inc (float): Time increment for each scan, simulating the sensor's time resolution. Default is 5e-4.
        - range_data (np.ndarray): Array storing range data for each laser beam. Initialized to `range_max` for all beams.
        - angle_list (np.ndarray): Array of angles corresponding to each laser beam, distributed linearly from `angle_min` to `angle_max`.
        - color (str): Color of the sensor's representation in visualizations. Default is "r" (red).
        - obj_id (int): ID of the associated object, used to differentiate between multiple sensors or objects in the environment. Default is 0.
        - plot_patch_list (list): List storing plot patches (e.g., line collections) for visualization purposes.
        - plot_line_list (list): List storing plot lines for visualization purposes.
        - plot_text_list (list): List storing plot text elements for visualization purposes.
    """

    def __init__(
        self,
        state=None,
        obj_id=0,
        range_min=0,
        range_max=10,
        angle_range=pi,
        number=100,
        scan_time=0.1,
        noise=False,
        std=0.2,
        angle_std=0.02,
        offset=[0, 0, 0],
        alpha=0.3,
        has_velocity=False,
        **kwargs,
    ) -> None:
        """
        Initialize the Lidar2D sensor.

        
        """
        self.sensor_type = "lidar2d"

        self.range_min = range_min
        self.range_max = range_max

        self.angle_range = angle_range
        self.angle_min = -angle_range / 2
        self.angle_max = angle_range / 2
        self.angle_inc = angle_range / number

        self.number = number
        self.scan_time = scan_time
        self.noise = noise
        self.std = std
        self.angle_std = angle_std
        self.offset = np.c_[offset]
        self.lidar_origin = self.offset

        self.alpha = alpha
        self.has_velocity = has_velocity
        self.velocity = np.zeros((2, number))

        self.time_inc = (angle_range / (2 * pi)) * scan_time / number
        self.range_data = range_max * np.ones(number)
        self.angle_list = np.linspace(self.angle_min, self.angle_max, num=number)

        self._state = state
        self.init_geometry(self._state)

        self.color = kwargs.get("color", "r")

        self.obj_id = obj_id

        self.plot_patch_list = []
        self.plot_line_list = []
        self.plot_text_list = []

        self.build_map = kwargs.get("build_map", False)
        self.map_resolution = kwargs.get("map_resolution", 0.04)
        if self.build_map:
            # create a local grid map based on laser scan
            self.local_map = GridMap(
                height=2 * self.range_max+2,
                width=2 * self.range_max+2,
                resolution=self.map_resolution,
                offset=self.offset,
                color="k",
                static=False,
                **kwargs
            )
            self.update_local_map()

        # for debug
        self.local_map_update_index = 0

    def init_geometry(self, state):
        """
        Initialize the Lidar's scanning geometry.

        Args:
            state (np.ndarray): Current state of the sensor.
        """
        segment_point_list = []

        for i in range(self.number):
            x = self.range_data[i] * cos(self.angle_list[i])
            y = self.range_data[i] * sin(self.angle_list[i])

            point0 = np.zeros((1, 2))
            point = np.array([[x], [y]]).T

            segment = np.concatenate((point0, point), axis=0)

            segment_point_list.append(segment)

        self._init_geometry = MultiLineString(segment_point_list)
        self._init_geometry = geometry_transform(self._init_geometry, self.offset)
        self.lidar_origin = transform_point_with_state(self.offset, state)
        self._geometry = geometry_transform(self._init_geometry, state)


    def step(self, state):
        """
        Update the Lidar's state and process intersections with environment objects.

        Args:
            state (np.ndarray): New state of the sensor.
        """
        self._state = state

        self.lidar_origin = transform_point_with_state(self.offset, self._state)
        new_geometry = geometry_transform(self._init_geometry, self._state)

        new_geometry, intersect_indices = self.laser_geometry_process(new_geometry)

        if len(intersect_indices) == 0:
            self._geometry = new_geometry
            self.calculate_range()
        else:
            origin_point = Point(self.lidar_origin[0, 0], self.lidar_origin[1, 0])
            filtered_geoms = [
                g for g in new_geometry.geoms if g.intersects(origin_point)
            ]
            self._geometry = MultiLineString(filtered_geoms)
            self.calculate_range_vel(intersect_indices)
        
        # update local grid map
        if self.build_map:
            self.update_local_map()
            # self.plot_local_map()

    def laser_geometry_process(self, lidar_geometry):

        '''
        Find the intersected objects and return the intersected indices with the lidar geometry
        
        Args:
            lidar_geometry (shapely.geometry.MultiLineString): The geometry of the lidar.

        Returns:
            list: The indices of the intersected objects.
        '''

        filtered_objects = [
            obj
            for obj in env_param.objects
            if obj._id != self.obj_id and is_valid(obj._geometry) and not obj.unobstructed
        ]

        geometries = [obj._geometry for obj in filtered_objects]
        spatial_index = STRtree(geometries)
        potential_geometries_index = spatial_index.query(lidar_geometry)

        geometries_to_subtract = []
        intersect_indices = []

        for geom_index in potential_geometries_index:
            geo = geometries[geom_index]
            obj = filtered_objects[geom_index]

            if obj.shape == 'map':
                linestrings = [line for line in geo.geoms]
                tree = STRtree(linestrings)
                potential_intersections = tree.query(lidar_geometry)
                filtered_lines = [linestrings[i] for i in potential_intersections]
                filtered_multi_lines = MultiLineString(filtered_lines)

                if lidar_geometry.intersects(filtered_multi_lines):
                    geometries_to_subtract.append(filtered_multi_lines)
                    intersect_indices.append(geom_index)

            else:
                if lidar_geometry.intersects(geo):
                    geometries_to_subtract.append(geo)
                    intersect_indices.append(geom_index)

        if geometries_to_subtract:
            merged_geometry = unary_union(geometries_to_subtract)
            lidar_geometry = lidar_geometry.difference(merged_geometry)

        return lidar_geometry, intersect_indices

    def calculate_range(self):
        """
        Calculate the range data from the current geometry.
        """
        for index, l in enumerate(self._geometry.geoms):
            # self.range_data[index] = l.length
            if self.noise:
                self.range_data[index] = l.length + np.random.normal(0, self.std)
            else:
                self.range_data[index] = l.length

    def calculate_range_vel(self, intersect_index):
        """
        Calculate the range data and velocities from intersected geometries.

        Args:
            intersect_index (list): List of intersected object indices.
        """
        for index, l in enumerate(self._geometry.geoms):
            # self.range_data[index] = l.length
            self.range_data[index] = (
                l.length + np.random.normal(0, self.std) if self.noise else l.length
            )

            if self.has_velocity:
                if l.length < self.range_max - 0.02:
                    for index_obj in intersect_index:
                        obj = env_param.objects[index_obj]
                        if obj.geometry.distance(l) < 0.1:
                            self.velocity[:, index : index + 1] = obj.velocity_xy
                            break

    def get_scan(self):
        """
        Get the 2D lidar scan data. refer to the ros topic scan: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

        Returns:
            dict: Scan data including angles, ranges, and velocities.
        """
        scan_data = {}
        scan_data["angle_min"] = self.angle_min
        scan_data["angle_max"] = self.angle_max
        scan_data["angle_increment"] = self.angle_inc
        scan_data["time_increment"] = self.time_inc
        scan_data["scan_time"] = self.scan_time
        scan_data["range_min"] = self.range_min
        scan_data["range_max"] = self.range_max
        scan_data["ranges"] = self.range_data
        scan_data["intensities"] = None
        scan_data["velocity"] = self.velocity

        return scan_data

    def get_points(self):
        """
        Convert scan data to a point cloud.

        Returns:
            np.ndarray: Point cloud (2xN).
        """
        return self.scan_to_pointcloud()

    def get_offset(self):
        """
        Get the sensor's offset.

        Returns:
            list: Offset as a list.
        """
        return np.squeeze(self.offset).tolist()

    def plot(self, ax, **kwargs):
        """
        Plot the Lidar's detected lines on a given axis.

        Args:
            ax: Matplotlib axis.
            **kwargs: Plotting options.
        """
        lines = []

        for i in range(self.number):
            x = self.range_data[i] * cos(self.angle_list[i])
            y = self.range_data[i] * sin(self.angle_list[i])

            position = self._state[0:2, 0]
            trans, rot = get_transform(self._state)
            range_end_position = rot @ np.array([[x], [y]]) + trans

            if isinstance(ax, Axes3D):
                position = np.array([position[0], position[1], 0])
                end_position = np.array(
                    [range_end_position[0, 0], range_end_position[1, 0], 0]
                )
                segment = [position, end_position]
            else:
                segment = [position, range_end_position[0:2, 0]]

            lines.append(segment)

        if isinstance(ax, Axes3D):
            self.line_segments = Line3DCollection(
                lines, linewidths=1, colors=self.color, alpha=self.alpha, zorder=0
            )
            ax.add_collection3d(self.line_segments)
        else:
            self.line_segments = LineCollection(
                lines, linewidths=1, colors=self.color, alpha=self.alpha, zorder=0
            )
            ax.add_collection(self.line_segments)

        self.plot_patch_list.append(self.line_segments)

        # plot local grid map
        # map = self.local_map.map
    
    def set_laser_color(self, laser_indices, laser_color: str = 'blue'):

        """
        Set a specific color of the selected lasers.

        Args:
            laser_indices (list): The indices of the lasers to set the color.
            laser_color (str): The color to set the selected lasers. Default is 'blue'.
        """

        current_color = [self.color] * self.number

        for index in laser_indices:
            if index < self.number:
                current_color[index] = laser_color

        self.line_segments.set_color(current_color)

    def plot_clear(self):
        """
        Clear the plot elements from the axis.
        """
        [patch.remove() for patch in self.plot_patch_list]
        [line.pop(0).remove() for line in self.plot_line_list]
        [text.remove() for text in self.plot_text_list]

        self.plot_patch_list = []
        self.plot_line_list = []
        self.plot_text_list = []

    def scan_to_pointcloud(self):
        """
        Convert the Lidar scan data to a point cloud.

        Returns:
            np.ndarray: Point cloud (2xN).
        """
        point_cloud = []

        ranges = self.range_data
        angles = np.linspace(self.angle_min, self.angle_max, len(ranges))

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (self.range_max - 0.02):
                point = np.array([[scan_range * cos(angle)], [scan_range * sin(angle)]])
                point_cloud.append(point)

        if len(point_cloud) == 0:
            return None

        point_array = np.hstack(point_cloud)

        return point_array

    def update_local_map(self):
        """
        Get the local grid map based on the Lidar scan data.

        Returns:
            GridMap: The local grid map.
        """
        occ = self.local_map.update_map_from_lidar(self.get_scan())
        # print("shape:", self.local_map.map.shape)
        # print("occupied: {}".format(np.count_nonzero(self.local_map.map == 1.0)))
        # print("unknown: {}".format(np.count_nonzero(self.local_map.map == 0.5)))    
        # print("free: {}".format(np.count_nonzero(self.local_map.map == 0)))

        # test
        # xy_res = np.array(occ).shape
        # plt.figure(1, figsize=(10, 4))
        # plt.subplot(122)
        # plt.imshow(occ, cmap="PiYG_r")
        # # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
        # plt.clim(-0.4, 1.4)
        # plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
        # plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
        # plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
        # plt.colorbar()
        # plt.subplot(121)
        # # plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(np.size(oy))], "ro-")
        # plt.axis("equal")
        # plt.plot(0.0, 0.0, "ob")
        # plt.gca().set_aspect("equal", "box")
        # bottom, top = plt.ylim()  # return the current y-lim
        # plt.ylim((top, bottom))  # rescale y axis, to match the grid orientation
        # plt.grid(True)
        # plt.show()

    def check_visible(self, object):
        # check if the object is visible to the robot using local grid map
        ego_pose = self._state[0:3, 0]  # (x,y,yaw)
        obj_pose = object.state[0:3, 0]  # (x,y,yaw)

        cos_theta = np.cos(ego_pose[2])
        sin_theta = np.sin(ego_pose[2])

        dx = obj_pose[0] - ego_pose[0] 
        dy = obj_pose[1] - ego_pose[1] 

        x_ego = cos_theta * dx + sin_theta * dy
        y_ego = -sin_theta * dx + cos_theta * dy

        rel_position = np.array([x_ego, y_ego])
        obj_radius = object.radius

        x0 = rel_position[0] - obj_radius
        x1 = rel_position[0] + obj_radius
        y0 = rel_position[1] - obj_radius
        y1 = rel_position[1] + obj_radius

        # cxcy_0 = xy_to_coord(np.array([x0, y0]), self.local_map.resolution, [self.local_map.map_width, self.local_map.map_height])

        # cxcy_1 = xy_to_coord(np.array([x1, y1]), self.local_map.resolution, [self.local_map.map_width, self.local_map.map_height])


        x0 = int(round(x0/self.local_map.resolution+self.local_map.x_center))
        x1 = int(round(x1/self.local_map.resolution+self.local_map.x_center))
        y0 = int(round(y0/self.local_map.resolution+self.local_map.y_center))
        y1 = int(round(y1/self.local_map.resolution+self.local_map.y_center))

        if x0 < 0 or x1 >= self.local_map.map.shape[1] or y0 < 0 or y1 >= self.local_map.map.shape[0]:
            return False
        
        # all in unknown space (TODO: DEBUG)
        # obj_map = np.array(self.local_map.map[x0:x1, y0:y1])
        obj_map = np.array(self.local_map.map[y0:y1, x0:x1])

        if np.sum(obj_map == 0.5) == (x1-x0)*(y1-y0):
            return False
        
        return True

    def plot_local_map(self):
        """
        Plot the local grid map.
        """
        grid_map = self.local_map.map
        ax = plt.gca()
        ax.imshow(grid_map, cmap='gray', origin='lower')
        ax.set_title("Local Grid Map")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        plt.savefig("/home/hjyeee/local_map_{:03d}.png".format(self.local_map_update_index))
        plt.close()
        self.local_map_update_index += 1