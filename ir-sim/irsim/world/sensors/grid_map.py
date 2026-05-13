"""

LIDAR to 2D grid map example

author: Erno Horvath, Csaba Hajdu based on Atsushi Sakai's scripts

"""

import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

EXTEND_AREA = 2.0
# EXTEND_AREA = 0.0

class GridMap:
    """
    unknown/unoccupied: 0.5
    free: 0
    occupied: 1
    """
    def __init__(
        self,
        height=10,
        width=10,
        resolution=0.1,
        offset=[0, 0, 0],
        color="k",
        static=False,
        **kwargs,
    ):
        self.map_height = height
        self.map_width = width
        self.resolution = resolution
        self.offset = offset
        self.color = color
        self.static = static

        # Initialize map with all unknown values
        self.map = np.ones(
            (int(height / resolution), int(width / resolution)), dtype=np.int8
        ) * 0.5

        self.x_center = int(self.map.shape[0] / 2)
        self.y_center = int(self.map.shape[1] / 2)
    
    def update_map_from_lidar(self, scan_data):
        """
        TODO: 还没有考虑传感器坐标系和机器人坐标系之间的变换关系
        """
        ranges = np.array(scan_data['ranges'])
        angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))
        ox = []
        oy = []
        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]
            # if scan_range < ( scan_data['range_max'] - 0.02):
            #     ox.append(scan_range * np.sin(angle))
            #     oy.append(scan_range * np.cos(angle))
            ox.append(scan_range * np.sin(angle))
            oy.append(scan_range * np.cos(angle))
            # print(scan_range * np.sin(angle), scan_range * np.cos(angle))

        occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = \
        self.generate_ray_casting_grid_map(ox, oy, self.resolution, False)


        # print(occupancy_map.shape)
        # print(min_x, max_x, min_y, max_y)
        x0 = int(min_x/xy_resolution+self.x_center)
        x1 = int(max_x/xy_resolution+self.x_center)
        y0 = int(min_y/xy_resolution+self.y_center)
        y1 = int(max_y/xy_resolution+self.y_center)
        # print(x0, x1, y0, y1)
        self.map[x0:x1, y0:y1] = occupancy_map

        return occupancy_map


    def calc_grid_map_config(self, ox, oy, xy_resolution):
        """
        Calculates the size, and the maximum distances according to the the
        measurement center
        """
        min_x = round(min(ox) - EXTEND_AREA / 2.0)
        min_y = round(min(oy) - EXTEND_AREA / 2.0)
        max_x = round(max(ox) + EXTEND_AREA / 2.0)
        max_y = round(max(oy) + EXTEND_AREA / 2.0)
        xw = int(round((max_x - min_x) / xy_resolution))
        yw = int(round((max_y - min_y) / xy_resolution))
        # print("The grid map is ", xw, "x", yw, ".")
        # print("min_x, min_y, max_x, max_y", min_x, min_y, max_x, max_y)
        return min_x, min_y, max_x, max_y, xw, yw

    
    def init_flood_fill(self, center_point, obstacle_points, xy_points, min_coord, xy_resolution):
        """
        center_point: center point
        obstacle_points: detected obstacles points (x,y)
        xy_points: (x,y) point pairs
        """
        center_x, center_y = center_point
        prev_ix, prev_iy = center_x - 1, center_y
        ox, oy = obstacle_points
        xw, yw = xy_points
        min_x, min_y = min_coord
        occupancy_map = (np.ones((xw, yw))) * 0.5
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(math.floor((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(math.floor((y - min_y) / xy_resolution))
            # print("y, min_y", y/xy_resolution, min_y/xy_resolution)
            # print(occupancy_map.shape)
            # print("ix, iy", ix, iy)
            free_area = bresenham((prev_ix, prev_iy), (ix, iy))
            for fa in free_area:
                occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
            prev_ix = ix
            prev_iy = iy
        return occupancy_map

    def flood_fill(self, center_point, occupancy_map):
        """
        center_point: starting point (x,y) of fill
        occupancy_map: occupancy map generated from Bresenham ray-tracing
        """
        # Fill empty areas with queue method
        sx, sy = occupancy_map.shape
        fringe = deque()
        fringe.appendleft(center_point)
        while fringe:
            n = fringe.pop()
            nx, ny = n
            # West
            if nx > 0:
                if occupancy_map[nx - 1, ny] == 0.5:
                    occupancy_map[nx - 1, ny] = 0.0
                    fringe.appendleft((nx - 1, ny))
            # East
            if nx < sx - 1:
                if occupancy_map[nx + 1, ny] == 0.5:
                    occupancy_map[nx + 1, ny] = 0.0
                    fringe.appendleft((nx + 1, ny))
            # North
            if ny > 0:
                if occupancy_map[nx, ny - 1] == 0.5:
                    occupancy_map[nx, ny - 1] = 0.0
                    fringe.appendleft((nx, ny - 1))
            # South
            if ny < sy - 1:
                if occupancy_map[nx, ny + 1] == 0.5:
                    occupancy_map[nx, ny + 1] = 0.0
                    fringe.appendleft((nx, ny + 1))

    def generate_ray_casting_grid_map(self, ox, oy, xy_resolution, breshen=True):
        """
        The breshen boolean tells if it's computed with bresenham ray casting
        (True) or with flood fill (False)
        """
        min_x, min_y, max_x, max_y, x_w, y_w = self.calc_grid_map_config(
            ox, oy, xy_resolution)
        # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
        occupancy_map = np.ones((x_w, y_w)) / 2
        center_x = int(
            round(-min_x / xy_resolution))  # center x coordinate of the grid map
        center_y = int(
            round(-min_y / xy_resolution))  # center y coordinate of the grid map
        # occupancy grid computed with bresenham ray casting
        if breshen:
            for (x, y) in zip(ox, oy):
                # x coordinate of the the occupied area
                ix = int(math.floor((x - min_x) / xy_resolution))
                # y coordinate of the the occupied area
                iy = int(math.floor((y - min_y) / xy_resolution))
                laser_beams = bresenham((center_x, center_y), (
                    ix, iy))  # line form the lidar to the occupied point
                for laser_beam in laser_beams:
                    occupancy_map[laser_beam[0]][
                        laser_beam[1]] = 0.0  # free area 0.0
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        # occupancy grid computed with with flood fill
        else:
            occupancy_map = self.init_flood_fill((center_x, center_y), (ox, oy),
                                            (x_w, y_w),
                                            (min_x, min_y), xy_resolution)
            self.flood_fill((center_x, center_y), occupancy_map)
            occupancy_map = np.array(occupancy_map, dtype=float)
            for (x, y) in zip(ox, oy):
                ix = int(math.floor((x - min_x) / xy_resolution))
                iy = int(math.floor((y - min_y) / xy_resolution))
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution

def atan_zero_to_twopi(y, x):
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return angle

def bresenham(start, end):
    """
    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points
    