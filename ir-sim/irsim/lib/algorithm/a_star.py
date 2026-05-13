import heapq
import numpy as np
import cv2
from scipy.interpolate import splprep, splev

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.ndimage import distance_transform_edt

# 定义八个方向（包括斜向）
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] # 斜向


class AStar():
    def __init__(self, map_free, dialate=10, skip_num=10, smooth=False, viz=False):
        self.dialate = dialate
        self.skip_num = skip_num
        # self.scale = 4/12
        self.smooth = smooth
        self.map_dialate = self.post_proc_map(map_free, dialate)
        self.distance_field = distance_transform_edt(1-map_free)
        self.gradient_field = self.compute_gradient(self.distance_field)



        # self.map_dialate_sfm = self.post_proc_map(map_free, dialate//2)
        # self.distance_field_sfm = distance_transform_edt(1-map_free)
        # self.gradient_field_sfm = self.compute_gradient(self.distance_field)

        # if dialate == 8:
            # save_distance_and_gradient_cv(map_free, self.distance_field, self.gradient_field, out_prefix='/home/hjyeee/df_astar')

        self.map_size = self.map_dialate.shape
        
        self.viz = viz
        # if dialate == 2:
        #     cv2.imwrite('/home/hjyeee/map_dialate.png', np.transpose(self.map_dialate)*255)
        # else:
        #     cv2.imwrite('/home/hjyeee/map_dialate2.png', np.transpose(self.map_dialate)*255)

    def post_proc_map(self, map, dialate):
        kernel = np.ones((dialate, dialate), np.uint8)
        dilated_img = cv2.dilate(map, kernel, iterations=1)

        return dilated_img

    @staticmethod
    def compute_gradient(distance_field):
        """计算EDT的梯度场"""
        
        grad = np.gradient(distance_field)
        
        # 计算梯度模长
        magnitude = np.sqrt(grad[0]**2 + grad[1]**2)
        
        # 处理零梯度区域
        magnitude[magnitude == 0] = 1e-6  # 避免除以零
        
        # 归一化梯度
        grad_norm = grad / magnitude
        
        return grad_norm

    def get_path(self, start, goal):

        start = tuple(start)
        goal = tuple(goal)
        

        distance, parents= self.get_cost(start, goal)

        # 重建路径并可视化
        if distance[goal] != float('inf'):
            path = self.reconstruct_path(parents, start, goal)
            # control_points = self.select_control_points(path)
            control_points = path

            if self.smooth:
                
                # 使用B样条平滑控制点路径
                smoothed_path = self.smooth_with_b_spline(control_points, path.shape[0])
                print('*** Found b-spline path with: ', smoothed_path.shape)

                # 可视化结果
                if self.viz:

                    self.viz_cost_and_path(distance, start, goal, smoothed_path)                
                    # plt.scatter(control_points[:,1], control_points[:,0], color='g')
                    
                return smoothed_path
            else:
                return control_points
        else:
            print("[Astar] !!! No path found from start to goal.")
            if self.viz:
                self.viz_cost_and_path(distance, start, goal, path=None)
            return None

    def sample_path(self, path_coords):
        if path_coords.shape[0] > self.skip_num:
            way_cords = np.vstack([path_coords[::self.skip_num], path_coords[-1]])
        else:
            way_cords = np.expand_dims(path_coords[-1],axis=0)
        return way_cords


    def get_cost(self, start, goal):
        
        u_lim, v_lim = self.map_size
        # print("[Astar] map size:", u_lim, v_lim)
        distance = np.full((u_lim, v_lim), np.inf)  # 用于存储从起点到每个节点的最短距离
        distance[start] = 0
        pq = [(0, start)]  # 优先队列（最小堆），存储 (距离, 坐标)
        parent = np.full((u_lim, v_lim, 2), -1)  # 记录父节点坐标

        while pq:
            current_dist, current_node = heapq.heappop(pq)
            current_dist = current_dist - np.sqrt((current_node[0]-goal[0])**2+(current_node[1]-goal[1])**2)

            if current_node == goal:
                break  # 到达终点
            
            for direction in DIRECTIONS:
                neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
                
                if 0 <= neighbor[0] < u_lim and 0 <= neighbor[1] < v_lim and self.map_dialate[neighbor] == 0:
                    # 计算从当前节点到邻居节点的距离
                    weight = 1 if 0 in direction else np.sqrt(2)
                    heuristic = np.sqrt((neighbor[0]-goal[0])**2+(neighbor[1]-goal[1])**2)
                    distance_through_current = current_dist + weight + heuristic
                    
                    if distance_through_current < distance[neighbor]:
                        distance[neighbor] = distance_through_current
                        parent[neighbor] = current_node
                        heapq.heappush(pq, (distance_through_current, neighbor))

        return distance, parent

    def reconstruct_path(self, parent, start, goal):
    
        # 从终点回溯到起点，重建最短路径
        x, y = goal
        path = []
        while parent[x, y][0] != -1:
            path.append((x, y))
            x, y = parent[x, y]
        path.append(start)
        path.reverse()
        return np.array(path)
    
    def calculate_curvature(self, path):
        # 获取x和y坐标
        x = path[:, 0]
        y = path[:, 1]

        # 计算三角形的面积部分 (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)
        dx1 = x[2:] - x[:-2]  # x3 - x1
        dy1 = y[2:] - y[:-2]  # y3 - y1
        dx2 = x[1:-1] - x[:-2]  # x2 - x1
        dy2 = y[1:-1] - y[:-2]  # y2 - y1

        # 计算曲率的分子部分
        area = np.abs(dx1 * dy2 - dy1 * dx2)

        # 计算路径长度的3/2次方部分 (x2 - x1)^2 + (y2 - y1)^2
        length_sq = (dx2 ** 2 + dy2 ** 2) ** 1.5

        # 计算曲率，避免除零错误
        curvatures = np.divide(2 * area, length_sq, where=length_sq != 0)

        return curvatures

    def select_control_points(self, path, curvature_threshold=0.2):
        curvatures = self.calculate_curvature(path)
        control_points = []  # 始终包括起点 path[0]
        j = 0 
        for i, curv in enumerate(curvatures):
            if curv > curvature_threshold:
                control_points.append(path[i + 1])  # 曲率变化较大的点作为控制点
                j = 0
            else:
                j += 1
                if j > 4:
                    control_points.append(path[i])
                    j = 0
        control_points.append(path[-1])
        return np.array(control_points)


    def smooth_with_b_spline(self, control_points, num):

        tck, u = splprep(control_points.T, s=0)  # 使用B样条拟合
        x = np.linspace(0, 1, num)
        new_points = splev(x, tck)  # 插值100个点
        return np.array(new_points).T


    def viz_cost_and_path(self, distance, start, goal, path):
        plt.figure()
        plt.imshow(self.map_dialate)
        plt.imshow(distance)
        plt.plot(start[1], start[0],'gx')
        plt.plot(goal[1], goal[0], 'rx')
        if path is not None:
            plt.plot( path[:, 1], path[:, 0], color='b')
        # plt.show()



def compute_gradient(distance_field: np.ndarray):
    # np.gradient 返回 (gy, gx)，这里转成 (gx, gy)
    gy, gx = np.gradient(distance_field)
    return gx, gy

def to_colormap(img_f: np.ndarray, cmap=cv2.COLORMAP_VIRIDIS):
    """将float数组归一化到0-255并着色为BGR图像"""
    if img_f.size == 0:
        raise ValueError("empty image")
    mn, mx = float(np.nanmin(img_f)), float(np.nanmax(img_f))
    if mx == mn:
        norm = np.zeros_like(img_f, dtype=np.uint8)
    else:
        norm = ((img_f - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cmap)

def draw_quiver_on(dist_bgr: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                   map_free: np.ndarray = None,
                   step: int = 6,   # 采样步长，越大箭头越稀疏
                   length: float = 10.0,  # 箭头长度（像素）
                   thickness: int = 1):
    """在BGR图上画单位梯度方向箭头（基于OpenCV坐标：原点左上、y向下）"""
    h, w = dist_bgr.shape[:2]
    out = dist_bgr.copy()

    # 单位向量（只显示方向）
    mag = np.hypot(gx, gy) + 1e-9
    ux = gx / mag
    uy = gy / mag

    # 采样网格
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)

    for y in ys:
        for x in xs:
            if map_free is not None and map_free[y, x] == 0:  # 障碍上不画
                continue
            # OpenCV坐标：右为 +x，下为 +y；梯度 ∇d 指向“远离障碍”
            x2 = int(round(x + ux[y, x] * length))
            y2 = int(round(y + uy[y, x] * length))
            cv2.arrowedLine(out, (x, y), (x2, y2), color=(0, 0, 0), thickness=thickness, tipLength=0.3)
    return out

def save_distance_and_gradient_cv(map_free: np.ndarray,
                                  distance_field: np.ndarray,
                                  gradient_field=None,
                                  out_prefix: str = "df"):
    """
    用OpenCV将距离场/梯度场进行着色可视化并存盘：
      out_prefix_distance.png
      out_prefix_gradmag.png
      out_prefix_quiver.png
      out_prefix_data.npz (存原始数组)
    gradient_field: (2,H,W) 或 (H,W,2)，否则自动用 np.gradient 计算
    """
    h, w = distance_field.shape

    # 解析/计算梯度
    if gradient_field is None:
        gx, gy = compute_gradient(distance_field)
    else:
        gf = np.asarray(gradient_field)
        if gf.ndim != 3:
            raise ValueError("gradient_field 需为3维 (2,H,W) 或 (H,W,2)")
        if gf.shape[0] == 2:
            gx, gy = gf[0], gf[1]
        elif gf.shape[2] == 2:
            gx, gy = gf[..., 0], gf[..., 1]
        else:
            raise ValueError(f"无法解析 gradient_field 形状: {gf.shape}")

    # 距离场热力图
    dist_bgr = to_colormap(distance_field, cmap=cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(f"{out_prefix}_distance.png", dist_bgr)

    # 梯度模长热力图
    grad_mag = np.hypot(gx, gy)
    gradmag_bgr = to_colormap(grad_mag, cmap=cv2.COLORMAP_MAGMA)
    cv2.imwrite(f"{out_prefix}_gradmag.png", gradmag_bgr)

    # 在距离场上画箭头（梯度方向）
    quiver_bgr = draw_quiver_on(dist_bgr, gx, gy, map_free=map_free, step=6, length=10, thickness=1)
    # 可选：障碍区域叠加成深色（让可行域更清晰）
    if map_free is not None:
        mask = (map_free == 0)
        quiver_bgr[mask] = (30, 30, 30)
    cv2.imwrite(f"{out_prefix}_quiver.png", quiver_bgr)

    # 保存数值数据（便于后续复现/调参）
    np.savez_compressed(f"{out_prefix}_data.npz",
                        map_free=map_free.astype(np.uint8),
                        distance_field=distance_field.astype(np.float32),
                        gx=gx.astype(np.float32),
                        gy=gy.astype(np.float32))

    print(f"[OK] Saved: {out_prefix}_distance.png, {out_prefix}_gradmag.png, {out_prefix}_quiver.png, {out_prefix}_data.npz")