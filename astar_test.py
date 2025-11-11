import math
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
import triangle as tr
from typing import List, Tuple, Callable, Dict, Optional
import matplotlib.patches as patches 
from scipy.optimize import linear_sum_assignment
import heapq # <-- NEW: For A* Priority Queue

# ==============================================================================
# CONSTANTS
# ==============================================================================
TOLERANCE = 0.001
N_FRONTIER_SAMPLES = 0 # Set to 3 or 5 for good greedy selection
MIN_FRONTIER_LENGTH = 10
TOTAL_ANIMATION_STEPS = 10 # Total frames for the *longest* path
# ==============================================================================
# (All helper functions: two_sum, split, orient2d, etc. are identical)
# ==============================================================================
epsilon = 1.1102230246251565e-16
splitter = 134217729.0

def two_sum(a, b):
    x = a + b
    b_virtual = x - a
    a_virtual = x - b_virtual
    b_roundoff = b - b_virtual
    a_roundoff = a - a_virtual
    return x, a_roundoff + b_roundoff

def split(a):
    c = splitter * a
    a_big = c - a
    a_hi = c - a_big
    a_lo = a - a_hi
    return a_hi, a_lo

def two_product(a, b):
    x = a * b
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    err1 = x - (a_hi * b_hi)
    err2 = err1 - (a_lo * b_hi)
    err3 = err2 - (a_hi * b_lo)
    y = a_lo * b_lo - err3
    return x, y

def orient2d_adapt(pa, pb, pc):
    acx = pa[0] - pc[0]
    bcx = pb[0] - pc[0]
    acy = pa[1] - pc[1]
    bcy = pb[1] - pc[1]
    detleft, detleft_err = two_product(acx, bcy)
    detright, detright_err = two_product(acy, bcx)
    det, det_err = two_sum(detleft, -detright)
    b_virtual = det - detleft
    a_virtual = det - b_virtual
    b_roundoff = -detright - b_virtual
    a_roundoff = detleft - a_virtual
    det_err += a_roundoff + b_roundoff
    return det_err

def orient2d(pa, pb, pc):
    det = (pa[0] - pc[0]) * (pb[1] - pc[1]) - (pa[1] - pc[1]) * (pb[0] - pc[0])
    if det != 0.0:
        det_bound = (abs(pa[0] - pc[0]) + abs(pb[0] - pc[0])) * (abs(pa[1] - pc[1]) + abs(pb[1] - pc[1]))
        if abs(det) >= epsilon * det_bound:
            return det
    return orient2d_adapt(pa, pb, pc)

# ==============================================================================
# Type Aliases (from TE code)
# ==============================================================================
PointT = Tuple[float, float]
Segment = Tuple[float, float, float, float]
DelaunatorLike = Dict[str, np.ndarray]
# --- NEW: Type for our A* graph ---
AStarGraph = Dict[int, List[Tuple[int, float]]] # {t_idx: [(neighbor_t_idx, cost), ...]}

# ==============================================================================
# HELPER TO REPLACE DELAUNATOR (from TE code)
# ==============================================================================
def build_halfedges(triangles: np.ndarray) -> np.ndarray:
    num_triangles = len(triangles)
    num_halfedges = num_triangles * 3
    halfedges = -np.ones(num_halfedges, dtype=int)
    edge_map = {}
    for t_idx, tri in enumerate(triangles):
        for i in range(3):
            p1, p2 = tri[i], tri[(i + 1) % 3]
            edge = tuple(sorted((p1, p2)))
            if edge not in edge_map: edge_map[edge] = []
            edge_map[edge].append(t_idx * 3 + i)
    for edge, hes in edge_map.items():
        if len(hes) == 2:
            he1, he2 = hes
            halfedges[he1], halfedges[he2] = he2, he1
    return halfedges


def containing_triangle(d: DelaunatorLike, qx: float, qy: float) -> int:
    """Finds the triangle containing a point by a brute-force check."""
    coords = d['coords'].reshape(-1, 2)
    q = (qx, qy)
    num_triangles = len(d['triangles']) // 3

    for t_idx in range(num_triangles):
        p_indices = points_of_tri(d, t_idx)
        p1, p2, p3 = coords[p_indices]
        if orient2d(p1, p2, q) >= 0 and orient2d(p2, p3, q) >= 0 and orient2d(p3, p1, q) >= 0:
            return t_idx
    return -1
# ==============================================================================
# CORE TRIANGULATION HELPER FUNCTIONS (from TE code)
# ==============================================================================
def next_edge(e: int) -> int: return e - 2 if e % 3 == 2 else e + 1
def prev_edge(e: int) -> int: return e + 2 if e % 3 == 0 else e - 1 
def points_of_tri(d: DelaunatorLike, t: int) -> np.ndarray: return d['triangles'][t*3 : t*3 + 3]

def is_left_of(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> bool:
    return orient2d((x1, y1), (x2, y2), (px, py)) > 0

def is_right_of(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> bool:
    return orient2d((x1, y1), (x2, y2), (px, py)) < 0

def order_angles(qx: float, qy: float, p1x: float, p1y: float, p2x: float, p2y: float) -> List[float]:
    seg_left = is_left_of(qx, qy, p2x, p2y, p1x, p1y)
    lx, ly = (p1x, p1y) if seg_left else (p2x, p2y)
    rx, ry = (p2x, p2y) if seg_left else (p1x, p1y)
    return [lx, ly, rx, ry]
    
def order_del_angles(d: DelaunatorLike, qx: float, qy: float, p1: int, p2: int) -> List[float]:
    coords = d['coords'].reshape(-1, 2)
    p1x, p1y = coords[p1]
    p2x, p2y = coords[p2]
    return order_angles(qx, qy, p1x, p1y, p2x, p2y)

def is_within_cone(px: float, py: float, slx: float, sly: float, srx: float, sry: float, rlx: float, rly: float, rrx: float, rry: float) -> bool:
    if is_left_of(px, py, slx, sly, rrx, rry): return False
    if is_left_of(px, py, rlx, rly, srx, sry): return False
    if rrx == slx and rry == sly: return False
    if srx == rlx and sry == rly: return False
    return True

def restrict_angles(px: float, py: float, slx: float, sly: float, srx: float, sry: float, rlx: float, rly: float, rrx: float, rry: float) -> Tuple[List[float], bool, bool]:
    nlx, nly, res_left = (slx, sly, True) if is_right_of(px, py, rlx, rly, slx, sly) else (rlx, rly, False)
    nrx, nry, res_right = (srx, sry, True) if is_left_of(px, py, rrx, rry, srx, sry) else (rrx, rry, False)
    return ([nlx, nly, nrx, nry], res_left, res_right)

def seg_intersect_ray(s1x: float, s1y: float, s2x: float, s2y: float, r1x: float, r1y: float, r2x: float, r2y: float) -> float:
    rdx, rdy = r2x - r1x, r2y - r1y
    sdx, sdy = s2x - s1x, s2y - s1y
    denominator = sdx * rdy - sdy * rdx
    if denominator == 0: return float('inf')
    t2 = (rdx * (s1y - r1y) + rdy * (r1x - s1x)) / denominator
    if rdx != 0: t1 = (s1x + sdx * t2 - r1x) / rdx
    else:
        if rdy != 0: t1 = (s1y + sdy * t2 - r1y) / rdy
        else: return float('inf')
    if t1 < -1e-9 or t2 < -1e-9 or t2 > 1.0 + 1e-9: return float('inf')
    return t1

# ==============================================================================
# CORE TE ALGORITHM (from TE code) - (No Changes)
# ... (triangular_expansion function is identical) ...
# ==============================================================================
def triangular_expansion(
    d: DelaunatorLike, qx: float, qy: float, obstructs: Callable[[int], bool]
) -> List[Segment]:
    memo = {}
    def expand(edg_in: int, rlx: float, rly: float, rrx: float, rry: float) -> List[Segment]:
        key = (edg_in, rlx, rly, rrx, rry)
        if key in memo: return memo[key]
        
        ret: List[Segment] = []
        edges = [next_edge(edg_in), prev_edge(edg_in)]

        for edg in edges:
            p1, p2, adj_out = d['triangles'][edg], d['triangles'][next_edge(edg)], d['halfedges'][edg]
            slx, sly, srx, sry = order_del_angles(d, qx, qy, p1, p2)

            if not is_within_cone(qx, qy, slx, sly, srx, sry, rlx, rly, rrx, rry): continue
            
            [nlx, nly, nrx, nry], res_l, res_r = restrict_angles(qx, qy, slx, sly, srx, sry, rlx, rly, rrx, rry)
            
            if orient2d((qx, qy), (nrx, nry), (nlx, nly)) <= 0.0: continue

            if adj_out != -1 and not obstructs(edg):
                ret.extend(expand(adj_out, nlx, nly, nrx, nry))
                continue

            if not res_l:
                inter = seg_intersect_ray(slx, sly, srx, sry, qx, qy, rlx, rly)
                if inter != float('inf'): slx, sly = qx + inter * (rlx-qx), qy + inter * (rly-qy)
            if not res_r:
                inter = seg_intersect_ray(slx, sly, srx, sry, qx, qy, rrx, rry)
                if inter != float('inf'): srx, sry = qx + inter * (rrx-qx), qy + inter * (rry-qy)
            ret.append((slx, sly, srx, sry))
        memo[key] = ret
        return ret

    tri_start = containing_triangle(d, qx, qy)
    if tri_start == -1:
        print(f"Warning: Query point {qx, qy} is outside the triangulation."); return []

    coords = d['coords'].reshape(-1, 2)
    ret: List[Segment] = []
    p_indices = points_of_tri(d, tri_start)
    points = coords[p_indices]
    points_sorted = sorted(points.tolist(), key=lambda p: np.arctan2(p[1] - qy, p[0] - qx))

    for i in range(3):
        p_start, p_end = points_sorted[i], points_sorted[(i + 1) % 3]
        rlx, rly, rrx, rry = order_angles(qx, qy, p_start[0], p_start[1], p_end[0], p_end[1])
        
        for edg in [tri_start * 3, tri_start * 3 + 1, tri_start * 3 + 2]:
            p1_idx, p2_idx = d['triangles'][edg], d['triangles'][next_edge(edg)]
            p1_coords, p2_coords = coords[p1_idx], coords[p2_idx]
            
            p_start_arr, p_end_arr = np.array(p_start), np.array(p_end)
            if (np.array_equal(p1_coords, p_start_arr) and np.array_equal(p2_coords, p_end_arr)) or \
               (np.array_equal(p1_coords, p_end_arr) and np.array_equal(p2_coords, p_start_arr)):
                adj = d['halfedges'][edg]
                if adj == -1 or obstructs(edg):
                    ret.append(order_angles(qx, qy, p1_coords[0], p1_coords[1], p2_coords[0], p2_coords[1]))
                else:
                    ret.extend(expand(adj, rlx, rly, rrx, rry))
                break
    return ret

# ==============================================================================
# WRAPPER FUNCTION (The "Adapter")
# ==============================================================================
def get_visibility_from_triangulation(
    viewpoint: Tuple[float, float],
    delaunator_data: DelaunatorLike,
    obstructs_func: Callable[[int], bool]
) -> Polygon:
    """
    Calculates the visibility polygon using the pre-computed triangulation.
    """
    qx, qy = viewpoint
    visibility_segments = triangular_expansion(delaunator_data, qx, qy, obstructs_func)
    if not visibility_segments:
        return Polygon()
    visible_triangles = []
    for seg in visibility_segments:
        triangle_coords = [viewpoint, (seg[0], seg[1]), (seg[2], seg[3])]
        visible_triangles.append(Polygon(triangle_coords))
    if not visible_triangles:
        return Polygon()
    raw_polygon = unary_union(visible_triangles)
    return raw_polygon.buffer(0)
# ==============================================================================
# FRONTIER CALCULATION (from Drone code)
# ==============================================================================
def get_windows(polygon, vis_polygon):
    """Finds visibility windows (frontiers) based on the *total shared map*."""
    clean_vis_polygon = vis_polygon.buffer(TOLERANCE).buffer(-TOLERANCE)
    if clean_vis_polygon.is_empty:
        return LineString()
    
    # This is the boundary of the *total explored area*
    multi_line_vis = clean_vis_polygon.boundary
    
    # This is the boundary of the *environment walls*
    environment_lines = polygon.boundary
    
    # Buffer the walls slightly
    buffered_environment = environment_lines.buffer(TOLERANCE)
    
    # The frontiers are the parts of the explored boundary
    # that are NOT walls.
    window_segments = multi_line_vis.difference(buffered_environment)
    return window_segments
# ==============================================================================
# --- (Removed Tree-Path functions: get_path_to_root, etc.) ---
# ==============================================================================

# ==============================================================================
# DRONE SIMULATION CLASSES AND FUNCTIONS
# ==============================================================================
class Drone:
    """A class to represent a single mobile drone with its own state."""
    def __init__(self, drone_id, initial_pos):
        self.id = drone_id
        self.pos = initial_pos
        self.status = 'available'  # 'available', 'exploring', 'stationary'
        self.parent = None         # Link to parent in the network tree
        self.vis_poly = None
        self.total_distance_traveled = 0.0
        # --- NEW: Tracks the drone's *final* destination for a batch ---
        self.current_task_target_pos = None 

    def __repr__(self):
        pos_str = f"({self.pos[0]:.2f}, {self.pos[1]:.2f})"
        dist_str = f"| Traveled: {self.total_distance_traveled:.2f} units"
        return f"Drone {self.id} @ {pos_str} ({self.status}) {dist_str}"

# ==============================================================================
# --- TIME-BASED ANIMATION HELPERS ---
# ==============================================================================
def path_distance(p1: Tuple, p2: Tuple) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_path_total_length(path_points: List[Tuple]) -> float:
    total_len = 0.0
    for i in range(len(path_points) - 1):
        total_len += path_distance(path_points[i], path_points[i+1])
    return total_len

def get_point_at_distance(path_points: List[Tuple], target_dist: float) -> Tuple[float, float]:
    if not path_points:
        return (0, 0)
    if target_dist <= 0:
        return path_points[0]
    total_dist = 0.0
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i+1]
        segment_len = path_distance(p1, p2)
        if total_dist + segment_len >= target_dist - TOLERANCE:
            dist_on_segment = target_dist - total_dist
            t = 0.0
            if segment_len > 0:
                t = dist_on_segment / segment_len
            interp_x = p1[0] * (1 - t) + p2[0] * t
            interp_y = p1[1] * (1 - t) + p2[1] * t
            return (interp_x, interp_y)
        total_dist += segment_len
    return path_points[-1]

# ==============================================================================
# ### A* GRAPH FUNCTIONS ###
# ==============================================================================

def get_triangle_centroid(t_idx: int, coords_flat: np.ndarray, triangles_flat: np.ndarray) -> PointT:
    """Gets the (x, y) centroid of a triangle index."""
    coords = coords_flat.reshape(-1, 2)
    p_indices = triangles_flat[t_idx*3 : t_idx*3 + 3]
    p1, p2, p3 = coords[p_indices]
    return ((p1[0] + p2[0] + p3[0]) / 3.0, (p1[1] + p2[1] + p3[1]) / 3.0)

def build_a_star_graph(d: DelaunatorLike, obstructs_func: Callable[[int], bool]) -> Tuple[AStarGraph, List[PointT]]:
    """Builds an adjacency list graph from the triangulation for A*."""
    print("     - Building A* navigation graph from triangulation...")
    graph: AStarGraph = {}
    
    num_triangles = len(d['triangles']) // 3
    coords_flat = d['coords']
    triangles_flat = d['triangles']
    halfedges = d['halfedges']
    
    # Pre-calculate all centroids
    centroids = [get_triangle_centroid(i, coords_flat, triangles_flat) for i in range(num_triangles)]
    
    for t_idx in range(num_triangles):
        if t_idx not in graph:
            graph[t_idx] = []
        
        # Check all 3 edges of the triangle
        for edge_idx in [t_idx * 3, t_idx * 3 + 1, t_idx * 3 + 2]:
            neighbor_edge_idx = halfedges[edge_idx]
            
            # Check if edge leads to a neighbor (not -1) and is not a wall
            if neighbor_edge_idx != -1 and not obstructs_func(edge_idx):
                neighbor_t_idx = neighbor_edge_idx // 3
                
                # Cost is the distance between centroids
                cost = path_distance(centroids[t_idx], centroids[neighbor_t_idx])
                
                graph[t_idx].append((neighbor_t_idx, cost))
                
    print(f"     - A* graph complete. {len(graph)} nodes.")
    return graph, centroids


def find_triangle_for_point(d: DelaunatorLike, qx: float, qy: float) -> int:
    """Finds the triangle containing a point by a brute-force check."""
    coords = d['coords'].reshape(-1, 2)
    q = (qx, qy)
    num_triangles = len(d['triangles']) // 3

    for t_idx in range(num_triangles):
        p_indices = points_of_tri(d, t_idx)
        p1, p2, p3 = coords[p_indices]
        if orient2d(p1, p2, q) >= 0 and orient2d(p2, p3, q) >= 0 and orient2d(p3, p1, q) >= 0:
            return t_idx
    
    # Fallback: if outside, find the *closest* triangle centroid
    print(f"     ! Warning: Point {q} is outside triangulation. Finding closest centroid.")
    min_dist = float('inf')
    closest_t_idx = -1
    centroids = [get_triangle_centroid(i, d['coords'], d['triangles']) for i in range(num_triangles)]
    for t_idx, centroid in enumerate(centroids):
        dist = path_distance(q, centroid)
        if dist < min_dist:
            min_dist = dist
            closest_t_idx = t_idx
    return closest_t_idx

def a_star_search(
    start_t: int, 
    end_t: int, 
    graph: AStarGraph, 
    centroids: List[PointT]
) -> Tuple[List[int], float]:
    """Finds the shortest path of triangles from start_t to end_t."""
    
    if start_t == -1 or end_t == -1:
        print("     ! A* Error: Start or End triangle not found.")
        return ([], 0.0)
    
    if start_t == end_t:
        return ([start_t], 0.0)

    end_centroid = centroids[end_t]
    
    pq: List[Tuple[float, float, int, List[int]]] = []
    # (priority, cost_g, current_t_idx, path_list)
    heapq.heappush(pq, (0.0, 0.0, start_t, [start_t]))
    
    visited = set()

    while pq:
        priority, cost_g, current_t, path = heapq.heappop(pq)
        
        if current_t in visited:
            continue
        visited.add(current_t)
        
        if current_t == end_t:
            return (path, cost_g) # Found it!

        for neighbor_t, move_cost in graph.get(current_t, []):
            if neighbor_t not in visited:
                new_cost_g = cost_g + move_cost
                # Heuristic: Euclidean distance from neighbor to end
                cost_h = path_distance(centroids[neighbor_t], end_centroid)
                new_priority = new_cost_g + cost_h
                
                new_path = path + [neighbor_t]
                heapq.heappush(pq, (new_priority, new_cost_g, neighbor_t, new_path))
                
    print(f"     ! A* Error: No path found from {start_t} to {end_t}.")
    return ([], 0.0) # No path found

def get_a_star_waypoint_path(
    start_pos: PointT, 
    end_pos: PointT,
    d: DelaunatorLike,
    graph: AStarGraph,
    centroids: List[PointT]
) -> Tuple[List[PointT], float]:
    """High-level wrapper to get a full (x,y) waypoint path."""
    
    start_t = find_triangle_for_point(d, start_pos[0], start_pos[1])
    end_t = find_triangle_for_point(d, end_pos[0], end_pos[1])
    
    if start_t == -1 or end_t == -1:
        # Point is outside, just return a straight line
        return ([start_pos, end_pos], path_distance(start_pos, end_pos))

    triangle_path, path_cost = a_star_search(start_t, end_t, graph, centroids)
    
    if not triangle_path:
        # No path found, return straight line as a (bad) fallback
        return ([start_pos, end_pos], path_distance(start_pos, end_pos))

    # --- Simple Path: Start + Centroids + End ---
    # We can add a "String Pulling" algorithm here later for optimization
    waypoints = [start_pos]
    waypoints.extend([centroids[t] for t in triangle_path[1:-1]]) # Add intermediate centroids
    waypoints.append(end_pos)
    
    # Recalculate true path length
    total_length = get_path_total_length(waypoints)
    
    return (waypoints, total_length)


# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================
def update_plot(ax, polygon, drone_fleet, shared_map, title_text, frontiers=None, evaluated_tasks=None, winning_task=None):
    """Clears and redraws the Matplotlib plot with enhanced information."""
    ax.clear()
    
    # zorder 0: Base environment fill
    ox, oy = polygon.exterior.xy
    ax.fill(ox, oy, color='lightgray', zorder=0)

    # zorder 1 & 2: Explored area (yellow) and its "holes"
    polygons_to_plot = []
    if shared_map.is_empty:
        pass
    elif shared_map.geom_type == 'Polygon':
        polygons_to_plot = [shared_map]
    elif shared_map.geom_type == 'MultiPolygon':
        polygons_to_plot = list(shared_map.geoms)
    
    for i, poly in enumerate(polygons_to_plot):
        if poly.is_empty or poly.geom_type != 'Polygon': continue
        
        vx, vy = poly.exterior.xy
        ax.fill(vx, vy, color='gold', alpha=0.5, label='Total Coverage' if i == 0 else '_nolegend_', zorder=1)
        
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, color='lightgray', zorder=2)

    # zorder 3: Environment hole (obstacle) fills
    for interior in polygon.interiors:
        hx, hy = interior.xy
        ax.fill(hx, hy, color='dimgray', zorder=3)
        
    # zorder 4: Environment boundaries (exterior AND holes)
    ax.plot(ox, oy, 'k-', linewidth=1, zorder=4)
    for interior in polygon.interiors:
        hx, hy = interior.xy
        ax.plot(hx, hy, 'k-', linewidth=1, zorder=4)
        
    # zorder 5: Network links
    for i, drone in enumerate(drone_fleet):
        if drone.status == 'stationary' and drone.parent:
            p_pos, c_pos = drone.parent.pos, drone.pos
            ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], 'k-', linewidth=1.5, zorder=5, label='Network Link' if i == 1 else '_nolegend_')
    
    # zorder 6: Frontiers
    if frontiers:
        if frontiers.geom_type == 'MultiLineString':
            for i, frontier in enumerate(frontiers.geoms):
                x, y = frontier.xy
                ax.plot(x, y, 'c--', linewidth=2, label='Frontiers' if i == 0 else '_nolegend_', zorder=6)
        elif frontiers.geom_type == 'LineString':
                 x, y = frontiers.xy
                 ax.plot(x, y, 'c--', linewidth=2, label='Frontiers', zorder=6)

    # zorder 7: Evaluated tasks text
    if evaluated_tasks:
        for task in evaluated_tasks:
            if 'pos' in task:
                ax.text(task['pos'][0], task['pos'][1] + 0.3, f"{task['gain']:.1f}", color='blue', fontsize=9, ha='center', fontweight='bold', zorder=7)
    
    # zorder 8: Winning task highlight
    if winning_task and 'pos' in winning_task:
       ax.plot(winning_task['pos'][0], winning_task['pos'][1], 'go', markersize=12, markerfacecolor='none', markeredgewidth=3, label='Winning Task', zorder=8)
           
    # zorder 9 & 10: Drones
    for drone in drone_fleet:
        if drone.status == 'stationary': 
            color, marker, size, label = 'red', 'o', 10, 'Stationary Drone'
        elif drone.status == 'exploring': 
            color, marker, size, label = 'blue', 's', 8, 'Exploring Drone'
        else: # 'available'
            color, marker, size, label = 'dimgray', 'x', 6, 'Available (Idle) Drone'
        
        drone_pos = drone.pos

        ax.plot(drone_pos[0], drone_pos[1], marker, color=color, markersize=size, label=label, zorder=9)
        ax.text(drone_pos[0] + 0.2, drone_pos[1] + 0.2, f'D{drone.id}', color=color, fontsize=10, fontweight='bold', zorder=10)
    
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title_text, fontsize=16)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.draw()
    plt.pause(0.01) # Use a very short pause for animations

# ==============================================================================
# --- TIME-BASED ANIMATION FUNCTIONS ---
# ==============================================================================

def animate_single_drone_path(
    drone: Drone,
    path_points: List[Tuple[float, float]],
    title_suffix: str,
    # --- We must pass all plotting objects to the animator ---
    ax, polygon, drone_fleet, shared_map, frontiers, round_num,
    evaluated_tasks=None, winning_task=None # Added for flexibility
):
    """Animates a *single* drone moving along a path of waypoints based on time."""
    
    drone.status = 'exploring'
    if not path_points: return
        
    total_path_len = get_path_total_length(path_points)
    if total_path_len == 0:
        return

    for step in range(TOTAL_ANIMATION_STEPS + 1):
        t = step / TOTAL_ANIMATION_STEPS # 0.0 to 1.0
        
        target_dist = t * total_path_len
        drone.pos = get_point_at_distance(path_points, target_dist)
        
        update_plot(ax, polygon, drone_fleet, shared_map, 
                    f"Round {round_num}: {title_suffix}",
                    frontiers=frontiers,
                    evaluated_tasks=evaluated_tasks,
                    winning_task=winning_task)

def animate_drones_parallel(
    animation_data_list: List[Dict],
    title_suffix: str,
    # is_return_trip: bool, # <-- Replaced with path_key
    path_key: str, # 'out_path_points' or 'in_path_points'
    # --- Plotting objects ---
    ax, polygon, drone_fleet, shared_map, frontiers, round_num
):
    """Animates *multiple* drones moving simultaneously based on time."""
    
    # 1. Set all drones to 'exploring' and find the longest path
    max_path_length = 0.0
    for data in animation_data_list:
        data['drone'].status = 'exploring'
        if path_key not in data:
            data['path_points'] = []
            data['total_path_len'] = 0.0
            continue
            
        data['path_points'] = data[path_key]
        data['total_path_len'] = get_path_total_length(data['path_points'])
        
        max_path_length = max(max_path_length, data['total_path_len'])
    
    if max_path_length <= 0:
        print("     - (Animation skipped: No paths to animate)")
        return

    # 2. Master loop: Iterate from t=0.0 to t=1.0
    for step in range(TOTAL_ANIMATION_STEPS + 1):
        t_global = step / TOTAL_ANIMATION_STEPS # 0.0 to 1.0
        current_global_dist = t_global * max_path_length
        
        # Update ALL drones
        for data in animation_data_list:
            if not data['path_points']:
                continue
                
            drone = data['drone']
            target_dist = min(current_global_dist, data['total_path_len'])
            drone.pos = get_point_at_distance(data['path_points'], target_dist)
        
        title = f"Round {round_num}: {title_suffix}"
        update_plot(ax, polygon, drone_fleet, shared_map, 
                    title, frontiers=frontiers)

# ==============================================================================
# --- MAIN SIMULATION BLOCK ---
# ==============================================================================
if __name__ == "__main__":
    simulation_start_time = time.perf_counter()
    
    # 1. Define the Environment Polygon
    outer_boundary2 = [(0, 3),(0,11),(5,11),(5,12),(0,12),(0,21),(5,21),(5,16),(12,16),(12,17),(6,17),
                       (6,21),(21,21),(21,17),(15,17),(15,16),(21,16),(21,9),(15,9),(15,6),(16,6),
                       (16,7),(21,7),(21,0),(16,0),(16,1),(15,1),(15,0),(9,0),(9,8),(10,8),(10,9),
                       (9,9),(9,12),(8,12),(8,9),(7,9),(7,8),(8,8),(8,0),(3,0),(3,3)]
    holes = []
    
    holes_cathedral = [[[6.9375, 4.546875], [4.796875, 4.546875], [4.1875, 6.984375], [6.328125, 6.984375], [6.9375, 4.546875]], [[4.03125, 1.171875], [3.2695309999999997, 1.171875], [2.695313, 1.546875], [2.3125, 2.296875], [1.9257810000000002, 3.046875], [1.734375, 4.179687], [1.734375, 5.6875], [1.734375, 7.1875], [1.9257810000000002, 8.3125], [2.3125, 9.0625], [2.695313, 9.8125], [3.2695309999999997, 10.1875], [4.03125, 10.1875], [4.800781, 10.1875], [5.375, 9.8125], [5.75, 9.0625], [6.132813, 8.3125], [6.328125, 7.1875], [6.328125, 5.6875], [6.328125, 4.179687], [6.132813, 3.046875], [5.75, 2.296875], [5.375, 1.546875], [4.800781, 1.171875], [4.03125, 1.171875]], [[4.03125, 0.0], [5.257813, 0.0], [6.195313, 0.4921869999999995], [6.84375, 1.46875], [7.488281000000001, 2.4375], [7.8125, 3.84375], [7.8125, 5.6875], [7.8125, 7.5234369999999995], [7.488281000000001, 8.929687], [6.84375, 9.90625], [6.195313, 10.875], [5.257813, 11.359375], [4.03125, 11.359375], [2.8125, 11.359375], [1.875, 10.875], [1.21875, 9.90625], [0.5703130000000001, 8.929687], [0.25, 7.5234369999999995], [0.25, 5.6875], [0.25, 3.84375], [0.5703130000000001, 2.4375], [1.21875, 1.46875], [1.875, 0.4921869999999995], [2.8125, 0.0], [4.03125, 0.0]], [[0.0, 6.4375], [3.953125, 6.4375], [3.953125, 7.625], [0.0, 7.625], [0.0, 6.4375]], [[518.765625, 909.214844], [503.945313, 909.214844], [495.644531, 900.324219], [496.238281, 883.132813], [504.535156, 872.464844], [518.765625, 873.058594], [529.433594, 883.132813], [528.839844, 900.324219], [518.765625, 909.214844]], [[606.816406, 909.214844], [591.996094, 909.214844], [583.695313, 900.324219], [584.289063, 883.132813], [592.585937, 872.464844], [606.816406, 873.058594], [617.484375, 883.132813], [616.890625, 900.324219], [606.816406, 909.214844]], [[341.957031, 909.214844], [327.136719, 909.214844], [318.835937, 900.324219], [319.429687, 883.132813], [327.726563, 872.464844], [341.957031, 873.058594], [352.625, 883.132813], [352.03125, 900.324219], [341.957031, 909.214844]], [[430.007813, 909.214844], [415.1875, 909.214844], [406.886719, 900.324219], [407.480469, 883.132813], [415.777344, 872.464844], [430.007813, 873.058594], [440.675781, 883.132813], [440.082031, 900.324219], [430.007813, 909.214844]], [[872.382813, 909.214844], [857.5625, 909.214844], [849.265625, 900.324219], [849.859375, 883.132813], [858.152344, 872.464844], [872.382813, 873.058594], [883.050781, 883.132813], [882.457031, 900.324219], [872.382813, 909.214844]], [[960.433594, 909.214844], [945.613281, 909.214844], [937.316406, 900.324219], [937.910156, 883.132813], [946.203125, 872.464844], [960.433594, 873.058594], [971.101563, 883.132813], [970.507813, 900.324219], [960.433594, 909.214844]], [[695.574219, 909.214844], [680.753906, 909.214844], [672.453125, 900.324219], [673.046875, 883.132813], [681.34375, 872.464844], [695.574219, 873.058594], [706.242187, 883.132813], [705.648437, 900.324219], [695.574219, 909.214844]], [[783.625, 909.214844], [768.804687, 909.214844], [760.503906, 900.324219], [761.101563, 883.132813], [769.394531, 872.464844], [783.625, 873.058594], [794.292969, 883.132813], [793.699219, 900.324219], [783.625, 909.214844]], [[1130.523437, 909.214844], [1115.703125, 909.214844], [1107.40625, 900.324219], [1108.0, 883.132813], [1116.292969, 872.464844], [1130.523437, 873.058594], [1141.191406, 883.132813], [1140.597656, 900.324219], [1130.523437, 909.214844]], [[1212.917969, 909.214844], [1198.097656, 909.214844], [1189.796875, 900.324219], [1190.390625, 883.132813], [1198.6875, 872.464844], [1212.917969, 873.058594], [1223.585937, 883.132813], [1222.992187, 900.324219], [1212.917969, 909.214844]], [[1047.425781, 909.214844], [1032.605469, 909.214844], [1024.304687, 900.324219], [1024.898437, 883.132813], [1033.195313, 872.464844], [1047.425781, 873.058594], [1058.09375, 883.132813], [1057.5, 900.324219], [1047.425781, 909.214844]], [[521.652344, 464.183594], [531.730469, 473.511719], [532.320313, 491.550781], [521.652344, 502.125], [507.429687, 502.75], [499.128906, 491.550781], [498.535156, 473.511719], [506.835937, 464.183594], [521.652344, 464.183594]], [[609.695313, 464.183594], [619.769531, 473.511719], [620.363281, 491.550781], [609.695313, 502.125], [595.46875, 502.75], [587.171875, 491.550781], [586.578125, 473.511719], [594.878906, 464.183594], [609.695313, 464.183594]], [[344.859375, 464.183594], [354.9375, 473.511719], [355.527344, 491.550781], [344.859375, 502.125], [330.632813, 502.75], [322.335937, 491.550781], [321.742187, 473.511719], [330.042969, 464.183594], [344.859375, 464.183594]], [[432.902344, 464.183594], [442.976563, 473.511719], [443.570313, 491.550781], [432.902344, 502.125], [418.675781, 502.75], [410.378906, 491.550781], [409.789063, 473.511719], [418.082031, 464.183594], [432.902344, 464.183594]], [[875.238281, 464.183594], [885.3125, 473.511719], [885.90625, 491.550781], [875.238281, 502.125], [861.011719, 502.75], [852.714844, 491.550781], [852.121094, 473.511719], [860.421875, 464.183594], [875.238281, 464.183594]], [[963.28125, 464.183594], [973.355469, 473.511719], [973.949219, 491.550781], [963.28125, 502.125], [949.054687, 502.75], [940.753906, 491.550781], [940.164063, 473.511719], [948.460937, 464.183594], [963.28125, 464.183594]], [[698.445313, 464.183594], [708.519531, 473.511719], [709.113281, 491.550781], [698.445313, 502.125], [684.21875, 502.75], [675.921875, 491.550781], [675.328125, 473.511719], [683.625, 464.183594], [698.445313, 464.183594]], [[786.484375, 464.183594], [796.5625, 473.511719], [797.15625, 491.550781], [786.484375, 502.125], [772.261719, 502.75], [763.964844, 491.550781], [763.371094, 473.511719], [771.667969, 464.183594], [786.484375, 464.183594]], [[1133.351563, 464.183594], [1143.429687, 473.511719], [1144.023437, 491.550781], [1133.351563, 502.125], [1119.128906, 502.75], [1110.832031, 491.550781], [1110.238281, 473.511719], [1118.535156, 464.183594], [1133.351563, 464.183594]], [[1215.738281, 464.183594], [1225.8125, 473.511719], [1226.40625, 491.550781], [1215.738281, 502.125], [1201.511719, 502.75], [1193.21875, 491.550781], [1192.625, 473.511719], [1200.917969, 464.183594], [1215.738281, 464.183594]], [[1050.261719, 464.183594], [1060.339844, 473.511719], [1060.929687, 491.550781], [1050.261719, 502.125], [1036.039063, 502.75], [1027.738281, 491.550781], [1027.144531, 473.511719], [1035.445313, 464.183594], [1050.261719, 464.183594]], [[518.765625, 808.789063], [503.945313, 808.789063], [495.644531, 799.894531], [496.238281, 782.703125], [504.535156, 772.035156], [518.765625, 772.625], [529.433594, 782.703125], [528.839844, 799.894531], [518.765625, 808.789063]], [[606.816406, 808.789063], [591.996094, 808.789063], [583.695313, 799.894531], [584.289063, 782.703125], [592.585937, 772.035156], [606.816406, 772.625], [617.484375, 782.703125], [616.890625, 799.894531], [606.816406, 808.789063]], [[341.957031, 808.789063], [327.136719, 808.789063], [318.835937, 799.894531], [319.429687, 782.703125], [327.726563, 772.035156], [341.957031, 772.625], [352.625, 782.703125], [352.03125, 799.894531], [341.957031, 808.789063]], [[430.007813, 808.789063], [415.1875, 808.789063], [406.886719, 799.894531], [407.480469, 782.703125], [415.777344, 772.035156], [430.007813, 772.625], [440.675781, 782.703125], [440.082031, 799.894531], [430.007813, 808.789063]], [[872.382813, 808.789063], [857.5625, 808.789063], [849.265625, 799.894531], [849.859375, 782.703125], [858.152344, 772.035156], [872.382813, 772.625], [883.050781, 782.703125], [882.457031, 799.894531], [872.382813, 808.789063]], [[960.433594, 808.789063], [945.613281, 808.789063], [937.316406, 799.894531], [937.910156, 782.703125], [946.203125, 772.035156], [960.433594, 772.625], [971.101563, 782.703125], [970.507813, 799.894531], [960.433594, 808.789063]], [[695.574219, 808.789063], [680.753906, 808.789063], [672.453125, 799.894531], [673.046875, 782.703125], [681.34375, 772.035156], [695.574219, 772.625], [706.242187, 782.703125], [705.648437, 799.894531], [695.574219, 808.789063]], [[783.625, 808.789063], [768.804687, 808.789063], [760.503906, 799.894531], [761.101563, 782.703125], [769.394531, 772.035156], [783.625, 772.625], [794.292969, 782.703125], [793.699219, 799.894531], [783.625, 808.789063]], [[1130.523437, 808.789063], [1115.703125, 808.789063], [1107.40625, 799.894531], [1108.0, 782.703125], [1116.292969, 772.035156], [1130.523437, 772.625], [1141.191406, 782.703125], [1140.597656, 799.894531], [1130.523437, 808.789063]], [[1212.917969, 808.789063], [1198.097656, 808.789063], [1189.796875, 799.894531], [1190.390625, 782.703125], [1198.6875, 772.035156], [1212.917969, 772.625], [1223.585937, 782.703125], [1222.992187, 799.894531], [1212.917969, 808.789063]], [[1047.425781, 808.789063], [1032.605469, 808.789063], [1024.304687, 799.894531], [1024.898437, 782.703125], [1033.195313, 772.035156], [1047.425781, 772.625], [1058.09375, 782.703125], [1057.5, 799.894531], [1047.425781, 808.789063]], [[521.652344, 564.257813], [531.730469, 573.585937], [532.320313, 591.625], [521.652344, 602.199219], [507.429687, 602.824219], [499.128906, 591.625], [498.535156, 573.585937], [506.835937, 564.257813], [521.652344, 564.257813]], [[609.695313, 564.257813], [619.769531, 573.585937], [620.363281, 591.625], [609.695313, 602.199219], [595.46875, 602.824219], [587.171875, 591.625], [586.578125, 573.585937], [594.878906, 564.257813], [609.695313, 564.257813]], [[344.859375, 564.257813], [354.9375, 573.585937], [355.527344, 591.625], [344.859375, 602.199219], [330.632813, 602.824219], [322.335937, 591.625], [321.742187, 573.585937], [330.042969, 564.257813], [344.859375, 564.257813]], [[432.902344, 564.257813], [442.976563, 573.585937], [443.570313, 591.625], [432.902344, 602.199219], [418.675781, 602.824219], [410.378906, 591.625], [409.789063, 573.585937], [418.082031, 564.257813], [432.902344, 564.257813]], [[875.238281, 564.257813], [885.3125, 573.585937], [885.90625, 591.625], [875.238281, 602.199219], [861.011719, 602.824219], [852.714844, 591.625], [852.121094, 573.585937], [860.421875, 564.257813], [875.238281, 564.257813]], [[963.28125, 564.257813], [973.355469, 573.585937], [973.949219, 591.625], [963.28125, 602.199219], [949.054687, 602.824219], [940.753906, 591.625], [940.164063, 573.585937], [948.460937, 564.257813], [963.28125, 564.257813]], [[698.445313, 564.257813], [708.519531, 573.585937], [709.113281, 591.625], [698.445313, 602.199219], [684.21875, 602.824219], [675.921875, 591.625], [675.328125, 573.585937], [683.625, 564.257813], [698.445313, 564.257813]], [[786.484375, 564.257813], [796.5625, 573.585937], [797.15625, 591.625], [786.484375, 602.199219], [772.261719, 602.824219], [763.964844, 591.625], [763.371094, 573.585937], [771.667969, 564.257813], [786.484375, 564.257813]], [[1133.351563, 564.257813], [1143.429687, 573.585937], [1144.023437, 591.625], [1133.351563, 602.199219], [1119.128906, 602.824219], [1110.832031, 591.625], [1110.238281, 573.585937], [1118.535156, 564.257813], [1133.351563, 564.257813]], [[1215.738281, 564.257813], [1225.8125, 573.585937], [1226.40625, 591.625], [1215.738281, 602.199219], [1201.511719, 602.824219], [1193.21875, 591.625], [1192.625, 573.585937], [1200.917969, 564.257813], [1215.738281, 564.257813]], [[1050.261719, 564.257813], [1060.339844, 573.585937], [1060.929687, 591.625], [1050.261719, 602.199219], [1036.039063, 602.824219], [1027.738281, 591.625], [1027.144531, 573.585937], [1035.445313, 564.257813], [1050.261719, 564.257813]], [[1494.890625, 818.917969], [1466.589844, 818.917969], [1445.363281, 792.972656], [1445.363281, 761.136719], [1466.589844, 735.785156], [1494.300781, 736.375], [1519.652344, 761.136719], [1519.652344, 792.972656], [1494.890625, 818.917969]], [[1325.476563, 818.917969], [1297.175781, 818.917969], [1272.414063, 792.972656], [1272.414063, 761.136719], [1297.765625, 736.375], [1325.476563, 735.785156], [1346.703125, 761.136719], [1346.703125, 792.972656], [1325.476563, 818.917969]], [[1494.894531, 547.503906], [1519.660156, 573.445313], [1519.660156, 605.28125], [1494.304687, 630.042969], [1466.597656, 630.636719], [1445.371094, 605.28125], [1445.371094, 573.445313], [1466.597656, 547.503906], [1494.894531, 547.503906]], [[1272.414063, 573.445313], [1297.175781, 547.503906], [1325.476563, 547.503906], [1346.703125, 573.445313], [1346.703125, 605.28125], [1325.476563, 630.636719], [1297.765625, 630.042969], [1272.414063, 605.28125], [1272.414063, 573.445313]], [[1397.355469, 1178.621094], [1384.5625, 1178.621094], [1377.402344, 1170.851563], [1377.914063, 1155.824219], [1385.078125, 1146.5], [1397.355469, 1147.015625], [1406.5625, 1155.824219], [1406.050781, 1170.851563], [1397.355469, 1178.621094]], [[1486.484375, 1180.425781], [1471.664063, 1180.425781], [1463.363281, 1171.535156], [1463.957031, 1154.347656], [1472.257813, 1143.675781], [1486.484375, 1144.269531], [1497.152344, 1154.347656], [1496.5625, 1171.535156], [1486.484375, 1180.425781]], [[1313.917969, 1180.425781], [1299.097656, 1180.425781], [1290.796875, 1171.535156], [1291.390625, 1154.347656], [1299.691406, 1143.675781], [1313.917969, 1144.269531], [1324.585937, 1154.347656], [1323.996094, 1171.535156], [1313.917969, 1180.425781]], [[1312.492187, 1084.933594], [1299.703125, 1084.933594], [1292.542969, 1077.167969], [1293.050781, 1062.136719], [1300.214844, 1052.8125], [1312.492187, 1053.328125], [1321.699219, 1062.136719], [1321.191406, 1077.167969], [1312.492187, 1084.933594]], [[1312.492187, 988.046875], [1299.703125, 988.046875], [1292.542969, 980.273437], [1293.050781, 965.25], [1300.214844, 955.921875], [1312.492187, 956.441406], [1321.699219, 965.25], [1321.191406, 980.273437], [1312.492187, 988.046875]], [[1489.28125, 1084.9375], [1476.488281, 1084.9375], [1467.792969, 1077.167969], [1467.28125, 1062.136719], [1476.488281, 1053.328125], [1488.769531, 1052.8125], [1495.933594, 1062.136719], [1496.445313, 1077.167969], [1489.28125, 1084.9375]], [[1489.28125, 988.046875], [1476.488281, 988.046875], [1467.792969, 980.273437], [1467.28125, 965.246094], [1476.488281, 956.4375], [1488.769531, 955.921875], [1495.933594, 965.246094], [1496.445313, 980.273437], [1489.28125, 988.046875]], [[1314.753906, 898.183594], [1296.484375, 897.351563], [1290.257813, 892.367187], [1290.257813, 873.6875], [1296.902344, 852.929687], [1315.996094, 852.929687], [1325.546875, 882.402344], [1314.753906, 898.183594]], [[1493.582031, 897.351563], [1475.3125, 898.183594], [1464.515625, 882.402344], [1474.0625, 852.929687], [1493.164063, 852.929687], [1499.808594, 873.6875], [1499.808594, 892.367187], [1493.582031, 897.351563]], [[1398.773437, 186.34375], [1407.46875, 194.113281], [1407.980469, 209.140625], [1398.773437, 217.949219], [1386.496094, 218.46875], [1379.332031, 209.140625], [1378.820313, 194.113281], [1385.980469, 186.34375], [1398.773437, 186.34375]], [[1487.902344, 184.539063], [1497.980469, 193.429687], [1498.570313, 210.621094], [1487.902344, 220.695313], [1473.675781, 221.289063], [1465.375, 210.621094], [1464.78125, 193.429687], [1473.082031, 184.539063], [1487.902344, 184.539063]], [[1315.335937, 184.539063], [1325.414063, 193.429687], [1326.003906, 210.621094], [1315.335937, 220.695313], [1301.109375, 221.289063], [1292.808594, 210.621094], [1292.214844, 193.429687], [1300.515625, 184.539063], [1315.335937, 184.539063]], [[1313.910156, 280.03125], [1322.609375, 287.800781], [1323.117187, 302.828125], [1313.910156, 311.636719], [1301.632813, 312.152344], [1294.46875, 302.828125], [1293.960937, 287.800781], [1301.121094, 280.03125], [1313.910156, 280.03125]], [[1313.910156, 376.917969], [1322.609375, 384.691406], [1323.117187, 399.71875], [1313.910156, 408.527344], [1301.632813, 409.042969], [1294.46875, 399.71875], [1293.960937, 384.691406], [1301.121094, 376.917969], [1313.910156, 376.917969]], [[1469.210937, 287.796875], [1477.910156, 280.027344], [1490.699219, 280.027344], [1497.863281, 287.796875], [1497.351563, 302.828125], [1490.1875, 312.152344], [1477.910156, 311.636719], [1468.699219, 302.828125], [1469.210937, 287.796875]], [[1469.210937, 384.691406], [1477.910156, 376.921875], [1490.699219, 376.921875], [1497.863281, 384.691406], [1497.351563, 399.71875], [1490.1875, 409.042969], [1477.910156, 408.527344], [1468.699219, 399.71875], [1469.210937, 384.691406]], [[1316.171875, 466.78125], [1326.964844, 482.5625], [1317.414063, 512.0390629999999], [1298.320313, 512.0390629999999], [1291.675781, 491.277344], [1291.675781, 472.597656], [1297.902344, 467.613281], [1316.171875, 466.78125]], [[1465.933594, 482.5625], [1476.730469, 466.78125], [1495.0, 467.613281], [1501.226563, 472.597656], [1501.226563, 491.277344], [1494.582031, 512.0351559999999], [1475.480469, 512.0351559999999], [1465.933594, 482.5625]], [[1585.738281, 802.679687], [1572.945313, 802.679687], [1564.25, 794.910156], [1563.738281, 779.882813], [1572.945313, 771.074219], [1585.226563, 770.554687], [1592.386719, 779.882813], [1592.902344, 794.910156], [1585.738281, 802.679687]], [[1665.59375, 798.414063], [1654.710937, 798.414063], [1646.730469, 789.703125], [1646.730469, 774.460937], [1656.886719, 767.207031], [1665.59375, 767.207031], [1674.304687, 776.640625], [1674.304687, 787.523437], [1665.59375, 798.414063]], [[1712.042969, 775.1875], [1724.375, 775.914063], [1723.652344, 790.429687], [1711.3125, 791.152344], [1712.042969, 775.1875]], [[1744.699219, 773.011719], [1756.308594, 767.207031], [1766.46875, 779.542969], [1748.328125, 790.429687], [1744.699219, 773.011719]], [[1780.253906, 746.160156], [1786.785156, 741.082031], [1802.027344, 749.066406], [1789.691406, 761.398437], [1780.253906, 746.160156]], [[1800.578125, 708.425781], [1815.089844, 708.425781], [1815.089844, 720.035156], [1799.125, 719.308594], [1800.578125, 708.425781]], [[1567.765625, 580.792969], [1576.464844, 573.019531], [1589.253906, 573.019531], [1596.417969, 580.792969], [1595.90625, 595.816406], [1588.746094, 605.144531], [1576.464844, 604.625], [1567.253906, 595.816406], [1567.765625, 580.792969]], [[1669.128906, 577.289063], [1677.835937, 588.171875], [1677.835937, 599.058594], [1669.128906, 608.492187], [1660.417969, 608.492187], [1650.253906, 601.234375], [1650.253906, 585.996094], [1658.238281, 577.289063], [1669.128906, 577.289063]], [[1712.675781, 601.960937], [1711.949219, 585.996094], [1724.292969, 586.722656], [1725.015625, 601.234375], [1712.675781, 601.960937]], [[1751.144531, 607.765625], [1754.773437, 590.351563], [1772.917969, 601.234375], [1762.757813, 613.574219], [1751.144531, 607.765625]], [[1785.984375, 636.796875], [1795.417969, 621.554687], [1807.757813, 633.890625], [1792.515625, 641.871094], [1785.984375, 636.796875]], [[1799.773437, 675.984375], [1798.320313, 665.09375], [1814.289063, 664.371094], [1814.289063, 675.984375], [1799.773437, 675.984375]]]
    outer_boundary3 = [[925.929687, 357.847656], [903.554687, 357.847656], [903.554687, 382.414063], [877.109375, 402.4375], [859.742187, 402.4375], [837.839844, 377.089844], [837.839844, 357.847656], [811.234375, 357.6875], [811.078125, 382.570313], [793.085937, 402.28125], [768.207031, 402.28125], [749.117187, 377.40625], [749.117187, 358.003906], [722.675781, 358.003906], [722.359375, 388.203125], [703.898437, 402.910156], [683.558594, 402.753906], [678.082031, 397.121094], [678.550781, 372.082031], [674.011719, 372.082031], [673.855469, 357.53125], [629.105469, 357.6875], [629.105469, 367.230469], [620.65625, 367.230469], [620.1875, 397.121094], [615.492187, 402.910156], [593.433594, 402.4375], [576.84375, 392.738281], [577.488281, 357.859375], [550.363281, 357.859375], [550.957031, 381.441406], [536.808594, 397.949219], [500.25, 397.359375], [486.6875, 380.261719], [487.277344, 356.675781], [460.160156, 356.675781], [460.160156, 382.621094], [445.417969, 399.71875], [414.171875, 398.539063], [397.070313, 383.207031], [397.660156, 357.859375], [369.363281, 357.859375], [369.363281, 382.621094], [355.210937, 399.71875], [320.425781, 399.71875], [305.09375, 375.546875], [305.6875, 363.753906], [305.6875, 358.449219], [281.511719, 358.449219], [281.511719, 383.800781], [270.898437, 395.59375], [249.085937, 395.59375], [249.085937, 403.84375], [202.507813, 403.84375], [202.507813, 396.769531], [225.503906, 395.59375], [226.089844, 384.390625], [219.015625, 372.597656], [197.789063, 372.597656], [183.640625, 386.75], [184.226563, 410.332031], [193.664063, 420.351563], [242.488281, 421.804687], [243.808594, 485.394531], [223.941406, 485.394531], [223.941406, 429.089844], [175.585937, 430.417969], [175.585937, 392.660156], [170.289063, 392.0], [165.648437, 358.21875], [154.390625, 358.21875], [145.78125, 392.660156], [117.960937, 392.660156], [108.683594, 357.554687], [99.414063, 357.554687], [84.324219, 391.851563], [85.5, 391.335937], [84.839844, 432.402344], [59.007813, 432.402344], [28.542969, 444.988281], [28.542969, 454.921875], [55.035156, 465.519531], [56.359375, 497.980469], [29.203125, 506.589844], [29.203125, 517.1875], [57.683594, 531.09375], [86.167969, 531.09375], [86.167969, 558.253906], [94.777344, 558.253906], [95.4375, 575.472656], [173.597656, 575.472656], [172.9375, 559.578125], [178.898437, 558.917969], [178.234375, 525.796875], [223.941406, 526.460937], [225.265625, 512.5507809999999], [243.808594, 513.2109370000001], [243.808594, 570.175781], [254.40625, 576.800781], [254.40625, 594.019531], [245.796875, 611.242187], [220.625, 610.582031], [208.707031, 599.984375], [209.367187, 588.058594], [150.414063, 588.058594], [150.414063, 599.320313], [142.464844, 611.242187], [125.910156, 610.582031], [115.972656, 603.296875], [115.3125, 593.359375], [76.230469, 593.359375], [76.230469, 583.421875], [60.335937, 584.089844], [47.085937, 613.230469], [19.925781, 613.230469], [19.265625, 665.558594], [40.460937, 667.546875], [52.386719, 683.445313], [52.386719, 697.351563], [39.800781, 716.5625], [21.253906, 717.886719], [19.925781, 770.214844], [41.789063, 770.214844], [53.707031, 790.746094], [89.476563, 790.085937], [90.140625, 820.550781], [108.023437, 820.550781], [108.023437, 792.734375], [123.921875, 770.214844], [140.480469, 770.214844], [159.027344, 791.410156], [158.363281, 821.878906], [172.277344, 822.539063], [172.277344, 790.085937], [198.769531, 789.421875], [214.667969, 770.214844], [238.511719, 770.214844], [249.773437, 782.136719], [250.4375, 802.671875], [241.160156, 807.972656], [241.160156, 868.246094], [211.355469, 868.90625], [211.355469, 854.332031], [175.585937, 855.0], [176.910156, 831.816406], [81.53125, 831.152344], [81.53125, 854.332031], [57.683594, 854.332031], [22.578125, 867.582031], [22.578125, 880.167969], [53.046875, 890.765625], [53.046875, 920.574219], [20.589844, 931.832031], [21.253906, 942.429687], [52.386719, 957.003906], [76.890625, 958.328125], [77.554687, 987.472656], [86.828125, 987.472656], [97.425781, 1021.917969], [102.726563, 1021.253906], [112.0, 987.472656], [139.816406, 986.808594], [153.730469, 1021.253906], [171.613281, 986.148437], [171.613281, 951.707031], [210.027344, 952.367187], [210.691406, 941.769531], [224.601563, 942.429687], [225.265625, 896.0625], [237.851563, 896.0625], [236.523437, 960.316406], [200.757813, 959.652344], [179.5625, 972.238281], [178.898437, 991.445313], [193.472656, 1008.003906], [210.027344, 1008.003906], [221.953125, 1000.722656], [221.953125, 985.488281], [200.757813, 984.824219], [201.417969, 977.539063], [239.839844, 976.871094], [239.839844, 986.808594], [274.28125, 986.808594], [274.28125, 1017.28125], [302.101563, 1016.617187], [302.761719, 996.085937], [320.648437, 980.1875], [349.128906, 980.847656], [366.347656, 994.097656], [367.011719, 1016.617187], [390.859375, 1016.617187], [390.859375, 992.773437], [412.054687, 976.871094], [433.910156, 978.199219], [453.785156, 994.097656], [451.796875, 1016.617187], [480.277344, 1015.953125], [480.941406, 995.421875], [504.125, 977.539063], [527.308594, 977.539063], [541.882813, 990.785156], [541.214844, 1014.628906], [569.035156, 1013.96875], [570.363281, 991.445313], [584.933594, 976.210937], [614.078125, 976.210937], [630.640625, 990.785156], [630.640625, 1011.980469], [654.484375, 1012.640625], [655.804687, 990.785156], [675.679687, 974.890625], [700.851563, 974.890625], [720.71875, 990.785156], [721.386719, 1011.320313], [745.890625, 1012.640625], [745.890625, 990.125], [765.097656, 974.890625], [788.28125, 974.890625], [809.476563, 988.796875], [808.15625, 1011.320313], [835.308594, 1011.320313], [835.308594, 987.472656], [852.535156, 973.5625], [878.367187, 973.5625], [895.585937, 984.160156], [894.925781, 1011.980469], [923.40625, 1012.640625], [924.734375, 988.136719], [941.953125, 971.574219], [963.148437, 970.25], [975.074219, 981.511719], [975.074219, 1005.355469], [1020.777344, 1006.019531], [1022.761719, 980.847656], [1036.007813, 970.25], [1057.207031, 970.25], [1072.441406, 985.488281], [1073.105469, 1009.992187], [1094.960937, 1009.992187], [1095.625, 992.109375], [1109.535156, 970.25], [1185.707031, 970.914063], [1196.96875, 961.640625], [1216.179687, 961.640625], [1227.4375, 970.914063], [1227.4375, 985.488281], [1212.863281, 985.488281], [1212.203125, 1002.707031], [1169.148437, 1002.042969], [1146.628906, 1021.253906], [1147.289063, 1043.113281], [1157.226563, 1057.683594], [1183.058594, 1058.347656], [1195.644531, 1049.074219], [1195.644531, 1038.476563], [1175.109375, 1037.8125], [1175.109375, 1033.175781], [1211.542969, 1032.515625], [1212.203125, 1050.398437], [1222.140625, 1056.359375], [1220.8125, 1076.894531], [1194.980469, 1098.089844], [1182.398437, 1097.429687], [1183.058594, 1132.53125], [1208.890625, 1133.195313], [1222.140625, 1144.457031], [1222.800781, 1166.972656], [1200.277344, 1201.421875], [1184.382813, 1201.421875], [1184.382813, 1236.523437], [1210.214844, 1237.851563], [1240.023437, 1271.632813], [1239.359375, 1295.476563], [1273.140625, 1296.140625], [1273.804687, 1263.019531], [1303.609375, 1251.761719], [1328.121094, 1260.371094], [1328.121094, 1280.902344], [1365.210937, 1281.566406], [1365.875, 1262.359375], [1379.121094, 1252.421875], [1402.964844, 1252.421875], [1416.210937, 1263.683594], [1416.210937, 1281.566406], [1453.96875, 1282.230469], [1455.957031, 1268.316406], [1469.203125, 1253.75], [1489.738281, 1254.410156], [1516.234375, 1265.007813], [1515.570313, 1293.488281], [1550.675781, 1294.816406], [1551.335937, 1262.359375], [1571.871094, 1244.472656], [1606.980469, 1243.8125], [1614.925781, 1262.359375], [1630.824219, 1251.761719], [1615.589844, 1230.566406], [1622.210937, 1221.953125], [1646.058594, 1221.953125], [1646.058594, 1199.433594], [1622.210937, 1198.769531], [1621.550781, 1187.507813], [1632.148437, 1166.972656], [1616.910156, 1155.054687], [1601.679687, 1174.261719], [1563.921875, 1174.925781], [1563.261719, 1158.363281], [1577.171875, 1131.871094], [1595.054687, 1132.53125], [1595.054687, 1104.714844], [1581.144531, 1104.050781], [1567.234375, 1092.128906], [1566.574219, 1055.035156], [1603.664063, 1054.371094], [1615.589844, 1077.554687], [1635.460937, 1064.308594], [1622.875, 1044.433594], [1633.472656, 1033.175781], [1650.691406, 1033.175781], [1651.359375, 1009.332031], [1626.847656, 1008.667969], [1624.859375, 996.085937], [1634.796875, 976.210937], [1616.910156, 966.273437], [1603.664063, 990.125], [1567.898437, 989.457031], [1567.898437, 963.625], [1594.394531, 941.769531], [1595.054687, 924.546875], [1577.171875, 919.910156], [1567.234375, 894.742187], [1566.574219, 871.558594], [1590.417969, 871.558594], [1612.941406, 877.515625], [1612.277344, 903.351563], [1637.445313, 901.363281], [1650.03125, 866.917969], [1675.203125, 866.917969], [1674.542969, 906.664063], [1655.992187, 919.246094], [1665.929687, 937.792969], [1681.824219, 924.546875], [1691.761719, 933.160156], [1690.4375, 959.652344], [1709.644531, 959.652344], [1709.644531, 929.84375], [1717.59375, 919.910156], [1733.492187, 933.160156], [1740.777344, 915.273437], [1722.894531, 906.0], [1722.894531, 867.582031], [1750.714844, 867.582031], [1765.285156, 893.414063], [1781.183594, 886.792969], [1781.84375, 859.632813], [1794.429687, 844.402344], [1810.988281, 845.0625], [1828.210937, 876.855469], [1816.289063, 892.753906], [1826.222656, 902.027344], [1841.460937, 882.15625], [1855.367187, 883.480469], [1864.644531, 897.390625], [1880.539063, 884.144531], [1864.644531, 864.933594], [1868.617187, 857.648437], [1888.488281, 848.375], [1885.171875, 836.453125], [1862.65625, 840.425781], [1839.472656, 815.917969], [1840.132813, 802.671875], [1858.679687, 786.773437], [1881.203125, 792.734375], [1893.785156, 768.226563], [1883.851563, 763.589844], [1876.5625, 751.667969], [1877.226563, 721.863281], [1925.582031, 721.195313], [1925.582031, 743.058594], [1941.476563, 742.390625], [1949.425781, 722.523437], [1983.867187, 722.523437], [2001.753906, 702.652344], [2018.976563, 703.3125], [2019.265625, 682.171875], [2001.65625, 682.171875], [1983.441406, 656.0625], [1945.792969, 655.457031], [1938.507813, 637.84375], [1926.363281, 637.84375], [1926.011719, 656.242187], [1875.773437, 654.527344], [1876.34375, 634.546875], [1884.335937, 617.421875], [1895.183594, 610.566406], [1886.050781, 587.734375], [1851.796875, 588.875], [1844.945313, 570.605469], [1858.644531, 552.910156], [1881.480469, 552.910156], [1886.621094, 547.773437], [1886.050781, 534.636719], [1869.496094, 527.21875], [1864.925781, 516.3710940000001], [1882.625, 500.957031], [1876.34375, 489.539063], [1864.355469, 483.832031], [1863.785156, 492.964844], [1851.226563, 500.957031], [1840.378906, 491.253906], [1840.378906, 480.976563], [1831.816406, 481.546875], [1823.820313, 486.113281], [1822.679687, 519.796875], [1812.402344, 534.636719], [1800.414063, 534.636719], [1787.855469, 519.2226559999999], [1788.425781, 495.25], [1779.292969, 485.542969], [1761.59375, 485.542969], [1761.027344, 495.816406], [1751.886719, 510.664063], [1733.050781, 510.664063], [1724.484375, 500.386719], [1724.484375, 476.980469], [1744.46875, 467.273437], [1742.753906, 449.574219], [1737.617187, 443.296875], [1721.0625, 454.714844], [1712.5, 447.863281], [1713.640625, 425.027344], [1695.371094, 425.027344], [1694.800781, 449.007813], [1684.523437, 453.570313], [1669.683594, 439.871094], [1664.542969, 445.011719], [1661.117187, 459.851563], [1677.675781, 470.699219], [1677.675781, 496.390625], [1671.394531, 506.09375], [1653.695313, 505.523437], [1643.417969, 493.539063], [1643.417969, 473.554687], [1613.734375, 472.984375], [1613.164063, 494.105469], [1594.894531, 502.671875], [1575.480469, 502.097656], [1568.632813, 486.683594], [1578.339844, 460.425781], [1587.472656, 453.003906], [1601.746094, 454.144531], [1601.746094, 426.742187], [1586.332031, 427.3125], [1572.628906, 411.898437], [1573.203125, 389.632813], [1578.910156, 379.359375], [1604.597656, 379.925781], [1617.730469, 399.335937], [1638.855469, 387.347656], [1626.292969, 368.507813], [1635.425781, 357.09375], [1656.550781, 357.09375], [1657.121094, 333.683594], [1633.144531, 333.683594], [1628.003906, 321.128906], [1639.421875, 300.574219], [1626.292969, 289.726563], [1606.308594, 313.703125], [1581.191406, 314.847656], [1574.34375, 305.714844], [1574.34375, 287.441406], [1584.046875, 264.03515600000003], [1602.316406, 264.03515600000003], [1602.886719, 232.066406], [1584.046875, 232.636719], [1573.203125, 212.65625], [1572.628906, 192.675781], [1576.625, 185.824219], [1606.882813, 185.824219], [1621.726563, 209.230469], [1641.707031, 196.671875], [1629.148437, 172.125], [1639.996094, 157.277344], [1656.550781, 156.710937], [1656.550781, 139.582031], [1634.859375, 138.441406], [1629.148437, 132.160156], [1641.136719, 108.753906], [1626.863281, 100.191406], [1608.027344, 120.171875], [1582.078125, 120.226563], [1555.667969, 87.601563], [1555.667969, 61.972656], [1523.042969, 61.972656], [1522.65625, 80.609375], [1514.113281, 96.535156], [1490.035156, 104.691406], [1475.664063, 105.082031], [1467.507813, 97.699219], [1466.730469, 63.132813], [1413.519531, 63.523437], [1413.910156, 92.261719], [1406.140625, 106.632813], [1386.335937, 106.632813], [1375.460937, 98.089844], [1375.460937, 63.914063], [1326.136719, 63.914063], [1326.136719, 100.03125], [1319.144531, 106.632813], [1306.71875, 106.632813], [1284.96875, 100.03125], [1276.421875, 86.050781], [1276.421875, 66.632813], [1240.304687, 65.855469], [1240.695313, 92.261719], [1218.585937, 111.828125], [1218.585937, 126.875], [1205.3125, 130.855469], [1181.855469, 130.855469], [1181.855469, 167.148437], [1204.867187, 166.707031], [1223.011719, 187.503906], [1222.570313, 216.714844], [1204.867187, 232.644531], [1182.296875, 232.644531], [1181.855469, 266.72265600000003], [1205.3125, 266.28125], [1214.164063, 270.707031], [1225.667969, 288.851563], [1225.226563, 304.78125], [1217.703125, 314.078125], [1217.261719, 384.445313], [1225.226563, 384.445313], [1224.78125, 409.671875], [1206.636719, 409.671875], [1206.636719, 404.804687], [1194.246094, 404.804687], [1194.246094, 398.167969], [1180.527344, 398.609375], [1180.96875, 344.613281], [1188.933594, 344.613281], [1188.933594, 374.265625], [1206.195313, 361.875], [1206.636719, 338.859375], [1191.589844, 326.914063], [1180.96875, 326.914063], [1171.675781, 329.566406], [1161.050781, 344.613281], [1160.609375, 397.722656], [1148.21875, 397.722656], [1143.792969, 402.589844], [1111.816406, 402.910156], [1098.828125, 387.886719], [1098.984375, 357.847656], [1076.453125, 357.847656], [1076.296875, 387.730469], [1058.617187, 402.4375], [1028.730469, 402.4375], [1010.265625, 383.195313], [1010.265625, 358.003906], [987.894531, 357.847656], [987.734375, 382.878906], [970.050781, 402.28125], [944.550781, 402.4375], [925.929687, 382.414063], [925.929687, 357.847656]]
    
    brc997d = [[43.0, 131.0], [43.0, 132.0], [43.0, 133.0], [42.0, 133.0], [41.0, 133.0], [40.0, 133.0], [39.0, 133.0], [38.0, 133.0], [37.0, 133.0], [36.0, 133.0], [36.0, 134.0], [35.0, 134.0], [34.0, 134.0], [34.0, 133.0], [33.0, 133.0], [32.0, 133.0], [31.0, 133.0], [31.0, 134.0], [30.0, 134.0], [30.0, 135.0], [30.0, 136.0], [30.0, 137.0], [30.0, 138.0], [30.0, 139.0], [29.0, 139.0], [29.0, 140.0], [29.0, 141.0], [29.0, 142.0], [29.0, 143.0], [29.0, 144.0], [29.0, 145.0], [30.0, 145.0], [30.0, 146.0], [30.0, 147.0], [30.0, 148.0], [30.0, 149.0], [30.0, 150.0], [31.0, 150.0], [31.0, 151.0], [31.0, 152.0], [31.0, 153.0], [32.0, 153.0], [32.0, 154.0], [32.0, 155.0], [32.0, 156.0], [33.0, 156.0], [33.0, 157.0], [33.0, 158.0], [34.0, 158.0], [34.0, 159.0], [34.0, 160.0], [34.0, 161.0], [35.0, 161.0], [35.0, 162.0], [35.0, 163.0], [35.0, 164.0], [35.0, 165.0], [35.0, 166.0], [35.0, 167.0], [35.0, 168.0], [35.0, 169.0], [35.0, 170.0], [35.0, 171.0], [35.0, 172.0], [34.0, 172.0], [34.0, 173.0], [34.0, 174.0], [34.0, 175.0], [34.0, 176.0], [33.0, 176.0], [33.0, 177.0], [33.0, 178.0], [33.0, 179.0], [33.0, 180.0], [33.0, 181.0], [34.0, 181.0], [34.0, 182.0], [34.0, 183.0], [34.0, 184.0], [34.0, 185.0], [34.0, 186.0], [35.0, 186.0], [35.0, 187.0], [35.0, 188.0], [35.0, 189.0], [35.0, 190.0], [36.0, 190.0], [36.0, 191.0], [36.0, 192.0], [37.0, 192.0], [37.0, 193.0], [37.0, 194.0], [38.0, 194.0], [38.0, 195.0], [38.0, 196.0], [39.0, 196.0], [39.0, 197.0], [39.0, 198.0], [40.0, 198.0], [40.0, 199.0], [40.0, 200.0], [41.0, 200.0], [41.0, 201.0], [41.0, 202.0], [42.0, 202.0], [42.0, 203.0], [42.0, 204.0], [42.0, 205.0], [43.0, 205.0], [43.0, 206.0], [43.0, 207.0], [44.0, 207.0], [44.0, 208.0], [44.0, 209.0], [45.0, 209.0], [45.0, 210.0], [45.0, 211.0], [46.0, 211.0], [46.0, 212.0], [46.0, 213.0], [47.0, 213.0], [47.0, 214.0], [48.0, 214.0], [48.0, 215.0], [48.0, 216.0], [49.0, 216.0], [49.0, 217.0], [49.0, 218.0], [50.0, 218.0], [50.0, 219.0], [51.0, 219.0], [51.0, 220.0], [51.0, 221.0], [52.0, 221.0], [52.0, 222.0], [52.0, 223.0], [53.0, 223.0], [53.0, 222.0], [54.0, 222.0], [55.0, 222.0], [55.0, 223.0], [55.0, 224.0], [55.0, 225.0], [54.0, 225.0], [53.0, 225.0], [53.0, 226.0], [54.0, 226.0], [54.0, 227.0], [54.0, 228.0], [54.0, 229.0], [55.0, 229.0], [55.0, 230.0], [56.0, 230.0], [56.0, 231.0], [57.0, 231.0], [58.0, 231.0], [59.0, 231.0], [60.0, 231.0], [60.0, 230.0], [61.0, 230.0], [62.0, 230.0], [63.0, 230.0], [63.0, 229.0], [64.0, 229.0], [65.0, 229.0], [65.0, 228.0], [66.0, 228.0], [67.0, 228.0], [68.0, 228.0], [69.0, 228.0], [70.0, 228.0], [71.0, 228.0], [72.0, 228.0], [72.0, 227.0], [72.0, 226.0], [72.0, 225.0], [72.0, 224.0], [71.0, 224.0], [71.0, 223.0], [70.0, 223.0], [70.0, 222.0], [70.0, 221.0], [69.0, 221.0], [69.0, 220.0], [69.0, 219.0], [68.0, 219.0], [68.0, 218.0], [68.0, 217.0], [67.0, 217.0], [67.0, 216.0], [67.0, 215.0], [66.0, 215.0], [66.0, 214.0], [66.0, 213.0], [66.0, 212.0], [65.0, 212.0], [65.0, 211.0], [65.0, 210.0], [65.0, 209.0], [65.0, 208.0], [65.0, 207.0], [65.0, 206.0], [65.0, 205.0], [65.0, 204.0], [65.0, 203.0], [65.0, 202.0], [65.0, 201.0], [66.0, 201.0], [66.0, 200.0], [66.0, 199.0], [66.0, 198.0], [66.0, 197.0], [66.0, 196.0], [67.0, 196.0], [67.0, 195.0], [67.0, 194.0], [67.0, 193.0], [68.0, 193.0], [68.0, 192.0], [69.0, 192.0], [69.0, 191.0], [69.0, 190.0], [68.0, 190.0], [68.0, 189.0], [68.0, 188.0], [68.0, 187.0], [69.0, 187.0], [69.0, 186.0], [70.0, 186.0], [70.0, 185.0], [70.0, 184.0], [71.0, 184.0], [72.0, 184.0], [72.0, 183.0], [72.0, 182.0], [73.0, 182.0], [74.0, 182.0], [74.0, 181.0], [73.0, 181.0], [73.0, 180.0], [72.0, 180.0], [72.0, 179.0], [72.0, 178.0], [72.0, 177.0], [73.0, 177.0], [74.0, 177.0], [75.0, 177.0], [75.0, 178.0], [75.0, 179.0], [75.0, 180.0], [76.0, 180.0], [77.0, 180.0], [77.0, 181.0], [78.0, 181.0], [78.0, 182.0], [79.0, 182.0], [79.0, 181.0], [79.0, 180.0], [79.0, 179.0], [80.0, 179.0], [80.0, 178.0], [80.0, 177.0], [81.0, 177.0], [81.0, 176.0], [80.0, 176.0], [80.0, 175.0], [81.0, 175.0], [81.0, 174.0], [82.0, 174.0], [82.0, 173.0], [83.0, 173.0], [83.0, 172.0], [84.0, 172.0], [84.0, 173.0], [85.0, 173.0], [85.0, 172.0], [86.0, 172.0], [87.0, 172.0], [88.0, 172.0], [88.0, 173.0], [89.0, 173.0], [90.0, 173.0], [91.0, 173.0], [92.0, 173.0], [92.0, 174.0], [93.0, 174.0], [94.0, 174.0], [94.0, 175.0], [95.0, 175.0], [95.0, 176.0], [96.0, 176.0], [97.0, 176.0], [97.0, 177.0], [97.0, 178.0], [97.0, 179.0], [98.0, 179.0], [98.0, 180.0], [98.0, 181.0], [99.0, 181.0], [99.0, 180.0], [100.0, 180.0], [100.0, 181.0], [100.0, 182.0], [100.0, 183.0], [100.0, 184.0], [100.0, 185.0], [100.0, 186.0], [100.0, 187.0], [99.0, 187.0], [99.0, 188.0], [99.0, 189.0], [99.0, 190.0], [98.0, 190.0], [98.0, 191.0], [97.0, 191.0], [97.0, 192.0], [96.0, 192.0], [96.0, 193.0], [96.0, 194.0], [95.0, 194.0], [95.0, 195.0], [94.0, 195.0], [93.0, 195.0], [93.0, 196.0], [92.0, 196.0], [92.0, 197.0], [91.0, 197.0], [91.0, 198.0], [90.0, 198.0], [90.0, 199.0], [89.0, 199.0], [89.0, 200.0], [88.0, 200.0], [88.0, 201.0], [87.0, 201.0], [87.0, 202.0], [86.0, 202.0], [86.0, 203.0], [86.0, 204.0], [86.0, 205.0], [86.0, 206.0], [85.0, 206.0], [85.0, 205.0], [85.0, 204.0], [84.0, 204.0], [84.0, 205.0], [84.0, 206.0], [84.0, 207.0], [84.0, 208.0], [85.0, 208.0], [85.0, 209.0], [85.0, 210.0], [85.0, 211.0], [86.0, 211.0], [86.0, 212.0], [86.0, 213.0], [87.0, 213.0], [87.0, 214.0], [88.0, 214.0], [88.0, 215.0], [89.0, 215.0], [90.0, 215.0], [91.0, 215.0], [92.0, 215.0], [93.0, 215.0], [93.0, 216.0], [94.0, 216.0], [94.0, 215.0], [94.0, 214.0], [95.0, 214.0], [96.0, 214.0], [96.0, 215.0], [97.0, 215.0], [97.0, 216.0], [98.0, 216.0], [99.0, 216.0], [100.0, 216.0], [100.0, 217.0], [101.0, 217.0], [101.0, 218.0], [102.0, 218.0], [103.0, 218.0], [103.0, 219.0], [103.0, 220.0], [103.0, 221.0], [104.0, 221.0], [105.0, 221.0], [106.0, 221.0], [107.0, 221.0], [107.0, 220.0], [108.0, 220.0], [108.0, 219.0], [109.0, 219.0], [109.0, 218.0], [109.0, 217.0], [110.0, 217.0], [110.0, 216.0], [111.0, 216.0], [112.0, 216.0], [112.0, 215.0], [113.0, 215.0], [114.0, 215.0], [114.0, 214.0], [115.0, 214.0], [116.0, 214.0], [116.0, 213.0], [117.0, 213.0], [118.0, 213.0], [118.0, 212.0], [119.0, 212.0], [120.0, 212.0], [121.0, 212.0], [122.0, 212.0], [122.0, 211.0], [123.0, 211.0], [124.0, 211.0], [125.0, 211.0], [126.0, 211.0], [127.0, 211.0], [128.0, 211.0], [128.0, 212.0], [129.0, 212.0], [130.0, 212.0], [130.0, 213.0], [130.0, 214.0], [131.0, 214.0], [131.0, 215.0], [132.0, 215.0], [132.0, 216.0], [131.0, 216.0], [131.0, 217.0], [132.0, 217.0], [132.0, 218.0], [132.0, 219.0], [132.0, 220.0], [133.0, 220.0], [133.0, 221.0], [133.0, 222.0], [133.0, 223.0], [133.0, 224.0], [133.0, 225.0], [133.0, 226.0], [133.0, 227.0], [132.0, 227.0], [132.0, 228.0], [131.0, 228.0], [131.0, 229.0], [131.0, 230.0], [130.0, 230.0], [130.0, 231.0], [129.0, 231.0], [129.0, 232.0], [129.0, 233.0], [128.0, 233.0], [128.0, 234.0], [127.0, 234.0], [126.0, 234.0], [126.0, 235.0], [125.0, 235.0], [124.0, 235.0], [124.0, 236.0], [123.0, 236.0], [122.0, 236.0], [122.0, 237.0], [121.0, 237.0], [120.0, 237.0], [119.0, 237.0], [118.0, 237.0], [117.0, 237.0], [116.0, 237.0], [115.0, 237.0], [114.0, 237.0], [113.0, 237.0], [112.0, 237.0], [111.0, 237.0], [111.0, 236.0], [110.0, 236.0], [109.0, 236.0], [108.0, 236.0], [107.0, 236.0], [106.0, 236.0], [105.0, 236.0], [105.0, 237.0], [105.0, 238.0], [105.0, 239.0], [105.0, 240.0], [105.0, 241.0], [105.0, 242.0], [106.0, 242.0], [106.0, 243.0], [106.0, 244.0], [106.0, 245.0], [106.0, 246.0], [106.0, 247.0], [106.0, 248.0], [106.0, 249.0], [107.0, 249.0], [107.0, 250.0], [108.0, 250.0], [109.0, 250.0], [109.0, 249.0], [110.0, 249.0], [111.0, 249.0], [112.0, 249.0], [113.0, 249.0], [114.0, 249.0], [115.0, 249.0], [116.0, 249.0], [116.0, 248.0], [117.0, 248.0], [118.0, 248.0], [119.0, 248.0], [119.0, 249.0], [120.0, 249.0], [120.0, 248.0], [121.0, 248.0], [122.0, 248.0], [123.0, 248.0], [124.0, 248.0], [124.0, 249.0], [124.0, 250.0], [124.0, 251.0], [124.0, 252.0], [123.0, 252.0], [122.0, 252.0], [121.0, 252.0], [121.0, 253.0], [122.0, 253.0], [123.0, 253.0], [124.0, 253.0], [125.0, 253.0], [126.0, 253.0], [127.0, 253.0], [127.0, 252.0], [128.0, 252.0], [129.0, 252.0], [130.0, 252.0], [131.0, 252.0], [132.0, 252.0], [133.0, 252.0], [134.0, 252.0], [134.0, 251.0], [135.0, 251.0], [135.0, 250.0], [136.0, 250.0], [136.0, 249.0], [137.0, 249.0], [137.0, 248.0], [138.0, 248.0], [139.0, 248.0], [140.0, 248.0], [141.0, 248.0], [142.0, 248.0], [143.0, 248.0], [144.0, 248.0], [145.0, 248.0], [146.0, 248.0], [147.0, 248.0], [148.0, 248.0], [149.0, 248.0], [150.0, 248.0], [150.0, 247.0], [150.0, 246.0], [149.0, 246.0], [149.0, 245.0], [148.0, 245.0], [148.0, 244.0], [147.0, 244.0], [147.0, 243.0], [147.0, 242.0], [147.0, 241.0], [147.0, 240.0], [148.0, 240.0], [149.0, 240.0], [149.0, 239.0], [149.0, 238.0], [149.0, 237.0], [149.0, 236.0], [149.0, 235.0], [149.0, 234.0], [149.0, 233.0], [149.0, 232.0], [149.0, 231.0], [149.0, 230.0], [149.0, 229.0], [149.0, 228.0], [149.0, 227.0], [150.0, 227.0], [151.0, 227.0], [151.0, 226.0], [151.0, 225.0], [151.0, 224.0], [152.0, 224.0], [153.0, 224.0], [154.0, 224.0], [154.0, 225.0], [154.0, 226.0], [154.0, 227.0], [155.0, 227.0], [156.0, 227.0], [156.0, 228.0], [156.0, 229.0], [156.0, 230.0], [156.0, 231.0], [156.0, 232.0], [156.0, 233.0], [156.0, 234.0], [156.0, 235.0], [156.0, 236.0], [156.0, 237.0], [156.0, 238.0], [157.0, 238.0], [157.0, 239.0], [158.0, 239.0], [159.0, 239.0], [160.0, 239.0], [161.0, 239.0], [162.0, 239.0], [163.0, 239.0], [163.0, 238.0], [163.0, 237.0], [163.0, 236.0], [163.0, 235.0], [162.0, 235.0], [162.0, 234.0], [162.0, 233.0], [162.0, 232.0], [162.0, 231.0], [162.0, 230.0], [163.0, 230.0], [163.0, 229.0], [164.0, 229.0], [165.0, 229.0], [165.0, 228.0], [165.0, 227.0], [165.0, 226.0], [164.0, 226.0], [164.0, 225.0], [164.0, 224.0], [164.0, 223.0], [163.0, 223.0], [162.0, 223.0], [161.0, 223.0], [160.0, 223.0], [160.0, 222.0], [159.0, 222.0], [158.0, 222.0], [158.0, 221.0], [158.0, 220.0], [159.0, 220.0], [159.0, 219.0], [159.0, 218.0], [159.0, 217.0], [159.0, 216.0], [159.0, 215.0], [159.0, 214.0], [159.0, 213.0], [159.0, 212.0], [160.0, 212.0], [160.0, 211.0], [160.0, 210.0], [160.0, 209.0], [160.0, 208.0], [160.0, 207.0], [160.0, 206.0], [160.0, 205.0], [160.0, 204.0], [160.0, 203.0], [161.0, 203.0], [161.0, 202.0], [161.0, 201.0], [162.0, 201.0], [162.0, 202.0], [163.0, 202.0], [164.0, 202.0], [165.0, 202.0], [166.0, 202.0], [167.0, 202.0], [168.0, 202.0], [168.0, 201.0], [168.0, 200.0], [167.0, 200.0], [167.0, 199.0], [166.0, 199.0], [166.0, 198.0], [165.0, 198.0], [165.0, 197.0], [164.0, 197.0], [164.0, 196.0], [163.0, 196.0], [162.0, 196.0], [162.0, 195.0], [161.0, 195.0], [161.0, 194.0], [160.0, 194.0], [159.0, 194.0], [159.0, 193.0], [158.0, 193.0], [158.0, 192.0], [157.0, 192.0], [156.0, 192.0], [156.0, 191.0], [155.0, 191.0], [154.0, 191.0], [154.0, 190.0], [153.0, 190.0], [153.0, 189.0], [152.0, 189.0], [151.0, 189.0], [151.0, 188.0], [150.0, 188.0], [149.0, 188.0], [149.0, 187.0], [148.0, 187.0], [148.0, 186.0], [147.0, 186.0], [147.0, 185.0], [147.0, 184.0], [147.0, 183.0], [146.0, 183.0], [146.0, 182.0], [146.0, 181.0], [146.0, 180.0], [145.0, 180.0], [145.0, 179.0], [145.0, 178.0], [144.0, 178.0], [144.0, 177.0], [144.0, 176.0], [143.0, 176.0], [143.0, 175.0], [143.0, 174.0], [143.0, 173.0], [143.0, 172.0], [143.0, 171.0], [143.0, 170.0], [142.0, 170.0], [142.0, 169.0], [142.0, 168.0], [142.0, 167.0], [142.0, 166.0], [142.0, 165.0], [142.0, 164.0], [141.0, 164.0], [141.0, 163.0], [141.0, 162.0], [141.0, 161.0], [141.0, 160.0], [141.0, 159.0], [141.0, 158.0], [141.0, 157.0], [141.0, 156.0], [141.0, 155.0], [140.0, 155.0], [139.0, 155.0], [139.0, 154.0], [140.0, 154.0], [140.0, 153.0], [141.0, 153.0], [141.0, 152.0], [141.0, 151.0], [141.0, 150.0], [141.0, 149.0], [141.0, 148.0], [141.0, 147.0], [141.0, 146.0], [141.0, 145.0], [140.0, 145.0], [140.0, 144.0], [140.0, 143.0], [141.0, 143.0], [141.0, 142.0], [140.0, 142.0], [139.0, 142.0], [138.0, 142.0], [137.0, 142.0], [137.0, 141.0], [136.0, 141.0], [135.0, 141.0], [135.0, 140.0], [134.0, 140.0], [134.0, 139.0], [133.0, 139.0], [133.0, 138.0], [133.0, 137.0], [133.0, 136.0], [133.0, 135.0], [133.0, 134.0], [133.0, 133.0], [133.0, 132.0], [134.0, 132.0], [134.0, 131.0], [135.0, 131.0], [136.0, 131.0], [136.0, 132.0], [137.0, 132.0], [138.0, 132.0], [138.0, 131.0], [139.0, 131.0], [139.0, 130.0], [139.0, 129.0], [140.0, 129.0], [140.0, 128.0], [141.0, 128.0], [141.0, 127.0], [142.0, 127.0], [142.0, 126.0], [143.0, 126.0], [143.0, 125.0], [144.0, 125.0], [144.0, 124.0], [145.0, 124.0], [145.0, 123.0], [146.0, 123.0], [146.0, 124.0], [147.0, 124.0], [148.0, 124.0], [148.0, 123.0], [149.0, 123.0], [150.0, 123.0], [151.0, 123.0], [151.0, 124.0], [151.0, 125.0], [150.0, 125.0], [150.0, 126.0], [150.0, 127.0], [150.0, 128.0], [150.0, 129.0], [149.0, 129.0], [149.0, 130.0], [149.0, 131.0], [148.0, 131.0], [148.0, 132.0], [147.0, 132.0], [147.0, 133.0], [146.0, 133.0], [146.0, 134.0], [145.0, 134.0], [145.0, 135.0], [144.0, 135.0], [144.0, 136.0], [143.0, 136.0], [143.0, 137.0], [143.0, 138.0], [144.0, 138.0], [145.0, 138.0], [146.0, 138.0], [147.0, 138.0], [147.0, 139.0], [148.0, 139.0], [148.0, 138.0], [149.0, 138.0], [150.0, 138.0], [150.0, 137.0], [151.0, 137.0], [151.0, 136.0], [152.0, 136.0], [153.0, 136.0], [153.0, 135.0], [154.0, 135.0], [155.0, 135.0], [155.0, 134.0], [156.0, 134.0], [157.0, 134.0], [157.0, 133.0], [158.0, 133.0], [158.0, 132.0], [159.0, 132.0], [160.0, 132.0], [161.0, 132.0], [162.0, 132.0], [162.0, 131.0], [162.0, 130.0], [162.0, 129.0], [161.0, 129.0], [161.0, 128.0], [161.0, 127.0], [162.0, 127.0], [162.0, 126.0], [163.0, 126.0], [164.0, 126.0], [165.0, 126.0], [166.0, 126.0], [166.0, 125.0], [166.0, 124.0], [166.0, 123.0], [167.0, 123.0], [167.0, 122.0], [168.0, 122.0], [169.0, 122.0], [169.0, 123.0], [170.0, 123.0], [170.0, 124.0], [171.0, 124.0], [171.0, 125.0], [171.0, 126.0], [171.0, 127.0], [172.0, 127.0], [172.0, 128.0], [172.0, 129.0], [172.0, 130.0], [173.0, 130.0], [174.0, 130.0], [175.0, 130.0], [176.0, 130.0], [177.0, 130.0], [178.0, 130.0], [179.0, 130.0], [179.0, 131.0], [180.0, 131.0], [181.0, 131.0], [182.0, 131.0], [183.0, 131.0], [184.0, 131.0], [185.0, 131.0], [185.0, 132.0], [186.0, 132.0], [187.0, 132.0], [188.0, 132.0], [189.0, 132.0], [190.0, 132.0], [191.0, 132.0], [191.0, 133.0], [192.0, 133.0], [193.0, 133.0], [193.0, 134.0], [194.0, 134.0], [195.0, 134.0], [196.0, 134.0], [196.0, 135.0], [197.0, 135.0], [198.0, 135.0], [198.0, 136.0], [199.0, 136.0], [200.0, 136.0], [200.0, 137.0], [201.0, 137.0], [202.0, 137.0], [203.0, 137.0], [203.0, 136.0], [204.0, 136.0], [204.0, 135.0], [204.0, 134.0], [205.0, 134.0], [205.0, 133.0], [206.0, 133.0], [206.0, 132.0], [207.0, 132.0], [207.0, 131.0], [207.0, 130.0], [206.0, 130.0], [206.0, 129.0], [206.0, 128.0], [206.0, 127.0], [205.0, 127.0], [205.0, 126.0], [205.0, 125.0], [205.0, 124.0], [204.0, 124.0], [204.0, 123.0], [204.0, 122.0], [203.0, 122.0], [202.0, 122.0], [201.0, 122.0], [201.0, 121.0], [200.0, 121.0], [199.0, 121.0], [199.0, 120.0], [198.0, 120.0], [197.0, 120.0], [197.0, 121.0], [196.0, 121.0], [196.0, 122.0], [195.0, 122.0], [194.0, 122.0], [194.0, 123.0], [193.0, 123.0], [192.0, 123.0], [192.0, 124.0], [191.0, 124.0], [191.0, 125.0], [191.0, 126.0], [190.0, 126.0], [189.0, 126.0], [189.0, 127.0], [188.0, 127.0], [187.0, 127.0], [186.0, 127.0], [185.0, 127.0], [184.0, 127.0], [184.0, 126.0], [184.0, 125.0], [185.0, 125.0], [185.0, 124.0], [184.0, 124.0], [183.0, 124.0], [182.0, 124.0], [181.0, 124.0], [181.0, 123.0], [180.0, 123.0], [179.0, 123.0], [178.0, 123.0], [177.0, 123.0], [176.0, 123.0], [175.0, 123.0], [175.0, 122.0], [174.0, 122.0], [173.0, 122.0], [172.0, 122.0], [172.0, 121.0], [172.0, 120.0], [171.0, 120.0], [170.0, 120.0], [170.0, 119.0], [170.0, 118.0], [170.0, 117.0], [170.0, 116.0], [171.0, 116.0], [172.0, 116.0], [172.0, 117.0], [173.0, 117.0], [173.0, 116.0], [173.0, 115.0], [174.0, 115.0], [175.0, 115.0], [176.0, 115.0], [177.0, 115.0], [177.0, 116.0], [178.0, 116.0], [179.0, 116.0], [180.0, 116.0], [181.0, 116.0], [182.0, 116.0], [183.0, 116.0], [183.0, 117.0], [184.0, 117.0], [185.0, 117.0], [186.0, 117.0], [186.0, 116.0], [186.0, 115.0], [186.0, 114.0], [187.0, 114.0], [188.0, 114.0], [189.0, 114.0], [190.0, 114.0], [190.0, 113.0], [191.0, 113.0], [192.0, 113.0], [192.0, 114.0], [191.0, 114.0], [191.0, 115.0], [191.0, 116.0], [190.0, 116.0], [190.0, 117.0], [191.0, 117.0], [191.0, 118.0], [192.0, 118.0], [192.0, 119.0], [193.0, 119.0], [193.0, 120.0], [194.0, 120.0], [195.0, 120.0], [195.0, 119.0], [196.0, 119.0], [196.0, 118.0], [196.0, 117.0], [197.0, 117.0], [197.0, 116.0], [198.0, 116.0], [198.0, 115.0], [199.0, 115.0], [199.0, 114.0], [200.0, 114.0], [200.0, 113.0], [200.0, 112.0], [201.0, 112.0], [201.0, 111.0], [202.0, 111.0], [202.0, 110.0], [201.0, 110.0], [201.0, 109.0], [201.0, 108.0], [201.0, 107.0], [200.0, 107.0], [199.0, 107.0], [199.0, 108.0], [198.0, 108.0], [197.0, 108.0], [196.0, 108.0], [195.0, 108.0], [194.0, 108.0], [194.0, 107.0], [193.0, 107.0], [193.0, 106.0], [193.0, 105.0], [192.0, 105.0], [192.0, 104.0], [191.0, 104.0], [191.0, 103.0], [190.0, 103.0], [190.0, 102.0], [189.0, 102.0], [189.0, 101.0], [189.0, 100.0], [188.0, 100.0], [188.0, 99.0], [187.0, 99.0], [187.0, 98.0], [186.0, 98.0], [186.0, 97.0], [186.0, 96.0], [186.0, 95.0], [186.0, 94.0], [186.0, 93.0], [186.0, 92.0], [186.0, 91.0], [186.0, 90.0], [186.0, 89.0], [186.0, 88.0], [187.0, 88.0], [187.0, 87.0], [187.0, 86.0], [187.0, 85.0], [188.0, 85.0], [188.0, 84.0], [188.0, 83.0], [188.0, 82.0], [188.0, 81.0], [189.0, 81.0], [189.0, 80.0], [190.0, 80.0], [190.0, 79.0], [190.0, 78.0], [191.0, 78.0], [191.0, 77.0], [192.0, 77.0], [192.0, 76.0], [193.0, 76.0], [193.0, 75.0], [193.0, 74.0], [194.0, 74.0], [194.0, 73.0], [195.0, 73.0], [195.0, 72.0], [196.0, 72.0], [196.0, 71.0], [196.0, 70.0], [196.0, 69.0], [197.0, 69.0], [197.0, 68.0], [197.0, 67.0], [197.0, 66.0], [196.0, 66.0], [196.0, 65.0], [195.0, 65.0], [195.0, 64.0], [194.0, 64.0], [193.0, 64.0], [193.0, 63.0], [192.0, 63.0], [192.0, 62.0], [191.0, 62.0], [190.0, 62.0], [189.0, 62.0], [188.0, 62.0], [187.0, 62.0], [186.0, 62.0], [186.0, 63.0], [186.0, 64.0], [185.0, 64.0], [185.0, 65.0], [184.0, 65.0], [183.0, 65.0], [183.0, 64.0], [182.0, 64.0], [181.0, 64.0], [181.0, 65.0], [181.0, 66.0], [180.0, 66.0], [180.0, 67.0], [179.0, 67.0], [179.0, 68.0], [178.0, 68.0], [177.0, 68.0], [177.0, 69.0], [176.0, 69.0], [175.0, 69.0], [175.0, 70.0], [174.0, 70.0], [173.0, 70.0], [173.0, 71.0], [172.0, 71.0], [171.0, 71.0], [171.0, 72.0], [171.0, 73.0], [171.0, 74.0], [171.0, 75.0], [171.0, 76.0], [171.0, 77.0], [171.0, 78.0], [170.0, 78.0], [170.0, 79.0], [169.0, 79.0], [169.0, 80.0], [168.0, 80.0], [168.0, 81.0], [167.0, 81.0], [167.0, 82.0], [166.0, 82.0], [166.0, 83.0], [167.0, 83.0], [167.0, 84.0], [167.0, 85.0], [166.0, 85.0], [165.0, 85.0], [164.0, 85.0], [164.0, 84.0], [163.0, 84.0], [162.0, 84.0], [161.0, 84.0], [160.0, 84.0], [159.0, 84.0], [158.0, 84.0], [158.0, 83.0], [157.0, 83.0], [157.0, 84.0], [157.0, 85.0], [156.0, 85.0], [155.0, 85.0], [155.0, 84.0], [155.0, 83.0], [156.0, 83.0], [156.0, 82.0], [155.0, 82.0], [154.0, 82.0], [154.0, 81.0], [154.0, 80.0], [153.0, 80.0], [153.0, 79.0], [153.0, 78.0], [153.0, 77.0], [153.0, 76.0], [153.0, 75.0], [153.0, 74.0], [153.0, 73.0], [154.0, 73.0], [155.0, 73.0], [155.0, 72.0], [155.0, 71.0], [156.0, 71.0], [156.0, 70.0], [156.0, 69.0], [156.0, 68.0], [156.0, 67.0], [156.0, 66.0], [155.0, 66.0], [155.0, 65.0], [154.0, 65.0], [154.0, 64.0], [154.0, 63.0], [155.0, 63.0], [155.0, 62.0], [156.0, 62.0], [156.0, 61.0], [157.0, 61.0], [157.0, 60.0], [157.0, 59.0], [157.0, 58.0], [157.0, 57.0], [157.0, 56.0], [157.0, 55.0], [156.0, 55.0], [155.0, 55.0], [155.0, 54.0], [155.0, 53.0], [155.0, 52.0], [154.0, 52.0], [154.0, 51.0], [154.0, 50.0], [153.0, 50.0], [153.0, 49.0], [152.0, 49.0], [152.0, 48.0], [151.0, 48.0], [150.0, 48.0], [149.0, 48.0], [148.0, 48.0], [148.0, 47.0], [147.0, 47.0], [147.0, 46.0], [147.0, 45.0], [147.0, 44.0], [146.0, 44.0], [146.0, 43.0], [145.0, 43.0], [144.0, 43.0], [144.0, 42.0], [143.0, 42.0], [143.0, 41.0], [142.0, 41.0], [141.0, 41.0], [140.0, 41.0], [140.0, 40.0], [139.0, 40.0], [138.0, 40.0], [138.0, 39.0], [137.0, 39.0], [137.0, 38.0], [137.0, 37.0], [136.0, 37.0], [136.0, 36.0], [136.0, 35.0], [135.0, 35.0], [135.0, 34.0], [135.0, 33.0], [135.0, 32.0], [134.0, 32.0], [134.0, 31.0], [134.0, 30.0], [134.0, 29.0], [134.0, 28.0], [134.0, 27.0], [134.0, 26.0], [134.0, 25.0], [134.0, 24.0], [134.0, 23.0], [134.0, 22.0], [133.0, 22.0], [133.0, 21.0], [133.0, 20.0], [133.0, 19.0], [133.0, 18.0], [133.0, 17.0], [132.0, 17.0], [131.0, 17.0], [130.0, 17.0], [130.0, 16.0], [129.0, 16.0], [128.0, 16.0], [127.0, 16.0], [127.0, 15.0], [126.0, 15.0], [126.0, 14.0], [125.0, 14.0], [125.0, 13.0], [124.0, 13.0], [124.0, 12.0], [123.0, 12.0], [123.0, 11.0], [122.0, 11.0], [122.0, 10.0], [121.0, 10.0], [121.0, 9.0], [120.0, 9.0], [120.0, 8.0], [119.0, 8.0], [119.0, 7.0], [118.0, 7.0], [118.0, 6.0], [117.0, 6.0], [117.0, 5.0], [116.0, 5.0], [115.0, 5.0], [115.0, 4.0], [114.0, 4.0], [114.0, 3.0], [113.0, 3.0], [112.0, 3.0], [112.0, 2.0], [111.0, 2.0], [110.0, 2.0], [109.0, 2.0], [108.0, 2.0], [107.0, 2.0], [107.0, 1.0], [106.0, 1.0], [105.0, 1.0], [104.0, 1.0], [104.0, 2.0], [103.0, 2.0], [102.0, 2.0], [101.0, 2.0], [100.0, 2.0], [100.0, 3.0], [99.0, 3.0], [98.0, 3.0], [97.0, 3.0], [96.0, 3.0], [95.0, 3.0], [94.0, 3.0], [93.0, 3.0], [93.0, 2.0], [92.0, 2.0], [91.0, 2.0], [90.0, 2.0], [89.0, 2.0], [88.0, 2.0], [87.0, 2.0], [87.0, 3.0], [86.0, 3.0], [85.0, 3.0], [85.0, 4.0], [84.0, 4.0], [84.0, 5.0], [84.0, 6.0], [83.0, 6.0], [83.0, 7.0], [82.0, 7.0], [82.0, 8.0], [81.0, 8.0], [80.0, 8.0], [79.0, 8.0], [78.0, 8.0], [77.0, 8.0], [76.0, 8.0], [75.0, 8.0], [75.0, 9.0], [74.0, 9.0], [73.0, 9.0], [73.0, 10.0], [72.0, 10.0], [72.0, 11.0], [71.0, 11.0], [71.0, 12.0], [71.0, 13.0], [70.0, 13.0], [69.0, 13.0], [68.0, 13.0], [67.0, 13.0], [67.0, 14.0], [66.0, 14.0], [66.0, 15.0], [65.0, 15.0], [65.0, 16.0], [65.0, 17.0], [64.0, 17.0], [64.0, 18.0], [63.0, 18.0], [63.0, 19.0], [63.0, 20.0], [62.0, 20.0], [62.0, 21.0], [62.0, 22.0], [61.0, 22.0], [61.0, 23.0], [60.0, 23.0], [60.0, 24.0], [60.0, 25.0], [59.0, 25.0], [59.0, 26.0], [59.0, 27.0], [60.0, 27.0], [60.0, 28.0], [61.0, 28.0], [61.0, 29.0], [62.0, 29.0], [63.0, 29.0], [63.0, 30.0], [64.0, 30.0], [64.0, 31.0], [65.0, 31.0], [65.0, 32.0], [66.0, 32.0], [66.0, 33.0], [66.0, 34.0], [67.0, 34.0], [67.0, 35.0], [67.0, 36.0], [68.0, 36.0], [68.0, 37.0], [69.0, 37.0], [69.0, 38.0], [69.0, 39.0], [69.0, 40.0], [70.0, 40.0], [70.0, 41.0], [69.0, 41.0], [69.0, 42.0], [69.0, 43.0], [69.0, 44.0], [69.0, 45.0], [69.0, 46.0], [69.0, 47.0], [68.0, 47.0], [68.0, 48.0], [68.0, 49.0], [69.0, 49.0], [70.0, 49.0], [70.0, 50.0], [70.0, 51.0], [69.0, 51.0], [68.0, 51.0], [68.0, 52.0], [69.0, 52.0], [69.0, 53.0], [69.0, 54.0], [68.0, 54.0], [68.0, 55.0], [67.0, 55.0], [67.0, 56.0], [66.0, 56.0], [65.0, 56.0], [64.0, 56.0], [64.0, 57.0], [63.0, 57.0], [63.0, 58.0], [62.0, 58.0], [61.0, 58.0], [60.0, 58.0], [60.0, 59.0], [59.0, 59.0], [58.0, 59.0], [58.0, 58.0], [57.0, 58.0], [57.0, 59.0], [56.0, 59.0], [55.0, 59.0], [55.0, 60.0], [56.0, 60.0], [56.0, 61.0], [56.0, 62.0], [56.0, 63.0], [57.0, 63.0], [57.0, 64.0], [57.0, 65.0], [57.0, 66.0], [57.0, 67.0], [57.0, 68.0], [57.0, 69.0], [57.0, 70.0], [57.0, 71.0], [57.0, 72.0], [57.0, 73.0], [57.0, 74.0], [57.0, 75.0], [57.0, 76.0], [57.0, 77.0], [58.0, 77.0], [58.0, 78.0], [59.0, 78.0], [59.0, 79.0], [60.0, 79.0], [60.0, 80.0], [61.0, 80.0], [61.0, 81.0], [62.0, 81.0], [62.0, 82.0], [63.0, 82.0], [63.0, 83.0], [64.0, 83.0], [65.0, 83.0], [66.0, 83.0], [67.0, 83.0], [67.0, 84.0], [68.0, 84.0], [69.0, 84.0], [70.0, 84.0], [71.0, 84.0], [71.0, 85.0], [72.0, 85.0], [72.0, 86.0], [72.0, 87.0], [72.0, 88.0], [72.0, 89.0], [72.0, 90.0], [72.0, 91.0], [72.0, 92.0], [73.0, 92.0], [73.0, 93.0], [73.0, 94.0], [73.0, 95.0], [72.0, 95.0], [72.0, 96.0], [72.0, 97.0], [72.0, 98.0], [72.0, 99.0], [72.0, 100.0], [72.0, 101.0], [72.0, 102.0], [72.0, 103.0], [72.0, 104.0], [72.0, 105.0], [73.0, 105.0], [73.0, 106.0], [74.0, 106.0], [75.0, 106.0], [75.0, 107.0], [76.0, 107.0], [77.0, 107.0], [77.0, 108.0], [78.0, 108.0], [79.0, 108.0], [79.0, 109.0], [80.0, 109.0], [80.0, 110.0], [80.0, 111.0], [80.0, 112.0], [80.0, 113.0], [79.0, 113.0], [79.0, 114.0], [78.0, 114.0], [78.0, 115.0], [77.0, 115.0], [77.0, 116.0], [76.0, 116.0], [76.0, 117.0], [77.0, 117.0], [77.0, 118.0], [77.0, 119.0], [77.0, 120.0], [76.0, 120.0], [75.0, 120.0], [74.0, 120.0], [74.0, 121.0], [73.0, 121.0], [73.0, 122.0], [72.0, 122.0], [72.0, 123.0], [72.0, 124.0], [73.0, 124.0], [74.0, 124.0], [75.0, 124.0], [75.0, 123.0], [76.0, 123.0], [77.0, 123.0], [78.0, 123.0], [78.0, 124.0], [79.0, 124.0], [79.0, 125.0], [78.0, 125.0], [78.0, 126.0], [77.0, 126.0], [77.0, 127.0], [76.0, 127.0], [76.0, 128.0], [76.0, 129.0], [77.0, 129.0], [78.0, 129.0], [78.0, 130.0], [79.0, 130.0], [80.0, 130.0], [80.0, 131.0], [81.0, 131.0], [81.0, 132.0], [81.0, 133.0], [81.0, 134.0], [81.0, 135.0], [80.0, 135.0], [79.0, 135.0], [78.0, 135.0], [78.0, 134.0], [77.0, 134.0], [76.0, 134.0], [75.0, 134.0], [74.0, 134.0], [73.0, 134.0], [72.0, 134.0], [71.0, 134.0], [71.0, 135.0], [72.0, 135.0], [72.0, 136.0], [73.0, 136.0], [73.0, 137.0], [73.0, 138.0], [73.0, 139.0], [72.0, 139.0], [72.0, 140.0], [72.0, 141.0], [72.0, 142.0], [72.0, 143.0], [71.0, 143.0], [71.0, 144.0], [70.0, 144.0], [69.0, 144.0], [69.0, 145.0], [68.0, 145.0], [67.0, 145.0], [66.0, 145.0], [66.0, 146.0], [65.0, 146.0], [64.0, 146.0], [63.0, 146.0], [63.0, 145.0], [63.0, 144.0], [62.0, 144.0], [62.0, 143.0], [61.0, 143.0], [61.0, 142.0], [61.0, 141.0], [60.0, 141.0], [60.0, 140.0], [59.0, 140.0], [58.0, 140.0], [58.0, 141.0], [57.0, 141.0], [56.0, 141.0], [55.0, 141.0], [55.0, 140.0], [54.0, 140.0], [54.0, 139.0], [54.0, 138.0], [53.0, 138.0], [53.0, 137.0], [52.0, 137.0], [51.0, 137.0], [51.0, 138.0], [50.0, 138.0], [50.0, 139.0], [49.0, 139.0], [49.0, 140.0], [48.0, 140.0], [48.0, 139.0], [47.0, 139.0], [47.0, 138.0], [47.0, 137.0], [47.0, 136.0], [47.0, 135.0], [47.0, 134.0], [47.0, 133.0], [46.0, 133.0], [45.0, 133.0], [44.0, 133.0], [44.0, 132.0], [44.0, 131.0], [45.0, 131.0], [45.0, 130.0], [46.0, 130.0], [47.0, 130.0], [48.0, 130.0], [49.0, 130.0], [49.0, 131.0], [50.0, 131.0], [50.0, 130.0], [51.0, 130.0], [51.0, 129.0], [51.0, 128.0], [51.0, 127.0], [50.0, 127.0], [49.0, 127.0], [49.0, 126.0], [48.0, 126.0], [47.0, 126.0], [47.0, 127.0], [46.0, 127.0], [45.0, 127.0], [45.0, 128.0], [44.0, 128.0], [43.0, 128.0], [43.0, 129.0], [42.0, 129.0], [41.0, 129.0], [40.0, 129.0], [40.0, 130.0], [39.0, 130.0], [39.0, 131.0], [40.0, 131.0], [41.0, 131.0], [42.0, 131.0], [43.0, 131.0]]
    brc997d_holes = [[[110.0, 216.0], [109.0, 216.0], [109.0, 215.0], [109.0, 214.0], [110.0, 214.0], [110.0, 215.0], [110.0, 216.0]], [[192.0, 113.0], [192.0, 112.0], [192.0, 111.0], [193.0, 111.0], [193.0, 110.0], [193.0, 109.0], [194.0, 109.0], [194.0, 110.0], [195.0, 110.0], [195.0, 111.0], [194.0, 111.0], [194.0, 112.0], [193.0, 112.0], [193.0, 113.0], [192.0, 113.0]], [[195.0, 111.0], [196.0, 111.0], [197.0, 111.0], [197.0, 112.0], [196.0, 112.0], [195.0, 112.0], [195.0, 111.0]], [[63.0, 82.0], [63.0, 81.0], [63.0, 80.0], [63.0, 79.0], [63.0, 78.0], [64.0, 78.0], [65.0, 78.0], [66.0, 78.0], [66.0, 79.0], [66.0, 80.0], [66.0, 81.0], [66.0, 82.0], [65.0, 82.0], [64.0, 82.0], [63.0, 82.0]], [[184.0, 106.0], [184.0, 105.0], [184.0, 104.0], [185.0, 104.0], [186.0, 104.0], [186.0, 105.0], [186.0, 106.0], [185.0, 106.0], [184.0, 106.0]], [[184.0, 90.0], [185.0, 90.0], [185.0, 91.0], [185.0, 92.0], [184.0, 92.0], [183.0, 92.0], [183.0, 91.0], [183.0, 90.0], [184.0, 90.0]], [[183.0, 109.0], [182.0, 109.0], [181.0, 109.0], [181.0, 108.0], [181.0, 107.0], [182.0, 107.0], [183.0, 107.0], [183.0, 108.0], [183.0, 109.0]], [[175.0, 90.0], [174.0, 90.0], [173.0, 90.0], [173.0, 89.0], [173.0, 88.0], [173.0, 87.0], [174.0, 87.0], [175.0, 87.0], [176.0, 87.0], [176.0, 88.0], [176.0, 89.0], [176.0, 90.0], [175.0, 90.0]], [[173.0, 104.0], [174.0, 104.0], [175.0, 104.0], [176.0, 104.0], [176.0, 105.0], [176.0, 106.0], [176.0, 107.0], [176.0, 108.0], [175.0, 108.0], [174.0, 108.0], [173.0, 108.0], [173.0, 107.0], [172.0, 107.0], [172.0, 106.0], [172.0, 105.0], [173.0, 105.0], [173.0, 104.0]], [[174.0, 76.0], [174.0, 77.0], [173.0, 77.0], [173.0, 76.0], [174.0, 76.0]], [[170.0, 82.0], [171.0, 82.0], [172.0, 82.0], [173.0, 82.0], [173.0, 83.0], [172.0, 83.0], [172.0, 84.0], [171.0, 84.0], [171.0, 83.0], [170.0, 83.0], [170.0, 82.0]], [[168.0, 99.0], [167.0, 99.0], [167.0, 98.0], [168.0, 98.0], [168.0, 97.0], [169.0, 97.0], [170.0, 97.0], [170.0, 98.0], [170.0, 99.0], [169.0, 99.0], [169.0, 100.0], [168.0, 100.0], [168.0, 99.0]], [[167.0, 108.0], [167.0, 109.0], [166.0, 109.0], [166.0, 108.0], [165.0, 108.0], [165.0, 107.0], [165.0, 106.0], [165.0, 105.0], [166.0, 105.0], [166.0, 104.0], [167.0, 104.0], [168.0, 104.0], [168.0, 105.0], [168.0, 106.0], [167.0, 106.0], [167.0, 107.0], [167.0, 108.0]], [[161.0, 112.0], [161.0, 111.0], [162.0, 111.0], [162.0, 110.0], [163.0, 110.0], [164.0, 110.0], [164.0, 111.0], [164.0, 112.0], [164.0, 113.0], [163.0, 113.0], [163.0, 114.0], [162.0, 114.0], [161.0, 114.0], [160.0, 114.0], [160.0, 113.0], [160.0, 112.0], [161.0, 112.0]], [[152.0, 103.0], [153.0, 103.0], [154.0, 103.0], [154.0, 102.0], [155.0, 102.0], [155.0, 101.0], [156.0, 101.0], [156.0, 102.0], [157.0, 102.0], [158.0, 102.0], [158.0, 103.0], [158.0, 104.0], [159.0, 104.0], [159.0, 105.0], [159.0, 106.0], [158.0, 106.0], [158.0, 107.0], [157.0, 107.0], [157.0, 108.0], [156.0, 108.0], [155.0, 108.0], [154.0, 108.0], [153.0, 108.0], [153.0, 107.0], [153.0, 106.0], [152.0, 106.0], [152.0, 105.0], [152.0, 104.0], [152.0, 103.0]], [[153.0, 115.0], [154.0, 115.0], [155.0, 115.0], [155.0, 114.0], [156.0, 114.0], [157.0, 114.0], [158.0, 114.0], [158.0, 115.0], [158.0, 116.0], [157.0, 116.0], [157.0, 117.0], [156.0, 117.0], [155.0, 117.0], [154.0, 117.0], [153.0, 117.0], [153.0, 116.0], [153.0, 115.0]], [[151.0, 59.0], [152.0, 59.0], [152.0, 60.0], [153.0, 60.0], [153.0, 61.0], [153.0, 62.0], [152.0, 62.0], [152.0, 63.0], [151.0, 63.0], [150.0, 63.0], [150.0, 62.0], [149.0, 62.0], [149.0, 61.0], [149.0, 60.0], [150.0, 60.0], [150.0, 59.0], [151.0, 59.0]], [[146.0, 48.0], [147.0, 48.0], [147.0, 49.0], [146.0, 49.0], [146.0, 48.0]], [[147.0, 49.0], [148.0, 49.0], [149.0, 49.0], [149.0, 50.0], [149.0, 51.0], [148.0, 51.0], [148.0, 50.0], [147.0, 50.0], [147.0, 49.0]], [[147.0, 68.0], [148.0, 68.0], [149.0, 68.0], [149.0, 69.0], [149.0, 70.0], [149.0, 71.0], [148.0, 71.0], [147.0, 71.0], [147.0, 70.0], [147.0, 69.0], [147.0, 68.0]], [[143.0, 63.0], [143.0, 64.0], [143.0, 65.0], [142.0, 65.0], [141.0, 65.0], [141.0, 64.0], [141.0, 63.0], [142.0, 63.0], [143.0, 63.0]], [[142.0, 46.0], [143.0, 46.0], [143.0, 47.0], [142.0, 47.0], [142.0, 46.0]], [[141.0, 194.0], [141.0, 193.0], [142.0, 193.0], [143.0, 193.0], [143.0, 194.0], [143.0, 195.0], [143.0, 196.0], [142.0, 196.0], [141.0, 196.0], [141.0, 195.0], [141.0, 194.0]], [[137.0, 243.0], [137.0, 242.0], [138.0, 242.0], [139.0, 242.0], [139.0, 241.0], [140.0, 241.0], [140.0, 242.0], [141.0, 242.0], [141.0, 243.0], [141.0, 244.0], [141.0, 245.0], [140.0, 245.0], [139.0, 245.0], [138.0, 245.0], [138.0, 244.0], [137.0, 244.0], [137.0, 243.0]], [[136.0, 96.0], [136.0, 95.0], [136.0, 94.0], [136.0, 93.0], [136.0, 92.0], [137.0, 92.0], [138.0, 92.0], [139.0, 92.0], [140.0, 92.0], [140.0, 93.0], [140.0, 94.0], [140.0, 95.0], [139.0, 95.0], [139.0, 96.0], [138.0, 96.0], [137.0, 96.0], [136.0, 96.0]], [[108.0, 189.0], [108.0, 188.0], [107.0, 188.0], [107.0, 187.0], [107.0, 186.0], [107.0, 185.0], [107.0, 184.0], [107.0, 183.0], [107.0, 182.0], [107.0, 181.0], [108.0, 181.0], [108.0, 180.0], [108.0, 179.0], [109.0, 179.0], [109.0, 178.0], [110.0, 178.0], [111.0, 178.0], [111.0, 177.0], [112.0, 177.0], [113.0, 177.0], [113.0, 176.0], [114.0, 176.0], [115.0, 176.0], [116.0, 176.0], [116.0, 175.0], [117.0, 175.0], [118.0, 175.0], [119.0, 175.0], [119.0, 174.0], [120.0, 174.0], [121.0, 174.0], [121.0, 175.0], [121.0, 176.0], [122.0, 176.0], [122.0, 177.0], [123.0, 177.0], [124.0, 177.0], [124.0, 178.0], [125.0, 178.0], [126.0, 178.0], [126.0, 179.0], [127.0, 179.0], [127.0, 178.0], [128.0, 178.0], [129.0, 178.0], [129.0, 179.0], [129.0, 180.0], [130.0, 180.0], [131.0, 180.0], [131.0, 181.0], [131.0, 182.0], [132.0, 182.0], [132.0, 183.0], [133.0, 183.0], [133.0, 184.0], [134.0, 184.0], [134.0, 185.0], [135.0, 185.0], [135.0, 186.0], [135.0, 187.0], [136.0, 187.0], [136.0, 188.0], [136.0, 189.0], [136.0, 190.0], [136.0, 191.0], [136.0, 192.0], [136.0, 193.0], [135.0, 193.0], [135.0, 194.0], [135.0, 195.0], [134.0, 195.0], [134.0, 196.0], [134.0, 197.0], [133.0, 197.0], [133.0, 198.0], [132.0, 198.0], [132.0, 199.0], [132.0, 200.0], [131.0, 200.0], [131.0, 201.0], [130.0, 201.0], [130.0, 202.0], [129.0, 202.0], [129.0, 203.0], [128.0, 203.0], [128.0, 204.0], [127.0, 204.0], [127.0, 203.0], [126.0, 203.0], [125.0, 203.0], [124.0, 203.0], [123.0, 203.0], [122.0, 203.0], [121.0, 203.0], [120.0, 203.0], [120.0, 202.0], [119.0, 202.0], [119.0, 201.0], [119.0, 200.0], [118.0, 200.0], [118.0, 199.0], [118.0, 198.0], [117.0, 198.0], [117.0, 197.0], [116.0, 197.0], [116.0, 196.0], [116.0, 195.0], [115.0, 195.0], [115.0, 194.0], [114.0, 194.0], [114.0, 193.0], [113.0, 193.0], [113.0, 192.0], [112.0, 192.0], [112.0, 191.0], [111.0, 191.0], [111.0, 190.0], [110.0, 190.0], [110.0, 189.0], [109.0, 189.0], [108.0, 189.0]], [[134.0, 46.0], [134.0, 47.0], [134.0, 48.0], [133.0, 48.0], [132.0, 48.0], [132.0, 47.0], [132.0, 46.0], [133.0, 46.0], [134.0, 46.0]], [[132.0, 248.0], [133.0, 248.0], [134.0, 248.0], [134.0, 249.0], [134.0, 250.0], [133.0, 250.0], [132.0, 250.0], [131.0, 250.0], [131.0, 249.0], [131.0, 248.0], [132.0, 248.0]], [[121.0, 80.0], [120.0, 80.0], [120.0, 79.0], [119.0, 79.0], [119.0, 78.0], [119.0, 77.0], [120.0, 77.0], [120.0, 76.0], [120.0, 75.0], [121.0, 75.0], [121.0, 74.0], [121.0, 73.0], [122.0, 73.0], [123.0, 73.0], [123.0, 72.0], [124.0, 72.0], [125.0, 72.0], [126.0, 72.0], [127.0, 72.0], [128.0, 72.0], [129.0, 72.0], [130.0, 72.0], [130.0, 73.0], [131.0, 73.0], [131.0, 74.0], [132.0, 74.0], [132.0, 75.0], [132.0, 76.0], [133.0, 76.0], [133.0, 77.0], [132.0, 77.0], [132.0, 78.0], [131.0, 78.0], [130.0, 78.0], [130.0, 79.0], [131.0, 79.0], [131.0, 80.0], [131.0, 81.0], [131.0, 82.0], [131.0, 83.0], [132.0, 83.0], [132.0, 84.0], [132.0, 85.0], [132.0, 86.0], [132.0, 87.0], [133.0, 87.0], [133.0, 88.0], [133.0, 89.0], [133.0, 90.0], [132.0, 90.0], [132.0, 91.0], [131.0, 91.0], [131.0, 92.0], [132.0, 92.0], [132.0, 93.0], [132.0, 94.0], [131.0, 94.0], [130.0, 94.0], [129.0, 94.0], [129.0, 93.0], [128.0, 93.0], [128.0, 92.0], [127.0, 92.0], [126.0, 92.0], [126.0, 91.0], [126.0, 90.0], [126.0, 89.0], [125.0, 89.0], [125.0, 88.0], [125.0, 87.0], [125.0, 86.0], [125.0, 85.0], [124.0, 85.0], [124.0, 84.0], [124.0, 83.0], [124.0, 82.0], [124.0, 81.0], [123.0, 81.0], [123.0, 80.0], [122.0, 80.0], [121.0, 80.0]], [[131.0, 117.0], [130.0, 117.0], [130.0, 116.0], [131.0, 116.0], [131.0, 117.0]], [[122.0, 48.0], [122.0, 47.0], [122.0, 46.0], [123.0, 46.0], [123.0, 45.0], [124.0, 45.0], [125.0, 45.0], [126.0, 45.0], [126.0, 46.0], [127.0, 46.0], [127.0, 47.0], [127.0, 48.0], [127.0, 49.0], [127.0, 50.0], [126.0, 50.0], [125.0, 50.0], [124.0, 50.0], [123.0, 50.0], [122.0, 50.0], [122.0, 49.0], [122.0, 48.0]], [[124.0, 118.0], [124.0, 117.0], [125.0, 117.0], [125.0, 118.0], [124.0, 118.0]], [[121.0, 67.0], [120.0, 67.0], [120.0, 66.0], [120.0, 65.0], [120.0, 64.0], [121.0, 64.0], [122.0, 64.0], [122.0, 63.0], [123.0, 63.0], [123.0, 64.0], [124.0, 64.0], [124.0, 65.0], [124.0, 66.0], [124.0, 67.0], [123.0, 67.0], [122.0, 67.0], [121.0, 67.0]], [[123.0, 23.0], [123.0, 24.0], [122.0, 24.0], [121.0, 24.0], [121.0, 23.0], [121.0, 22.0], [121.0, 21.0], [122.0, 21.0], [123.0, 21.0], [123.0, 22.0], [123.0, 23.0]], [[118.0, 46.0], [119.0, 46.0], [120.0, 46.0], [120.0, 47.0], [120.0, 48.0], [119.0, 48.0], [118.0, 48.0], [117.0, 48.0], [117.0, 47.0], [118.0, 47.0], [118.0, 46.0]], [[111.0, 57.0], [111.0, 56.0], [110.0, 56.0], [109.0, 56.0], [109.0, 57.0], [108.0, 57.0], [107.0, 57.0], [106.0, 57.0], [105.0, 57.0], [105.0, 56.0], [104.0, 56.0], [103.0, 56.0], [102.0, 56.0], [101.0, 56.0], [101.0, 55.0], [100.0, 55.0], [99.0, 55.0], [98.0, 55.0], [98.0, 54.0], [98.0, 53.0], [98.0, 52.0], [99.0, 52.0], [99.0, 51.0], [99.0, 50.0], [99.0, 49.0], [99.0, 48.0], [100.0, 48.0], [101.0, 48.0], [102.0, 48.0], [103.0, 48.0], [104.0, 48.0], [104.0, 49.0], [105.0, 49.0], [106.0, 49.0], [107.0, 49.0], [108.0, 49.0], [108.0, 50.0], [109.0, 50.0], [110.0, 50.0], [111.0, 50.0], [111.0, 51.0], [111.0, 52.0], [112.0, 52.0], [113.0, 52.0], [114.0, 52.0], [114.0, 53.0], [115.0, 53.0], [116.0, 53.0], [117.0, 53.0], [117.0, 54.0], [117.0, 55.0], [116.0, 55.0], [116.0, 56.0], [116.0, 57.0], [115.0, 57.0], [114.0, 57.0], [113.0, 57.0], [112.0, 57.0], [111.0, 57.0]], [[112.0, 24.0], [112.0, 23.0], [112.0, 22.0], [113.0, 22.0], [114.0, 22.0], [115.0, 22.0], [115.0, 23.0], [115.0, 24.0], [115.0, 25.0], [114.0, 25.0], [113.0, 25.0], [112.0, 25.0], [112.0, 24.0]], [[111.0, 39.0], [110.0, 39.0], [110.0, 38.0], [110.0, 37.0], [111.0, 37.0], [112.0, 37.0], [112.0, 38.0], [112.0, 39.0], [111.0, 39.0]], [[103.0, 35.0], [103.0, 34.0], [103.0, 33.0], [104.0, 33.0], [104.0, 32.0], [105.0, 32.0], [106.0, 32.0], [106.0, 33.0], [107.0, 33.0], [107.0, 34.0], [107.0, 35.0], [106.0, 35.0], [106.0, 36.0], [105.0, 36.0], [104.0, 36.0], [103.0, 36.0], [103.0, 35.0]], [[96.0, 98.0], [95.0, 98.0], [94.0, 98.0], [94.0, 97.0], [94.0, 96.0], [93.0, 96.0], [93.0, 95.0], [93.0, 94.0], [92.0, 94.0], [92.0, 93.0], [93.0, 93.0], [93.0, 92.0], [93.0, 91.0], [94.0, 91.0], [95.0, 91.0], [96.0, 91.0], [96.0, 90.0], [96.0, 89.0], [96.0, 88.0], [96.0, 87.0], [96.0, 86.0], [97.0, 86.0], [97.0, 85.0], [97.0, 84.0], [97.0, 83.0], [97.0, 82.0], [97.0, 81.0], [98.0, 81.0], [98.0, 80.0], [98.0, 79.0], [99.0, 79.0], [100.0, 79.0], [100.0, 78.0], [100.0, 77.0], [101.0, 77.0], [101.0, 76.0], [102.0, 76.0], [103.0, 76.0], [104.0, 76.0], [104.0, 77.0], [104.0, 78.0], [104.0, 79.0], [103.0, 79.0], [103.0, 80.0], [104.0, 80.0], [105.0, 80.0], [105.0, 81.0], [105.0, 82.0], [105.0, 83.0], [104.0, 83.0], [104.0, 84.0], [104.0, 85.0], [104.0, 86.0], [104.0, 87.0], [104.0, 88.0], [104.0, 89.0], [103.0, 89.0], [103.0, 90.0], [103.0, 91.0], [103.0, 92.0], [104.0, 92.0], [104.0, 93.0], [105.0, 93.0], [106.0, 93.0], [106.0, 94.0], [106.0, 95.0], [106.0, 96.0], [105.0, 96.0], [105.0, 97.0], [104.0, 97.0], [104.0, 98.0], [103.0, 98.0], [103.0, 99.0], [102.0, 99.0], [101.0, 99.0], [100.0, 99.0], [99.0, 99.0], [98.0, 99.0], [97.0, 99.0], [97.0, 98.0], [96.0, 98.0]], [[103.0, 42.0], [102.0, 42.0], [102.0, 41.0], [102.0, 40.0], [102.0, 39.0], [103.0, 39.0], [104.0, 39.0], [105.0, 39.0], [105.0, 40.0], [105.0, 41.0], [105.0, 42.0], [104.0, 42.0], [104.0, 43.0], [103.0, 43.0], [103.0, 42.0]], [[96.0, 77.0], [95.0, 77.0], [95.0, 76.0], [95.0, 75.0], [95.0, 74.0], [96.0, 74.0], [97.0, 74.0], [97.0, 73.0], [98.0, 73.0], [99.0, 73.0], [99.0, 74.0], [99.0, 75.0], [99.0, 76.0], [99.0, 77.0], [98.0, 77.0], [98.0, 78.0], [97.0, 78.0], [96.0, 78.0], [96.0, 77.0]], [[92.0, 33.0], [92.0, 32.0], [92.0, 31.0], [92.0, 30.0], [93.0, 30.0], [93.0, 29.0], [94.0, 29.0], [95.0, 29.0], [96.0, 29.0], [97.0, 29.0], [97.0, 30.0], [97.0, 31.0], [97.0, 32.0], [97.0, 33.0], [97.0, 34.0], [96.0, 34.0], [95.0, 34.0], [94.0, 34.0], [93.0, 34.0], [93.0, 33.0], [92.0, 33.0]], [[83.0, 129.0], [83.0, 128.0], [83.0, 127.0], [83.0, 126.0], [84.0, 126.0], [84.0, 125.0], [85.0, 125.0], [85.0, 124.0], [86.0, 124.0], [87.0, 124.0], [87.0, 123.0], [88.0, 123.0], [89.0, 123.0], [90.0, 123.0], [91.0, 123.0], [92.0, 123.0], [93.0, 123.0], [93.0, 124.0], [94.0, 124.0], [94.0, 125.0], [95.0, 125.0], [95.0, 126.0], [96.0, 126.0], [96.0, 127.0], [96.0, 128.0], [96.0, 129.0], [96.0, 130.0], [95.0, 130.0], [94.0, 130.0], [93.0, 130.0], [93.0, 131.0], [94.0, 131.0], [94.0, 132.0], [94.0, 133.0], [94.0, 134.0], [94.0, 135.0], [94.0, 136.0], [94.0, 137.0], [94.0, 138.0], [94.0, 139.0], [94.0, 140.0], [95.0, 140.0], [95.0, 141.0], [95.0, 142.0], [94.0, 142.0], [93.0, 142.0], [93.0, 143.0], [93.0, 144.0], [93.0, 145.0], [93.0, 146.0], [92.0, 146.0], [91.0, 146.0], [90.0, 146.0], [90.0, 145.0], [90.0, 144.0], [89.0, 144.0], [89.0, 143.0], [88.0, 143.0], [87.0, 143.0], [87.0, 142.0], [87.0, 141.0], [87.0, 140.0], [87.0, 139.0], [87.0, 138.0], [87.0, 137.0], [87.0, 136.0], [87.0, 135.0], [87.0, 134.0], [86.0, 134.0], [86.0, 133.0], [86.0, 132.0], [86.0, 131.0], [85.0, 131.0], [84.0, 131.0], [83.0, 131.0], [83.0, 130.0], [83.0, 129.0]], [[94.0, 25.0], [94.0, 26.0], [93.0, 26.0], [92.0, 26.0], [92.0, 25.0], [92.0, 24.0], [93.0, 24.0], [94.0, 24.0], [94.0, 25.0]], [[86.0, 24.0], [87.0, 24.0], [88.0, 24.0], [88.0, 25.0], [88.0, 26.0], [88.0, 27.0], [87.0, 27.0], [86.0, 27.0], [86.0, 26.0], [86.0, 25.0], [86.0, 24.0]], [[86.0, 74.0], [86.0, 75.0], [86.0, 76.0], [86.0, 77.0], [85.0, 77.0], [84.0, 77.0], [84.0, 76.0], [84.0, 75.0], [84.0, 74.0], [85.0, 74.0], [86.0, 74.0]], [[74.0, 62.0], [74.0, 61.0], [75.0, 61.0], [76.0, 61.0], [77.0, 61.0], [77.0, 62.0], [77.0, 63.0], [76.0, 63.0], [75.0, 63.0], [75.0, 62.0], [74.0, 62.0]], [[74.0, 53.0], [74.0, 52.0], [73.0, 52.0], [73.0, 51.0], [73.0, 50.0], [73.0, 49.0], [74.0, 49.0], [75.0, 49.0], [75.0, 48.0], [76.0, 48.0], [76.0, 49.0], [77.0, 49.0], [77.0, 50.0], [77.0, 51.0], [77.0, 52.0], [76.0, 52.0], [75.0, 52.0], [75.0, 53.0], [74.0, 53.0]], [[62.0, 181.0], [61.0, 181.0], [61.0, 180.0], [60.0, 180.0], [59.0, 180.0], [59.0, 179.0], [59.0, 178.0], [58.0, 178.0], [58.0, 177.0], [57.0, 177.0], [57.0, 176.0], [57.0, 175.0], [57.0, 174.0], [57.0, 173.0], [57.0, 172.0], [57.0, 171.0], [58.0, 171.0], [58.0, 170.0], [59.0, 170.0], [59.0, 169.0], [60.0, 169.0], [60.0, 168.0], [61.0, 168.0], [62.0, 168.0], [62.0, 169.0], [63.0, 169.0], [63.0, 170.0], [64.0, 170.0], [65.0, 170.0], [65.0, 169.0], [66.0, 169.0], [67.0, 169.0], [68.0, 169.0], [68.0, 168.0], [69.0, 168.0], [70.0, 168.0], [70.0, 167.0], [71.0, 167.0], [72.0, 167.0], [73.0, 167.0], [73.0, 166.0], [74.0, 166.0], [75.0, 166.0], [75.0, 167.0], [75.0, 168.0], [76.0, 168.0], [76.0, 169.0], [76.0, 170.0], [77.0, 170.0], [77.0, 171.0], [77.0, 172.0], [77.0, 173.0], [76.0, 173.0], [76.0, 174.0], [75.0, 174.0], [74.0, 174.0], [73.0, 174.0], [73.0, 175.0], [72.0, 175.0], [71.0, 175.0], [70.0, 175.0], [70.0, 176.0], [69.0, 176.0], [68.0, 176.0], [68.0, 177.0], [67.0, 177.0], [66.0, 177.0], [66.0, 178.0], [66.0, 179.0], [67.0, 179.0], [67.0, 180.0], [66.0, 180.0], [66.0, 181.0], [65.0, 181.0], [64.0, 181.0], [63.0, 181.0], [62.0, 181.0]], [[73.0, 84.0], [74.0, 84.0], [74.0, 85.0], [75.0, 85.0], [75.0, 86.0], [74.0, 86.0], [74.0, 87.0], [73.0, 87.0], [73.0, 86.0], [73.0, 85.0], [73.0, 84.0]], [[69.0, 66.0], [70.0, 66.0], [70.0, 67.0], [70.0, 68.0], [69.0, 68.0], [68.0, 68.0], [68.0, 67.0], [68.0, 66.0], [69.0, 66.0]], [[68.0, 58.0], [69.0, 58.0], [69.0, 59.0], [69.0, 60.0], [68.0, 60.0], [67.0, 60.0], [66.0, 60.0], [66.0, 59.0], [66.0, 58.0], [67.0, 58.0], [68.0, 58.0]], [[67.0, 189.0], [67.0, 190.0], [66.0, 190.0], [65.0, 190.0], [65.0, 189.0], [65.0, 188.0], [66.0, 188.0], [67.0, 188.0], [67.0, 189.0]], [[55.0, 154.0], [55.0, 153.0], [56.0, 153.0], [57.0, 153.0], [57.0, 152.0], [58.0, 152.0], [58.0, 153.0], [59.0, 153.0], [59.0, 154.0], [59.0, 155.0], [59.0, 156.0], [59.0, 157.0], [58.0, 157.0], [57.0, 157.0], [56.0, 157.0], [56.0, 156.0], [55.0, 156.0], [55.0, 155.0], [55.0, 154.0]], [[47.0, 198.0], [48.0, 198.0], [49.0, 198.0], [49.0, 199.0], [50.0, 199.0], [50.0, 200.0], [50.0, 201.0], [49.0, 201.0], [49.0, 202.0], [48.0, 202.0], [47.0, 202.0], [47.0, 201.0], [46.0, 201.0], [46.0, 200.0], [46.0, 199.0], [47.0, 199.0], [47.0, 198.0]], [[43.0, 196.0], [43.0, 195.0], [42.0, 195.0], [42.0, 194.0], [43.0, 194.0], [43.0, 193.0], [44.0, 193.0], [44.0, 194.0], [45.0, 194.0], [45.0, 195.0], [44.0, 195.0], [44.0, 196.0], [43.0, 196.0]], [[40.0, 155.0], [40.0, 154.0], [41.0, 154.0], [42.0, 154.0], [42.0, 155.0], [42.0, 156.0], [41.0, 156.0], [40.0, 156.0], [40.0, 157.0], [39.0, 157.0], [38.0, 157.0], [38.0, 156.0], [38.0, 155.0], [39.0, 155.0], [40.0, 155.0]], [[37.0, 183.0], [38.0, 183.0], [39.0, 183.0], [39.0, 184.0], [39.0, 185.0], [38.0, 185.0], [37.0, 185.0], [37.0, 184.0], [37.0, 183.0]], [[34.0, 147.0], [33.0, 147.0], [32.0, 147.0], [32.0, 146.0], [32.0, 145.0], [33.0, 145.0], [34.0, 145.0], [34.0, 146.0], [34.0, 147.0]]]
    
    brc203_boundary = [[23.0, 162.0], [23.0, 163.0], [23.0, 164.0], [23.0, 165.0], [23.0, 166.0], [22.0, 166.0], [22.0, 167.0], [22.0, 168.0], [22.0, 169.0], [22.0, 170.0], [21.0, 170.0], [20.0, 170.0], [19.0, 170.0], [18.0, 170.0], [18.0, 169.0], [17.0, 169.0], [17.0, 170.0], [17.0, 171.0], [18.0, 171.0], [19.0, 171.0], [19.0, 172.0], [19.0, 173.0], [20.0, 173.0], [20.0, 174.0], [20.0, 175.0], [20.0, 176.0], [19.0, 176.0], [19.0, 177.0], [19.0, 178.0], [18.0, 178.0], [18.0, 179.0], [17.0, 179.0], [17.0, 180.0], [17.0, 181.0], [17.0, 182.0], [17.0, 183.0], [18.0, 183.0], [19.0, 183.0], [19.0, 182.0], [20.0, 182.0], [21.0, 182.0], [22.0, 182.0], [22.0, 181.0], [23.0, 181.0], [24.0, 181.0], [24.0, 182.0], [24.0, 183.0], [23.0, 183.0], [23.0, 184.0], [22.0, 184.0], [22.0, 185.0], [22.0, 186.0], [21.0, 186.0], [20.0, 186.0], [20.0, 185.0], [19.0, 185.0], [18.0, 185.0], [17.0, 185.0], [17.0, 186.0], [18.0, 186.0], [19.0, 186.0], [19.0, 187.0], [19.0, 188.0], [19.0, 189.0], [19.0, 190.0], [20.0, 190.0], [21.0, 190.0], [21.0, 189.0], [21.0, 188.0], [22.0, 188.0], [23.0, 188.0], [24.0, 188.0], [24.0, 189.0], [25.0, 189.0], [25.0, 190.0], [26.0, 190.0], [27.0, 190.0], [28.0, 190.0], [28.0, 191.0], [28.0, 192.0], [29.0, 192.0], [30.0, 192.0], [30.0, 191.0], [31.0, 191.0], [31.0, 190.0], [32.0, 190.0], [33.0, 190.0], [34.0, 190.0], [34.0, 191.0], [35.0, 191.0], [36.0, 191.0], [37.0, 191.0], [38.0, 191.0], [39.0, 191.0], [39.0, 192.0], [39.0, 193.0], [39.0, 194.0], [38.0, 194.0], [38.0, 195.0], [37.0, 195.0], [36.0, 195.0], [35.0, 195.0], [34.0, 195.0], [34.0, 196.0], [33.0, 196.0], [33.0, 197.0], [33.0, 198.0], [33.0, 199.0], [34.0, 199.0], [35.0, 199.0], [36.0, 199.0], [36.0, 200.0], [36.0, 201.0], [36.0, 202.0], [36.0, 203.0], [35.0, 203.0], [34.0, 203.0], [33.0, 203.0], [33.0, 204.0], [33.0, 205.0], [34.0, 205.0], [34.0, 204.0], [35.0, 204.0], [36.0, 204.0], [36.0, 205.0], [36.0, 206.0], [36.0, 207.0], [35.0, 207.0], [34.0, 207.0], [33.0, 207.0], [33.0, 208.0], [34.0, 208.0], [35.0, 208.0], [35.0, 209.0], [35.0, 210.0], [36.0, 210.0], [36.0, 211.0], [37.0, 211.0], [37.0, 212.0], [37.0, 213.0], [38.0, 213.0], [38.0, 214.0], [38.0, 215.0], [39.0, 215.0], [39.0, 216.0], [39.0, 217.0], [39.0, 218.0], [38.0, 218.0], [38.0, 219.0], [37.0, 219.0], [37.0, 220.0], [37.0, 221.0], [37.0, 222.0], [37.0, 223.0], [36.0, 223.0], [35.0, 223.0], [35.0, 224.0], [36.0, 224.0], [37.0, 224.0], [38.0, 224.0], [38.0, 225.0], [38.0, 226.0], [38.0, 227.0], [37.0, 227.0], [36.0, 227.0], [35.0, 227.0], [34.0, 227.0], [34.0, 228.0], [33.0, 228.0], [33.0, 229.0], [33.0, 230.0], [34.0, 230.0], [34.0, 229.0], [35.0, 229.0], [36.0, 229.0], [36.0, 230.0], [36.0, 231.0], [36.0, 232.0], [36.0, 233.0], [36.0, 234.0], [36.0, 235.0], [35.0, 235.0], [34.0, 235.0], [33.0, 235.0], [33.0, 236.0], [33.0, 237.0], [33.0, 238.0], [34.0, 238.0], [35.0, 238.0], [35.0, 239.0], [36.0, 239.0], [37.0, 239.0], [38.0, 239.0], [39.0, 239.0], [39.0, 240.0], [39.0, 241.0], [39.0, 242.0], [38.0, 242.0], [38.0, 243.0], [37.0, 243.0], [36.0, 243.0], [35.0, 243.0], [34.0, 243.0], [34.0, 244.0], [33.0, 244.0], [32.0, 244.0], [31.0, 244.0], [31.0, 243.0], [31.0, 242.0], [30.0, 242.0], [29.0, 242.0], [28.0, 242.0], [27.0, 242.0], [26.0, 242.0], [25.0, 242.0], [24.0, 242.0], [23.0, 242.0], [22.0, 242.0], [21.0, 242.0], [20.0, 242.0], [20.0, 243.0], [21.0, 243.0], [22.0, 243.0], [22.0, 244.0], [23.0, 244.0], [23.0, 245.0], [24.0, 245.0], [25.0, 245.0], [25.0, 246.0], [25.0, 247.0], [24.0, 247.0], [24.0, 248.0], [23.0, 248.0], [23.0, 249.0], [23.0, 250.0], [22.0, 250.0], [22.0, 251.0], [22.0, 252.0], [22.0, 253.0], [22.0, 254.0], [22.0, 255.0], [23.0, 255.0], [24.0, 255.0], [24.0, 256.0], [24.0, 257.0], [24.0, 258.0], [23.0, 258.0], [23.0, 259.0], [22.0, 259.0], [22.0, 260.0], [21.0, 260.0], [21.0, 261.0], [21.0, 262.0], [21.0, 263.0], [21.0, 264.0], [21.0, 265.0], [21.0, 266.0], [20.0, 266.0], [19.0, 266.0], [19.0, 267.0], [19.0, 268.0], [18.0, 268.0], [17.0, 268.0], [17.0, 269.0], [17.0, 270.0], [17.0, 271.0], [18.0, 271.0], [19.0, 271.0], [19.0, 272.0], [20.0, 272.0], [21.0, 272.0], [22.0, 272.0], [23.0, 272.0], [24.0, 272.0], [25.0, 272.0], [26.0, 272.0], [27.0, 272.0], [27.0, 271.0], [28.0, 271.0], [29.0, 271.0], [29.0, 272.0], [30.0, 272.0], [31.0, 272.0], [31.0, 271.0], [32.0, 271.0], [33.0, 271.0], [34.0, 271.0], [35.0, 271.0], [35.0, 272.0], [36.0, 272.0], [37.0, 272.0], [38.0, 272.0], [38.0, 271.0], [38.0, 270.0], [38.0, 269.0], [38.0, 268.0], [38.0, 267.0], [38.0, 266.0], [38.0, 265.0], [38.0, 264.0], [39.0, 264.0], [39.0, 265.0], [39.0, 266.0], [39.0, 267.0], [39.0, 268.0], [39.0, 269.0], [40.0, 269.0], [40.0, 268.0], [40.0, 267.0], [41.0, 267.0], [42.0, 267.0], [43.0, 267.0], [44.0, 267.0], [45.0, 267.0], [46.0, 267.0], [47.0, 267.0], [47.0, 268.0], [47.0, 269.0], [47.0, 270.0], [47.0, 271.0], [48.0, 271.0], [48.0, 270.0], [48.0, 269.0], [48.0, 268.0], [48.0, 267.0], [48.0, 266.0], [48.0, 265.0], [48.0, 264.0], [48.0, 263.0], [48.0, 262.0], [48.0, 261.0], [48.0, 260.0], [48.0, 259.0], [47.0, 259.0], [47.0, 260.0], [47.0, 261.0], [47.0, 262.0], [46.0, 262.0], [45.0, 262.0], [44.0, 262.0], [43.0, 262.0], [42.0, 262.0], [42.0, 261.0], [42.0, 260.0], [42.0, 259.0], [42.0, 258.0], [42.0, 257.0], [42.0, 256.0], [42.0, 255.0], [42.0, 254.0], [43.0, 254.0], [44.0, 254.0], [45.0, 254.0], [46.0, 254.0], [47.0, 254.0], [47.0, 255.0], [48.0, 255.0], [48.0, 254.0], [48.0, 253.0], [47.0, 253.0], [47.0, 252.0], [47.0, 251.0], [47.0, 250.0], [48.0, 250.0], [48.0, 249.0], [48.0, 248.0], [47.0, 248.0], [46.0, 248.0], [46.0, 247.0], [46.0, 246.0], [46.0, 245.0], [46.0, 244.0], [45.0, 244.0], [45.0, 243.0], [44.0, 243.0], [43.0, 243.0], [43.0, 242.0], [43.0, 241.0], [43.0, 240.0], [43.0, 239.0], [44.0, 239.0], [45.0, 239.0], [46.0, 239.0], [47.0, 239.0], [47.0, 238.0], [48.0, 238.0], [48.0, 237.0], [48.0, 236.0], [48.0, 235.0], [47.0, 235.0], [46.0, 235.0], [45.0, 235.0], [45.0, 234.0], [45.0, 233.0], [45.0, 232.0], [46.0, 232.0], [46.0, 231.0], [46.0, 230.0], [47.0, 230.0], [47.0, 229.0], [48.0, 229.0], [48.0, 228.0], [47.0, 228.0], [47.0, 227.0], [47.0, 226.0], [46.0, 226.0], [46.0, 225.0], [47.0, 225.0], [47.0, 224.0], [48.0, 224.0], [49.0, 224.0], [50.0, 224.0], [51.0, 224.0], [52.0, 224.0], [53.0, 224.0], [54.0, 224.0], [55.0, 224.0], [56.0, 224.0], [57.0, 224.0], [58.0, 224.0], [58.0, 223.0], [58.0, 222.0], [59.0, 222.0], [60.0, 222.0], [61.0, 222.0], [62.0, 222.0], [62.0, 223.0], [62.0, 224.0], [63.0, 224.0], [63.0, 223.0], [63.0, 222.0], [63.0, 221.0], [63.0, 220.0], [63.0, 219.0], [64.0, 219.0], [65.0, 219.0], [66.0, 219.0], [66.0, 220.0], [66.0, 221.0], [66.0, 222.0], [67.0, 222.0], [67.0, 221.0], [68.0, 221.0], [68.0, 222.0], [69.0, 222.0], [70.0, 222.0], [71.0, 222.0], [71.0, 223.0], [71.0, 224.0], [71.0, 225.0], [70.0, 225.0], [70.0, 226.0], [71.0, 226.0], [71.0, 227.0], [71.0, 228.0], [71.0, 229.0], [71.0, 230.0], [71.0, 231.0], [70.0, 231.0], [70.0, 232.0], [69.0, 232.0], [69.0, 233.0], [68.0, 233.0], [67.0, 233.0], [66.0, 233.0], [66.0, 234.0], [66.0, 235.0], [66.0, 236.0], [66.0, 237.0], [66.0, 238.0], [66.0, 239.0], [67.0, 239.0], [67.0, 240.0], [68.0, 240.0], [69.0, 240.0], [70.0, 240.0], [71.0, 240.0], [72.0, 240.0], [73.0, 240.0], [74.0, 240.0], [75.0, 240.0], [76.0, 240.0], [77.0, 240.0], [78.0, 240.0], [78.0, 239.0], [78.0, 238.0], [78.0, 237.0], [77.0, 237.0], [77.0, 238.0], [77.0, 239.0], [76.0, 239.0], [75.0, 239.0], [74.0, 239.0], [74.0, 238.0], [74.0, 237.0], [75.0, 237.0], [75.0, 236.0], [76.0, 236.0], [76.0, 235.0], [76.0, 234.0], [76.0, 233.0], [77.0, 233.0], [78.0, 233.0], [79.0, 233.0], [80.0, 233.0], [81.0, 233.0], [81.0, 232.0], [82.0, 232.0], [83.0, 232.0], [84.0, 232.0], [84.0, 233.0], [85.0, 233.0], [86.0, 233.0], [86.0, 234.0], [87.0, 234.0], [87.0, 235.0], [87.0, 236.0], [87.0, 237.0], [87.0, 238.0], [87.0, 239.0], [87.0, 240.0], [88.0, 240.0], [89.0, 240.0], [90.0, 240.0], [91.0, 240.0], [92.0, 240.0], [93.0, 240.0], [94.0, 240.0], [95.0, 240.0], [95.0, 239.0], [96.0, 239.0], [96.0, 238.0], [97.0, 238.0], [98.0, 238.0], [99.0, 238.0], [100.0, 238.0], [101.0, 238.0], [101.0, 239.0], [102.0, 239.0], [103.0, 239.0], [103.0, 240.0], [103.0, 241.0], [103.0, 242.0], [102.0, 242.0], [102.0, 243.0], [101.0, 243.0], [100.0, 243.0], [99.0, 243.0], [98.0, 243.0], [98.0, 244.0], [98.0, 245.0], [98.0, 246.0], [98.0, 247.0], [98.0, 248.0], [98.0, 249.0], [98.0, 250.0], [98.0, 251.0], [98.0, 252.0], [98.0, 253.0], [98.0, 254.0], [98.0, 255.0], [99.0, 255.0], [99.0, 256.0], [99.0, 257.0], [99.0, 258.0], [98.0, 258.0], [98.0, 259.0], [98.0, 260.0], [98.0, 261.0], [98.0, 262.0], [98.0, 263.0], [98.0, 264.0], [98.0, 265.0], [98.0, 266.0], [98.0, 267.0], [98.0, 268.0], [98.0, 269.0], [98.0, 270.0], [98.0, 271.0], [99.0, 271.0], [99.0, 272.0], [99.0, 273.0], [99.0, 274.0], [98.0, 274.0], [98.0, 275.0], [98.0, 276.0], [98.0, 277.0], [98.0, 278.0], [98.0, 279.0], [98.0, 280.0], [98.0, 281.0], [99.0, 281.0], [100.0, 281.0], [101.0, 281.0], [101.0, 282.0], [101.0, 283.0], [101.0, 284.0], [101.0, 285.0], [101.0, 286.0], [102.0, 286.0], [103.0, 286.0], [103.0, 287.0], [103.0, 288.0], [103.0, 289.0], [102.0, 289.0], [102.0, 290.0], [101.0, 290.0], [100.0, 290.0], [100.0, 291.0], [101.0, 291.0], [101.0, 292.0], [101.0, 293.0], [100.0, 293.0], [100.0, 294.0], [99.0, 294.0], [98.0, 294.0], [98.0, 295.0], [98.0, 296.0], [99.0, 296.0], [99.0, 297.0], [99.0, 298.0], [99.0, 299.0], [99.0, 300.0], [98.0, 300.0], [97.0, 300.0], [96.0, 300.0], [95.0, 300.0], [95.0, 299.0], [94.0, 299.0], [94.0, 300.0], [93.0, 300.0], [92.0, 300.0], [91.0, 300.0], [91.0, 301.0], [91.0, 302.0], [90.0, 302.0], [89.0, 302.0], [89.0, 303.0], [88.0, 303.0], [88.0, 304.0], [88.0, 305.0], [87.0, 305.0], [87.0, 306.0], [88.0, 306.0], [88.0, 307.0], [88.0, 308.0], [87.0, 308.0], [87.0, 309.0], [87.0, 310.0], [87.0, 311.0], [87.0, 312.0], [87.0, 313.0], [87.0, 314.0], [86.0, 314.0], [86.0, 315.0], [87.0, 315.0], [87.0, 316.0], [87.0, 317.0], [86.0, 317.0], [86.0, 318.0], [85.0, 318.0], [84.0, 318.0], [83.0, 318.0], [83.0, 317.0], [82.0, 317.0], [82.0, 316.0], [81.0, 316.0], [81.0, 317.0], [81.0, 318.0], [82.0, 318.0], [82.0, 319.0], [83.0, 319.0], [83.0, 320.0], [84.0, 320.0], [84.0, 321.0], [85.0, 321.0], [85.0, 320.0], [86.0, 320.0], [86.0, 321.0], [86.0, 322.0], [86.0, 323.0], [86.0, 324.0], [85.0, 324.0], [85.0, 325.0], [84.0, 325.0], [83.0, 325.0], [82.0, 325.0], [82.0, 324.0], [81.0, 324.0], [81.0, 323.0], [80.0, 323.0], [79.0, 323.0], [79.0, 322.0], [78.0, 322.0], [77.0, 322.0], [76.0, 322.0], [75.0, 322.0], [74.0, 322.0], [73.0, 322.0], [73.0, 323.0], [72.0, 323.0], [71.0, 323.0], [71.0, 324.0], [70.0, 324.0], [69.0, 324.0], [69.0, 325.0], [70.0, 325.0], [71.0, 325.0], [72.0, 325.0], [72.0, 326.0], [73.0, 326.0], [74.0, 326.0], [75.0, 326.0], [76.0, 326.0], [76.0, 327.0], [77.0, 327.0], [78.0, 327.0], [79.0, 327.0], [79.0, 328.0], [80.0, 328.0], [81.0, 328.0], [82.0, 328.0], [82.0, 329.0], [83.0, 329.0], [84.0, 329.0], [85.0, 329.0], [85.0, 330.0], [86.0, 330.0], [87.0, 330.0], [88.0, 330.0], [88.0, 331.0], [89.0, 331.0], [90.0, 331.0], [91.0, 331.0], [91.0, 332.0], [92.0, 332.0], [93.0, 332.0], [94.0, 332.0], [95.0, 332.0], [95.0, 333.0], [96.0, 333.0], [97.0, 333.0], [98.0, 333.0], [98.0, 334.0], [98.0, 335.0], [97.0, 335.0], [97.0, 336.0], [97.0, 337.0], [97.0, 338.0], [96.0, 338.0], [96.0, 339.0], [95.0, 339.0], [94.0, 339.0], [93.0, 339.0], [93.0, 338.0], [92.0, 338.0], [91.0, 338.0], [90.0, 338.0], [89.0, 338.0], [89.0, 337.0], [88.0, 337.0], [87.0, 337.0], [86.0, 337.0], [86.0, 336.0], [85.0, 336.0], [85.0, 337.0], [85.0, 338.0], [84.0, 338.0], [84.0, 339.0], [83.0, 339.0], [82.0, 339.0], [81.0, 339.0], [80.0, 339.0], [79.0, 339.0], [79.0, 338.0], [78.0, 338.0], [78.0, 337.0], [78.0, 336.0], [78.0, 335.0], [78.0, 334.0], [77.0, 334.0], [77.0, 333.0], [76.0, 333.0], [75.0, 333.0], [74.0, 333.0], [74.0, 332.0], [73.0, 332.0], [72.0, 332.0], [71.0, 332.0], [70.0, 332.0], [70.0, 331.0], [69.0, 331.0], [68.0, 331.0], [67.0, 331.0], [67.0, 330.0], [66.0, 330.0], [65.0, 330.0], [65.0, 329.0], [65.0, 328.0], [64.0, 328.0], [64.0, 329.0], [63.0, 329.0], [63.0, 330.0], [62.0, 330.0], [61.0, 330.0], [61.0, 331.0], [61.0, 332.0], [60.0, 332.0], [60.0, 333.0], [60.0, 334.0], [60.0, 335.0], [60.0, 336.0], [61.0, 336.0], [62.0, 336.0], [62.0, 337.0], [63.0, 337.0], [63.0, 338.0], [63.0, 339.0], [62.0, 339.0], [62.0, 340.0], [61.0, 340.0], [60.0, 340.0], [59.0, 340.0], [59.0, 339.0], [58.0, 339.0], [57.0, 339.0], [56.0, 339.0], [56.0, 338.0], [55.0, 338.0], [54.0, 338.0], [54.0, 339.0], [53.0, 339.0], [53.0, 340.0], [53.0, 341.0], [52.0, 341.0], [52.0, 342.0], [51.0, 342.0], [50.0, 342.0], [50.0, 343.0], [49.0, 343.0], [48.0, 343.0], [47.0, 343.0], [47.0, 342.0], [47.0, 341.0], [47.0, 340.0], [47.0, 339.0], [47.0, 338.0], [46.0, 338.0], [45.0, 338.0], [44.0, 338.0], [43.0, 338.0], [42.0, 338.0], [41.0, 338.0], [40.0, 338.0], [39.0, 338.0], [38.0, 338.0], [37.0, 338.0], [36.0, 338.0], [35.0, 338.0], [35.0, 339.0], [34.0, 339.0], [34.0, 340.0], [33.0, 340.0], [32.0, 340.0], [31.0, 340.0], [31.0, 339.0], [31.0, 338.0], [30.0, 338.0], [29.0, 338.0], [28.0, 338.0], [27.0, 338.0], [26.0, 338.0], [25.0, 338.0], [24.0, 338.0], [23.0, 338.0], [22.0, 338.0], [21.0, 338.0], [21.0, 339.0], [21.0, 340.0], [20.0, 340.0], [19.0, 340.0], [19.0, 341.0], [19.0, 342.0], [18.0, 342.0], [17.0, 342.0], [16.0, 342.0], [16.0, 341.0], [15.0, 341.0], [15.0, 340.0], [15.0, 339.0], [15.0, 338.0], [14.0, 338.0], [13.0, 338.0], [12.0, 338.0], [11.0, 338.0], [10.0, 338.0], [9.0, 338.0], [8.0, 338.0], [7.0, 338.0], [6.0, 338.0], [5.0, 338.0], [4.0, 338.0], [4.0, 339.0], [5.0, 339.0], [5.0, 340.0], [5.0, 341.0], [5.0, 342.0], [4.0, 342.0], [4.0, 343.0], [4.0, 344.0], [4.0, 345.0], [4.0, 346.0], [5.0, 346.0], [5.0, 347.0], [5.0, 348.0], [5.0, 349.0], [5.0, 350.0], [5.0, 351.0], [6.0, 351.0], [7.0, 351.0], [8.0, 351.0], [9.0, 351.0], [10.0, 351.0], [10.0, 350.0], [11.0, 350.0], [11.0, 351.0], [12.0, 351.0], [13.0, 351.0], [14.0, 351.0], [15.0, 351.0], [15.0, 350.0], [16.0, 350.0], [16.0, 349.0], [15.0, 349.0], [15.0, 348.0], [15.0, 347.0], [14.0, 347.0], [14.0, 346.0], [14.0, 345.0], [15.0, 345.0], [16.0, 345.0], [17.0, 345.0], [17.0, 346.0], [17.0, 347.0], [18.0, 347.0], [19.0, 347.0], [19.0, 348.0], [19.0, 349.0], [19.0, 350.0], [19.0, 351.0], [20.0, 351.0], [21.0, 351.0], [21.0, 350.0], [21.0, 349.0], [21.0, 348.0], [22.0, 348.0], [22.0, 347.0], [23.0, 347.0], [24.0, 347.0], [25.0, 347.0], [26.0, 347.0], [27.0, 347.0], [28.0, 347.0], [29.0, 347.0], [30.0, 347.0], [30.0, 348.0], [30.0, 349.0], [30.0, 350.0], [30.0, 351.0], [31.0, 351.0], [31.0, 350.0], [32.0, 350.0], [32.0, 349.0], [33.0, 349.0], [33.0, 350.0], [34.0, 350.0], [35.0, 350.0], [35.0, 351.0], [36.0, 351.0], [37.0, 351.0], [38.0, 351.0], [39.0, 351.0], [40.0, 351.0], [41.0, 351.0], [42.0, 351.0], [43.0, 351.0], [44.0, 351.0], [45.0, 351.0], [46.0, 351.0], [47.0, 351.0], [47.0, 350.0], [47.0, 349.0], [47.0, 348.0], [47.0, 347.0], [48.0, 347.0], [49.0, 347.0], [50.0, 347.0], [50.0, 348.0], [50.0, 349.0], [51.0, 349.0], [52.0, 349.0], [52.0, 350.0], [53.0, 350.0], [53.0, 351.0], [54.0, 351.0], [55.0, 351.0], [56.0, 351.0], [56.0, 350.0], [57.0, 350.0], [58.0, 350.0], [59.0, 350.0], [59.0, 351.0], [60.0, 351.0], [60.0, 352.0], [61.0, 352.0], [61.0, 351.0], [62.0, 351.0], [62.0, 350.0], [63.0, 350.0], [64.0, 350.0], [64.0, 349.0], [65.0, 349.0], [66.0, 349.0], [67.0, 349.0], [67.0, 350.0], [67.0, 351.0], [67.0, 352.0], [67.0, 353.0], [66.0, 353.0], [66.0, 354.0], [65.0, 354.0], [65.0, 355.0], [66.0, 355.0], [67.0, 355.0], [67.0, 356.0], [68.0, 356.0], [69.0, 356.0], [70.0, 356.0], [70.0, 357.0], [71.0, 357.0], [71.0, 358.0], [71.0, 359.0], [71.0, 360.0], [70.0, 360.0], [69.0, 360.0], [69.0, 361.0], [68.0, 361.0], [68.0, 362.0], [69.0, 362.0], [69.0, 363.0], [69.0, 364.0], [70.0, 364.0], [70.0, 363.0], [71.0, 363.0], [71.0, 362.0], [72.0, 362.0], [72.0, 361.0], [73.0, 361.0], [73.0, 360.0], [74.0, 360.0], [74.0, 359.0], [75.0, 359.0], [76.0, 359.0], [77.0, 359.0], [77.0, 360.0], [78.0, 360.0], [78.0, 361.0], [78.0, 362.0], [78.0, 363.0], [79.0, 363.0], [80.0, 363.0], [80.0, 364.0], [81.0, 364.0], [81.0, 365.0], [81.0, 366.0], [82.0, 366.0], [83.0, 366.0], [83.0, 365.0], [83.0, 364.0], [82.0, 364.0], [82.0, 363.0], [82.0, 362.0], [82.0, 361.0], [82.0, 360.0], [83.0, 360.0], [83.0, 359.0], [84.0, 359.0], [85.0, 359.0], [85.0, 358.0], [86.0, 358.0], [87.0, 358.0], [87.0, 359.0], [88.0, 359.0], [89.0, 359.0], [89.0, 360.0], [89.0, 361.0], [89.0, 362.0], [90.0, 362.0], [90.0, 361.0], [91.0, 361.0], [92.0, 361.0], [93.0, 361.0], [94.0, 361.0], [95.0, 361.0], [96.0, 361.0], [97.0, 361.0], [98.0, 361.0], [99.0, 361.0], [100.0, 361.0], [101.0, 361.0], [101.0, 362.0], [102.0, 362.0], [103.0, 362.0], [104.0, 362.0], [104.0, 363.0], [105.0, 363.0], [105.0, 362.0], [105.0, 361.0], [106.0, 361.0], [106.0, 360.0], [106.0, 359.0], [107.0, 359.0], [107.0, 358.0], [107.0, 357.0], [108.0, 357.0], [109.0, 357.0], [109.0, 356.0], [110.0, 356.0], [111.0, 356.0], [112.0, 356.0], [113.0, 356.0], [114.0, 356.0], [115.0, 356.0], [115.0, 357.0], [116.0, 357.0], [117.0, 357.0], [118.0, 357.0], [118.0, 358.0], [118.0, 359.0], [118.0, 360.0], [119.0, 360.0], [119.0, 361.0], [119.0, 362.0], [120.0, 362.0], [120.0, 363.0], [120.0, 364.0], [121.0, 364.0], [122.0, 364.0], [123.0, 364.0], [124.0, 364.0], [125.0, 364.0], [125.0, 363.0], [125.0, 362.0], [126.0, 362.0], [126.0, 361.0], [125.0, 361.0], [125.0, 360.0], [125.0, 359.0], [124.0, 359.0], [123.0, 359.0], [123.0, 358.0], [122.0, 358.0], [121.0, 358.0], [121.0, 357.0], [120.0, 357.0], [119.0, 357.0], [119.0, 356.0], [118.0, 356.0], [117.0, 356.0], [117.0, 355.0], [116.0, 355.0], [115.0, 355.0], [115.0, 354.0], [114.0, 354.0], [113.0, 354.0], [113.0, 353.0], [112.0, 353.0], [111.0, 353.0], [111.0, 352.0], [110.0, 352.0], [109.0, 352.0], [109.0, 351.0], [109.0, 350.0], [110.0, 350.0], [110.0, 349.0], [110.0, 348.0], [111.0, 348.0], [111.0, 347.0], [111.0, 346.0], [112.0, 346.0], [113.0, 346.0], [114.0, 346.0], [114.0, 347.0], [115.0, 347.0], [116.0, 347.0], [116.0, 348.0], [117.0, 348.0], [118.0, 348.0], [118.0, 349.0], [119.0, 349.0], [119.0, 350.0], [120.0, 350.0], [121.0, 350.0], [121.0, 351.0], [122.0, 351.0], [123.0, 351.0], [123.0, 352.0], [124.0, 352.0], [125.0, 352.0], [125.0, 351.0], [126.0, 351.0], [126.0, 350.0], [127.0, 350.0], [127.0, 349.0], [128.0, 349.0], [129.0, 349.0], [130.0, 349.0], [131.0, 349.0], [131.0, 350.0], [132.0, 350.0], [132.0, 351.0], [132.0, 352.0], [132.0, 353.0], [133.0, 353.0], [133.0, 354.0], [132.0, 354.0], [132.0, 355.0], [131.0, 355.0], [131.0, 356.0], [132.0, 356.0], [133.0, 356.0], [133.0, 357.0], [134.0, 357.0], [135.0, 357.0], [135.0, 358.0], [136.0, 358.0], [137.0, 358.0], [137.0, 359.0], [138.0, 359.0], [139.0, 359.0], [139.0, 360.0], [140.0, 360.0], [141.0, 360.0], [141.0, 361.0], [141.0, 362.0], [142.0, 362.0], [143.0, 362.0], [144.0, 362.0], [144.0, 361.0], [145.0, 361.0], [146.0, 361.0], [146.0, 360.0], [147.0, 360.0], [147.0, 359.0], [148.0, 359.0], [148.0, 358.0], [149.0, 358.0], [149.0, 357.0], [149.0, 356.0], [149.0, 355.0], [150.0, 355.0], [150.0, 354.0], [149.0, 354.0], [148.0, 354.0], [147.0, 354.0], [147.0, 353.0], [147.0, 352.0], [147.0, 351.0], [148.0, 351.0], [149.0, 351.0], [150.0, 351.0], [150.0, 350.0], [151.0, 350.0], [152.0, 350.0], [153.0, 350.0], [153.0, 351.0], [154.0, 351.0], [155.0, 351.0], [155.0, 350.0], [156.0, 350.0], [157.0, 350.0], [157.0, 349.0], [158.0, 349.0], [159.0, 349.0], [159.0, 348.0], [159.0, 347.0], [160.0, 347.0], [161.0, 347.0], [162.0, 347.0], [163.0, 347.0], [163.0, 348.0], [163.0, 349.0], [163.0, 350.0], [163.0, 351.0], [163.0, 352.0], [164.0, 352.0], [165.0, 352.0], [166.0, 352.0], [167.0, 352.0], [168.0, 352.0], [169.0, 352.0], [170.0, 352.0], [170.0, 351.0], [170.0, 350.0], [171.0, 350.0], [172.0, 350.0], [173.0, 350.0], [174.0, 350.0], [175.0, 350.0], [176.0, 350.0], [177.0, 350.0], [177.0, 351.0], [178.0, 351.0], [179.0, 351.0], [179.0, 352.0], [180.0, 352.0], [181.0, 352.0], [182.0, 352.0], [183.0, 352.0], [184.0, 352.0], [185.0, 352.0], [186.0, 352.0], [187.0, 352.0], [188.0, 352.0], [189.0, 352.0], [190.0, 352.0], [191.0, 352.0], [191.0, 351.0], [192.0, 351.0], [192.0, 350.0], [193.0, 350.0], [193.0, 351.0], [194.0, 351.0], [195.0, 351.0], [195.0, 352.0], [196.0, 352.0], [197.0, 352.0], [198.0, 352.0], [199.0, 352.0], [200.0, 352.0], [201.0, 352.0], [202.0, 352.0], [203.0, 352.0], [204.0, 352.0], [205.0, 352.0], [206.0, 352.0], [207.0, 352.0], [207.0, 351.0], [208.0, 351.0], [208.0, 350.0], [209.0, 350.0], [209.0, 351.0], [210.0, 351.0], [211.0, 351.0], [211.0, 352.0], [212.0, 352.0], [213.0, 352.0], [214.0, 352.0], [215.0, 352.0], [216.0, 352.0], [217.0, 352.0], [218.0, 352.0], [219.0, 352.0], [220.0, 352.0], [221.0, 352.0], [222.0, 352.0], [223.0, 352.0], [223.0, 351.0], [224.0, 351.0], [224.0, 350.0], [225.0, 350.0], [225.0, 351.0], [226.0, 351.0], [227.0, 351.0], [227.0, 352.0], [228.0, 352.0], [229.0, 352.0], [230.0, 352.0], [231.0, 352.0], [232.0, 352.0], [233.0, 352.0], [234.0, 352.0], [235.0, 352.0], [236.0, 352.0], [237.0, 352.0], [238.0, 352.0], [239.0, 352.0], [239.0, 351.0], [240.0, 351.0], [240.0, 350.0], [241.0, 350.0], [241.0, 351.0], [242.0, 351.0], [243.0, 351.0], [243.0, 352.0], [244.0, 352.0], [245.0, 352.0], [246.0, 352.0], [247.0, 352.0], [248.0, 352.0], [249.0, 352.0], [250.0, 352.0], [251.0, 352.0], [252.0, 352.0], [253.0, 352.0], [254.0, 352.0], [255.0, 352.0], [256.0, 352.0], [256.0, 351.0], [257.0, 351.0], [257.0, 350.0], [258.0, 350.0], [258.0, 351.0], [259.0, 351.0], [260.0, 351.0], [260.0, 350.0], [261.0, 350.0], [262.0, 350.0], [263.0, 350.0], [263.0, 351.0], [264.0, 351.0], [264.0, 352.0], [265.0, 352.0], [266.0, 352.0], [267.0, 352.0], [268.0, 352.0], [269.0, 352.0], [270.0, 352.0], [270.0, 351.0], [271.0, 351.0], [271.0, 350.0], [271.0, 349.0], [271.0, 348.0], [271.0, 347.0], [272.0, 347.0], [273.0, 347.0], [274.0, 347.0], [274.0, 346.0], [274.0, 345.0], [274.0, 344.0], [274.0, 343.0], [273.0, 343.0], [273.0, 342.0], [272.0, 342.0], [271.0, 342.0], [271.0, 341.0], [271.0, 340.0], [271.0, 339.0], [270.0, 339.0], [269.0, 339.0], [268.0, 339.0], [267.0, 339.0], [266.0, 339.0], [265.0, 339.0], [264.0, 339.0], [264.0, 338.0], [263.0, 338.0], [263.0, 337.0], [263.0, 336.0], [262.0, 336.0], [262.0, 335.0], [262.0, 334.0], [263.0, 334.0], [264.0, 334.0], [265.0, 334.0], [265.0, 335.0], [265.0, 336.0], [266.0, 336.0], [266.0, 335.0], [266.0, 334.0], [267.0, 334.0], [268.0, 334.0], [268.0, 335.0], [269.0, 335.0], [269.0, 334.0], [269.0, 333.0], [269.0, 332.0], [268.0, 332.0], [267.0, 332.0], [267.0, 333.0], [266.0, 333.0], [265.0, 333.0], [264.0, 333.0], [263.0, 333.0], [262.0, 333.0], [261.0, 333.0], [261.0, 332.0], [260.0, 332.0], [260.0, 331.0], [259.0, 331.0], [259.0, 330.0], [259.0, 329.0], [259.0, 328.0], [258.0, 328.0], [258.0, 327.0], [258.0, 326.0], [258.0, 325.0], [258.0, 324.0], [257.0, 324.0], [257.0, 325.0], [257.0, 326.0], [257.0, 327.0], [257.0, 328.0], [257.0, 329.0], [257.0, 330.0], [257.0, 331.0], [257.0, 332.0], [257.0, 333.0], [257.0, 334.0], [257.0, 335.0], [258.0, 335.0], [259.0, 335.0], [259.0, 336.0], [259.0, 337.0], [259.0, 338.0], [258.0, 338.0], [258.0, 339.0], [257.0, 339.0], [256.0, 339.0], [255.0, 339.0], [255.0, 338.0], [254.0, 338.0], [253.0, 338.0], [252.0, 338.0], [251.0, 338.0], [250.0, 338.0], [249.0, 338.0], [248.0, 338.0], [247.0, 338.0], [246.0, 338.0], [245.0, 338.0], [244.0, 338.0], [243.0, 338.0], [242.0, 338.0], [242.0, 339.0], [241.0, 339.0], [240.0, 339.0], [239.0, 339.0], [239.0, 340.0], [239.0, 341.0], [238.0, 341.0], [238.0, 342.0], [237.0, 342.0], [236.0, 342.0], [235.0, 342.0], [234.0, 342.0], [233.0, 342.0], [232.0, 342.0], [232.0, 341.0], [231.0, 341.0], [231.0, 340.0], [231.0, 339.0], [230.0, 339.0], [230.0, 338.0], [229.0, 338.0], [228.0, 338.0], [227.0, 338.0], [226.0, 338.0], [226.0, 339.0], [225.0, 339.0], [224.0, 339.0], [223.0, 339.0], [223.0, 338.0], [222.0, 338.0], [221.0, 338.0], [220.0, 338.0], [219.0, 338.0], [218.0, 338.0], [217.0, 338.0], [216.0, 338.0], [215.0, 338.0], [214.0, 338.0], [213.0, 338.0], [212.0, 338.0], [211.0, 338.0], [210.0, 338.0], [210.0, 339.0], [209.0, 339.0], [208.0, 339.0], [207.0, 339.0], [207.0, 338.0], [206.0, 338.0], [205.0, 338.0], [204.0, 338.0], [203.0, 338.0], [202.0, 338.0], [201.0, 338.0], [200.0, 338.0], [199.0, 338.0], [198.0, 338.0], [197.0, 338.0], [196.0, 338.0], [195.0, 338.0], [194.0, 338.0], [194.0, 339.0], [193.0, 339.0], [192.0, 339.0], [191.0, 339.0], [191.0, 338.0], [190.0, 338.0], [189.0, 338.0], [188.0, 338.0], [187.0, 338.0], [186.0, 338.0], [185.0, 338.0], [184.0, 338.0], [183.0, 338.0], [182.0, 338.0], [181.0, 338.0], [180.0, 338.0], [179.0, 338.0], [178.0, 338.0], [178.0, 339.0], [178.0, 340.0], [178.0, 341.0], [177.0, 341.0], [176.0, 341.0], [175.0, 341.0], [174.0, 341.0], [174.0, 340.0], [174.0, 339.0], [174.0, 338.0], [175.0, 338.0], [175.0, 337.0], [174.0, 337.0], [173.0, 337.0], [172.0, 337.0], [171.0, 337.0], [170.0, 337.0], [169.0, 337.0], [168.0, 337.0], [167.0, 337.0], [166.0, 337.0], [165.0, 337.0], [164.0, 337.0], [163.0, 337.0], [163.0, 338.0], [163.0, 339.0], [163.0, 340.0], [163.0, 341.0], [163.0, 342.0], [163.0, 343.0], [162.0, 343.0], [161.0, 343.0], [160.0, 343.0], [160.0, 342.0], [159.0, 342.0], [158.0, 342.0], [157.0, 342.0], [157.0, 341.0], [156.0, 341.0], [156.0, 340.0], [156.0, 339.0], [155.0, 339.0], [155.0, 338.0], [154.0, 338.0], [153.0, 338.0], [153.0, 339.0], [152.0, 339.0], [151.0, 339.0], [150.0, 339.0], [150.0, 338.0], [149.0, 338.0], [149.0, 337.0], [149.0, 336.0], [148.0, 336.0], [148.0, 337.0], [147.0, 337.0], [146.0, 337.0], [146.0, 338.0], [145.0, 338.0], [144.0, 338.0], [144.0, 339.0], [143.0, 339.0], [143.0, 338.0], [143.0, 337.0], [144.0, 337.0], [145.0, 337.0], [145.0, 336.0], [146.0, 336.0], [146.0, 335.0], [147.0, 335.0], [148.0, 335.0], [148.0, 334.0], [148.0, 333.0], [149.0, 333.0], [149.0, 332.0], [149.0, 331.0], [148.0, 331.0], [148.0, 330.0], [147.0, 330.0], [147.0, 329.0], [146.0, 329.0], [146.0, 328.0], [145.0, 328.0], [145.0, 327.0], [144.0, 327.0], [143.0, 327.0], [143.0, 328.0], [142.0, 328.0], [142.0, 327.0], [141.0, 327.0], [141.0, 328.0], [140.0, 328.0], [140.0, 329.0], [139.0, 329.0], [138.0, 329.0], [138.0, 330.0], [137.0, 330.0], [136.0, 330.0], [136.0, 331.0], [135.0, 331.0], [134.0, 331.0], [134.0, 332.0], [133.0, 332.0], [133.0, 333.0], [133.0, 334.0], [132.0, 334.0], [132.0, 335.0], [132.0, 336.0], [132.0, 337.0], [132.0, 338.0], [131.0, 338.0], [131.0, 339.0], [130.0, 339.0], [130.0, 340.0], [129.0, 340.0], [128.0, 340.0], [127.0, 340.0], [126.0, 340.0], [126.0, 339.0], [125.0, 339.0], [125.0, 338.0], [125.0, 337.0], [124.0, 337.0], [124.0, 338.0], [123.0, 338.0], [122.0, 338.0], [122.0, 339.0], [121.0, 339.0], [120.0, 339.0], [120.0, 340.0], [119.0, 340.0], [118.0, 340.0], [118.0, 341.0], [117.0, 341.0], [116.0, 341.0], [116.0, 340.0], [115.0, 340.0], [115.0, 339.0], [115.0, 338.0], [115.0, 337.0], [114.0, 337.0], [114.0, 336.0], [114.0, 335.0], [115.0, 335.0], [116.0, 335.0], [116.0, 334.0], [117.0, 334.0], [118.0, 334.0], [118.0, 333.0], [119.0, 333.0], [119.0, 332.0], [120.0, 332.0], [121.0, 332.0], [121.0, 331.0], [122.0, 331.0], [123.0, 331.0], [123.0, 330.0], [124.0, 330.0], [125.0, 330.0], [125.0, 329.0], [126.0, 329.0], [127.0, 329.0], [127.0, 328.0], [128.0, 328.0], [128.0, 327.0], [129.0, 327.0], [130.0, 327.0], [130.0, 326.0], [131.0, 326.0], [132.0, 326.0], [132.0, 325.0], [133.0, 325.0], [134.0, 325.0], [134.0, 324.0], [135.0, 324.0], [135.0, 323.0], [136.0, 323.0], [137.0, 323.0], [137.0, 322.0], [136.0, 322.0], [135.0, 322.0], [135.0, 321.0], [134.0, 321.0], [133.0, 321.0], [132.0, 321.0], [131.0, 321.0], [131.0, 322.0], [130.0, 322.0], [130.0, 323.0], [129.0, 323.0], [128.0, 323.0], [128.0, 324.0], [128.0, 325.0], [127.0, 325.0], [127.0, 326.0], [126.0, 326.0], [126.0, 325.0], [125.0, 325.0], [125.0, 324.0], [125.0, 323.0], [125.0, 322.0], [126.0, 322.0], [126.0, 321.0], [126.0, 320.0], [126.0, 319.0], [127.0, 319.0], [127.0, 318.0], [128.0, 318.0], [128.0, 317.0], [128.0, 316.0], [127.0, 316.0], [127.0, 317.0], [126.0, 317.0], [125.0, 317.0], [124.0, 317.0], [124.0, 316.0], [123.0, 316.0], [123.0, 315.0], [122.0, 315.0], [122.0, 314.0], [121.0, 314.0], [121.0, 313.0], [122.0, 313.0], [122.0, 312.0], [122.0, 311.0], [123.0, 311.0], [123.0, 310.0], [124.0, 310.0], [125.0, 310.0], [125.0, 309.0], [124.0, 309.0], [123.0, 309.0], [122.0, 309.0], [122.0, 308.0], [122.0, 307.0], [122.0, 306.0], [122.0, 305.0], [121.0, 305.0], [121.0, 304.0], [120.0, 304.0], [120.0, 303.0], [119.0, 303.0], [119.0, 304.0], [118.0, 304.0], [118.0, 303.0], [117.0, 303.0], [117.0, 302.0], [117.0, 301.0], [117.0, 300.0], [116.0, 300.0], [115.0, 300.0], [114.0, 300.0], [113.0, 300.0], [112.0, 300.0], [111.0, 300.0], [110.0, 300.0], [110.0, 299.0], [110.0, 298.0], [110.0, 297.0], [110.0, 296.0], [111.0, 296.0], [111.0, 295.0], [111.0, 294.0], [110.0, 294.0], [109.0, 294.0], [109.0, 293.0], [109.0, 292.0], [108.0, 292.0], [108.0, 291.0], [109.0, 291.0], [109.0, 290.0], [108.0, 290.0], [107.0, 290.0], [107.0, 289.0], [107.0, 288.0], [107.0, 287.0], [107.0, 286.0], [108.0, 286.0], [109.0, 286.0], [110.0, 286.0], [111.0, 286.0], [112.0, 286.0], [112.0, 285.0], [112.0, 284.0], [112.0, 283.0], [112.0, 282.0], [112.0, 281.0], [112.0, 280.0], [112.0, 279.0], [112.0, 278.0], [112.0, 277.0], [112.0, 276.0], [112.0, 275.0], [111.0, 275.0], [111.0, 274.0], [111.0, 273.0], [110.0, 273.0], [109.0, 273.0], [109.0, 272.0], [108.0, 272.0], [108.0, 271.0], [108.0, 270.0], [108.0, 269.0], [109.0, 269.0], [109.0, 268.0], [110.0, 268.0], [111.0, 268.0], [112.0, 268.0], [112.0, 267.0], [112.0, 266.0], [112.0, 265.0], [112.0, 264.0], [112.0, 263.0], [112.0, 262.0], [112.0, 261.0], [112.0, 260.0], [112.0, 259.0], [111.0, 259.0], [111.0, 258.0], [111.0, 257.0], [111.0, 256.0], [111.0, 255.0], [110.0, 255.0], [110.0, 254.0], [109.0, 254.0], [109.0, 253.0], [109.0, 252.0], [110.0, 252.0], [110.0, 251.0], [111.0, 251.0], [112.0, 251.0], [112.0, 250.0], [112.0, 249.0], [112.0, 248.0], [112.0, 247.0], [112.0, 246.0], [112.0, 245.0], [112.0, 244.0], [111.0, 244.0], [111.0, 243.0], [110.0, 243.0], [109.0, 243.0], [108.0, 243.0], [107.0, 243.0], [107.0, 242.0], [107.0, 241.0], [107.0, 240.0], [107.0, 239.0], [108.0, 239.0], [108.0, 238.0], [109.0, 238.0], [110.0, 238.0], [111.0, 238.0], [111.0, 239.0], [112.0, 239.0], [112.0, 238.0], [113.0, 238.0], [114.0, 238.0], [114.0, 239.0], [115.0, 239.0], [115.0, 240.0], [116.0, 240.0], [116.0, 239.0], [116.0, 238.0], [116.0, 237.0], [116.0, 236.0], [117.0, 236.0], [117.0, 235.0], [118.0, 235.0], [119.0, 235.0], [120.0, 235.0], [121.0, 235.0], [122.0, 235.0], [122.0, 236.0], [123.0, 236.0], [123.0, 237.0], [123.0, 238.0], [123.0, 239.0], [123.0, 240.0], [124.0, 240.0], [125.0, 240.0], [126.0, 240.0], [127.0, 240.0], [127.0, 239.0], [128.0, 239.0], [128.0, 238.0], [129.0, 238.0], [130.0, 238.0], [130.0, 239.0], [131.0, 239.0], [131.0, 240.0], [132.0, 240.0], [133.0, 240.0], [133.0, 239.0], [132.0, 239.0], [132.0, 238.0], [131.0, 238.0], [131.0, 237.0], [131.0, 236.0], [131.0, 235.0], [132.0, 235.0], [133.0, 235.0], [134.0, 235.0], [135.0, 235.0], [135.0, 236.0], [136.0, 236.0], [136.0, 237.0], [137.0, 237.0], [137.0, 238.0], [137.0, 239.0], [136.0, 239.0], [136.0, 240.0], [137.0, 240.0], [138.0, 240.0], [139.0, 240.0], [140.0, 240.0], [141.0, 240.0], [142.0, 240.0], [143.0, 240.0], [143.0, 239.0], [144.0, 239.0], [144.0, 238.0], [144.0, 237.0], [144.0, 236.0], [144.0, 235.0], [144.0, 234.0], [144.0, 233.0], [144.0, 232.0], [144.0, 231.0], [143.0, 231.0], [143.0, 232.0], [142.0, 232.0], [141.0, 232.0], [140.0, 232.0], [140.0, 231.0], [139.0, 231.0], [139.0, 230.0], [139.0, 229.0], [139.0, 228.0], [139.0, 227.0], [139.0, 226.0], [140.0, 226.0], [141.0, 226.0], [141.0, 225.0], [140.0, 225.0], [140.0, 224.0], [141.0, 224.0], [141.0, 223.0], [142.0, 223.0], [143.0, 223.0], [143.0, 222.0], [143.0, 221.0], [143.0, 220.0], [143.0, 219.0], [144.0, 219.0], [145.0, 219.0], [146.0, 219.0], [146.0, 220.0], [146.0, 221.0], [146.0, 222.0], [146.0, 223.0], [147.0, 223.0], [147.0, 222.0], [148.0, 222.0], [149.0, 222.0], [150.0, 222.0], [151.0, 222.0], [151.0, 223.0], [151.0, 224.0], [152.0, 224.0], [153.0, 224.0], [154.0, 224.0], [155.0, 224.0], [156.0, 224.0], [157.0, 224.0], [158.0, 224.0], [159.0, 224.0], [160.0, 224.0], [160.0, 223.0], [161.0, 223.0], [161.0, 224.0], [162.0, 224.0], [163.0, 224.0], [163.0, 225.0], [163.0, 226.0], [163.0, 227.0], [162.0, 227.0], [162.0, 228.0], [161.0, 228.0], [161.0, 229.0], [162.0, 229.0], [163.0, 229.0], [163.0, 230.0], [163.0, 231.0], [163.0, 232.0], [164.0, 232.0], [164.0, 233.0], [164.0, 234.0], [164.0, 235.0], [163.0, 235.0], [162.0, 235.0], [161.0, 235.0], [161.0, 236.0], [161.0, 237.0], [162.0, 237.0], [162.0, 238.0], [163.0, 238.0], [163.0, 239.0], [164.0, 239.0], [165.0, 239.0], [166.0, 239.0], [167.0, 239.0], [167.0, 240.0], [167.0, 241.0], [167.0, 242.0], [166.0, 242.0], [166.0, 243.0], [165.0, 243.0], [164.0, 243.0], [164.0, 244.0], [165.0, 244.0], [166.0, 244.0], [167.0, 244.0], [167.0, 245.0], [167.0, 246.0], [167.0, 247.0], [167.0, 248.0], [167.0, 249.0], [166.0, 249.0], [165.0, 249.0], [164.0, 249.0], [163.0, 249.0], [163.0, 250.0], [162.0, 250.0], [161.0, 250.0], [161.0, 251.0], [161.0, 252.0], [161.0, 253.0], [161.0, 254.0], [161.0, 255.0], [162.0, 255.0], [163.0, 255.0], [163.0, 256.0], [163.0, 257.0], [163.0, 258.0], [163.0, 259.0], [163.0, 260.0], [163.0, 261.0], [163.0, 262.0], [163.0, 263.0], [163.0, 264.0], [163.0, 265.0], [163.0, 266.0], [163.0, 267.0], [164.0, 267.0], [164.0, 268.0], [165.0, 268.0], [166.0, 268.0], [167.0, 268.0], [167.0, 269.0], [168.0, 269.0], [169.0, 269.0], [169.0, 270.0], [169.0, 271.0], [170.0, 271.0], [171.0, 271.0], [171.0, 270.0], [172.0, 270.0], [172.0, 269.0], [172.0, 268.0], [173.0, 268.0], [173.0, 269.0], [173.0, 270.0], [174.0, 270.0], [174.0, 269.0], [175.0, 269.0], [176.0, 269.0], [176.0, 268.0], [176.0, 267.0], [177.0, 267.0], [178.0, 267.0], [178.0, 268.0], [178.0, 269.0], [178.0, 270.0], [179.0, 270.0], [179.0, 271.0], [179.0, 272.0], [180.0, 272.0], [181.0, 272.0], [181.0, 273.0], [182.0, 273.0], [182.0, 272.0], [182.0, 271.0], [183.0, 271.0], [184.0, 271.0], [184.0, 270.0], [184.0, 269.0], [184.0, 268.0], [185.0, 268.0], [186.0, 268.0], [186.0, 267.0], [187.0, 267.0], [188.0, 267.0], [189.0, 267.0], [190.0, 267.0], [191.0, 267.0], [191.0, 268.0], [191.0, 269.0], [192.0, 269.0], [192.0, 268.0], [192.0, 267.0], [192.0, 266.0], [192.0, 265.0], [192.0, 264.0], [192.0, 263.0], [192.0, 262.0], [192.0, 261.0], [192.0, 260.0], [191.0, 260.0], [191.0, 261.0], [190.0, 261.0], [189.0, 261.0], [188.0, 261.0], [187.0, 261.0], [187.0, 260.0], [187.0, 259.0], [187.0, 258.0], [187.0, 257.0], [187.0, 256.0], [187.0, 255.0], [187.0, 254.0], [187.0, 253.0], [188.0, 253.0], [189.0, 253.0], [190.0, 253.0], [191.0, 253.0], [191.0, 252.0], [190.0, 252.0], [190.0, 251.0], [191.0, 251.0], [192.0, 251.0], [192.0, 250.0], [192.0, 249.0], [192.0, 248.0], [192.0, 247.0], [192.0, 246.0], [192.0, 245.0], [192.0, 244.0], [191.0, 244.0], [191.0, 243.0], [191.0, 242.0], [190.0, 242.0], [189.0, 242.0], [189.0, 243.0], [189.0, 244.0], [189.0, 245.0], [189.0, 246.0], [189.0, 247.0], [188.0, 247.0], [187.0, 247.0], [187.0, 246.0], [186.0, 246.0], [185.0, 246.0], [185.0, 245.0], [185.0, 244.0], [185.0, 243.0], [186.0, 243.0], [187.0, 243.0], [187.0, 242.0], [186.0, 242.0], [185.0, 242.0], [184.0, 242.0], [183.0, 242.0], [182.0, 242.0], [181.0, 242.0], [180.0, 242.0], [179.0, 242.0], [179.0, 243.0], [179.0, 244.0], [178.0, 244.0], [177.0, 244.0], [176.0, 244.0], [175.0, 244.0], [175.0, 243.0], [174.0, 243.0], [173.0, 243.0], [172.0, 243.0], [171.0, 243.0], [171.0, 242.0], [171.0, 241.0], [171.0, 240.0], [171.0, 239.0], [172.0, 239.0], [173.0, 239.0], [174.0, 239.0], [174.0, 238.0], [175.0, 238.0], [176.0, 238.0], [176.0, 237.0], [176.0, 236.0], [176.0, 235.0], [175.0, 235.0], [174.0, 235.0], [173.0, 235.0], [173.0, 234.0], [173.0, 233.0], [173.0, 232.0], [174.0, 232.0], [174.0, 231.0], [174.0, 230.0], [174.0, 229.0], [175.0, 229.0], [176.0, 229.0], [176.0, 228.0], [175.0, 228.0], [175.0, 227.0], [175.0, 226.0], [175.0, 225.0], [175.0, 224.0], [174.0, 224.0], [173.0, 224.0], [173.0, 223.0], [173.0, 222.0], [173.0, 221.0], [172.0, 221.0], [171.0, 221.0], [171.0, 220.0], [171.0, 219.0], [171.0, 218.0], [171.0, 217.0], [171.0, 216.0], [171.0, 215.0], [172.0, 215.0], [172.0, 214.0], [172.0, 213.0], [172.0, 212.0], [173.0, 212.0], [173.0, 211.0], [174.0, 211.0], [175.0, 211.0], [175.0, 210.0], [175.0, 209.0], [175.0, 208.0], [176.0, 208.0], [176.0, 207.0], [175.0, 207.0], [174.0, 207.0], [174.0, 206.0], [174.0, 205.0], [174.0, 204.0], [175.0, 204.0], [175.0, 203.0], [174.0, 203.0], [174.0, 202.0], [174.0, 201.0], [174.0, 200.0], [174.0, 199.0], [175.0, 199.0], [176.0, 199.0], [176.0, 198.0], [175.0, 198.0], [175.0, 197.0], [174.0, 197.0], [174.0, 196.0], [173.0, 196.0], [172.0, 196.0], [171.0, 196.0], [171.0, 195.0], [171.0, 194.0], [171.0, 193.0], [171.0, 192.0], [172.0, 192.0], [173.0, 192.0], [174.0, 192.0], [175.0, 192.0], [175.0, 191.0], [176.0, 191.0], [177.0, 191.0], [178.0, 191.0], [178.0, 192.0], [179.0, 192.0], [179.0, 193.0], [180.0, 193.0], [181.0, 193.0], [182.0, 193.0], [183.0, 193.0], [184.0, 193.0], [185.0, 193.0], [186.0, 193.0], [187.0, 193.0], [187.0, 192.0], [187.0, 191.0], [188.0, 191.0], [189.0, 191.0], [190.0, 191.0], [190.0, 190.0], [189.0, 190.0], [189.0, 189.0], [189.0, 188.0], [190.0, 188.0], [190.0, 187.0], [190.0, 186.0], [191.0, 186.0], [192.0, 186.0], [192.0, 185.0], [192.0, 184.0], [192.0, 183.0], [192.0, 182.0], [192.0, 181.0], [192.0, 180.0], [191.0, 180.0], [191.0, 179.0], [191.0, 178.0], [191.0, 177.0], [191.0, 176.0], [192.0, 176.0], [192.0, 175.0], [192.0, 174.0], [192.0, 173.0], [192.0, 172.0], [192.0, 171.0], [192.0, 170.0], [192.0, 169.0], [192.0, 168.0], [192.0, 167.0], [192.0, 166.0], [192.0, 165.0], [192.0, 164.0], [191.0, 164.0], [191.0, 163.0], [191.0, 162.0], [190.0, 162.0], [189.0, 162.0], [188.0, 162.0], [187.0, 162.0], [186.0, 162.0], [185.0, 162.0], [184.0, 162.0], [183.0, 162.0], [182.0, 162.0], [181.0, 162.0], [180.0, 162.0], [179.0, 162.0], [179.0, 163.0], [178.0, 163.0], [178.0, 164.0], [177.0, 164.0], [176.0, 164.0], [176.0, 165.0], [176.0, 166.0], [176.0, 167.0], [176.0, 168.0], [176.0, 169.0], [176.0, 170.0], [176.0, 171.0], [175.0, 171.0], [174.0, 171.0], [173.0, 171.0], [172.0, 171.0], [171.0, 171.0], [171.0, 170.0], [171.0, 169.0], [171.0, 168.0], [171.0, 167.0], [171.0, 166.0], [171.0, 165.0], [171.0, 164.0], [171.0, 163.0], [172.0, 163.0], [173.0, 163.0], [174.0, 163.0], [175.0, 163.0], [175.0, 162.0], [174.0, 162.0], [173.0, 162.0], [172.0, 162.0], [171.0, 162.0], [170.0, 162.0], [169.0, 162.0], [168.0, 162.0], [167.0, 162.0], [166.0, 162.0], [165.0, 162.0], [164.0, 162.0], [163.0, 162.0], [163.0, 163.0], [164.0, 163.0], [165.0, 163.0], [166.0, 163.0], [166.0, 164.0], [166.0, 165.0], [166.0, 166.0], [166.0, 167.0], [166.0, 168.0], [166.0, 169.0], [166.0, 170.0], [166.0, 171.0], [165.0, 171.0], [164.0, 171.0], [163.0, 171.0], [162.0, 171.0], [161.0, 171.0], [161.0, 172.0], [161.0, 173.0], [161.0, 174.0], [161.0, 175.0], [161.0, 176.0], [162.0, 176.0], [163.0, 176.0], [163.0, 177.0], [163.0, 178.0], [163.0, 179.0], [162.0, 179.0], [162.0, 180.0], [161.0, 180.0], [161.0, 181.0], [162.0, 181.0], [163.0, 181.0], [163.0, 180.0], [164.0, 180.0], [165.0, 180.0], [166.0, 180.0], [166.0, 181.0], [166.0, 182.0], [166.0, 183.0], [166.0, 184.0], [166.0, 185.0], [166.0, 186.0], [165.0, 186.0], [164.0, 186.0], [163.0, 186.0], [163.0, 187.0], [163.0, 188.0], [163.0, 189.0], [162.0, 189.0], [161.0, 189.0], [161.0, 190.0], [162.0, 190.0], [162.0, 191.0], [163.0, 191.0], [163.0, 192.0], [164.0, 192.0], [165.0, 192.0], [166.0, 192.0], [167.0, 192.0], [167.0, 193.0], [167.0, 194.0], [167.0, 195.0], [166.0, 195.0], [166.0, 196.0], [165.0, 196.0], [164.0, 196.0], [164.0, 197.0], [163.0, 197.0], [163.0, 198.0], [162.0, 198.0], [162.0, 199.0], [162.0, 200.0], [163.0, 200.0], [164.0, 200.0], [165.0, 200.0], [165.0, 201.0], [165.0, 202.0], [165.0, 203.0], [164.0, 203.0], [164.0, 204.0], [163.0, 204.0], [163.0, 205.0], [163.0, 206.0], [163.0, 207.0], [162.0, 207.0], [162.0, 208.0], [163.0, 208.0], [163.0, 209.0], [163.0, 210.0], [163.0, 211.0], [162.0, 211.0], [162.0, 212.0], [161.0, 212.0], [160.0, 212.0], [159.0, 212.0], [159.0, 211.0], [159.0, 210.0], [158.0, 210.0], [157.0, 210.0], [156.0, 210.0], [155.0, 210.0], [154.0, 210.0], [153.0, 210.0], [152.0, 210.0], [151.0, 210.0], [151.0, 211.0], [151.0, 212.0], [150.0, 212.0], [149.0, 212.0], [148.0, 212.0], [148.0, 211.0], [147.0, 211.0], [147.0, 210.0], [146.0, 210.0], [146.0, 211.0], [146.0, 212.0], [146.0, 213.0], [146.0, 214.0], [146.0, 215.0], [145.0, 215.0], [144.0, 215.0], [144.0, 214.0], [143.0, 214.0], [143.0, 215.0], [142.0, 215.0], [141.0, 215.0], [141.0, 214.0], [140.0, 214.0], [139.0, 214.0], [139.0, 213.0], [138.0, 213.0], [138.0, 212.0], [137.0, 212.0], [137.0, 211.0], [136.0, 211.0], [136.0, 210.0], [136.0, 209.0], [136.0, 208.0], [137.0, 208.0], [137.0, 207.0], [138.0, 207.0], [138.0, 206.0], [139.0, 206.0], [139.0, 205.0], [139.0, 204.0], [139.0, 203.0], [140.0, 203.0], [141.0, 203.0], [142.0, 203.0], [143.0, 203.0], [144.0, 203.0], [144.0, 202.0], [144.0, 201.0], [144.0, 200.0], [144.0, 199.0], [144.0, 198.0], [144.0, 197.0], [144.0, 196.0], [144.0, 195.0], [143.0, 195.0], [143.0, 194.0], [142.0, 194.0], [141.0, 194.0], [140.0, 194.0], [139.0, 194.0], [138.0, 194.0], [137.0, 194.0], [136.0, 194.0], [135.0, 194.0], [134.0, 194.0], [133.0, 194.0], [132.0, 194.0], [131.0, 194.0], [130.0, 194.0], [130.0, 195.0], [129.0, 195.0], [128.0, 195.0], [127.0, 195.0], [127.0, 194.0], [126.0, 194.0], [125.0, 194.0], [124.0, 194.0], [124.0, 193.0], [124.0, 192.0], [124.0, 191.0], [123.0, 191.0], [123.0, 192.0], [123.0, 193.0], [122.0, 193.0], [122.0, 194.0], [121.0, 194.0], [120.0, 194.0], [119.0, 194.0], [119.0, 193.0], [118.0, 193.0], [118.0, 192.0], [117.0, 192.0], [117.0, 193.0], [117.0, 194.0], [116.0, 194.0], [115.0, 194.0], [114.0, 194.0], [114.0, 195.0], [113.0, 195.0], [112.0, 195.0], [112.0, 196.0], [111.0, 196.0], [111.0, 197.0], [110.0, 197.0], [109.0, 197.0], [109.0, 196.0], [108.0, 196.0], [108.0, 195.0], [108.0, 194.0], [109.0, 194.0], [109.0, 193.0], [108.0, 193.0], [107.0, 193.0], [107.0, 192.0], [107.0, 191.0], [107.0, 190.0], [108.0, 190.0], [109.0, 190.0], [110.0, 190.0], [111.0, 190.0], [111.0, 189.0], [112.0, 189.0], [113.0, 189.0], [113.0, 188.0], [113.0, 187.0], [113.0, 186.0], [113.0, 185.0], [113.0, 184.0], [113.0, 183.0], [113.0, 182.0], [113.0, 181.0], [113.0, 180.0], [113.0, 179.0], [112.0, 179.0], [112.0, 178.0], [112.0, 177.0], [111.0, 177.0], [111.0, 176.0], [112.0, 176.0], [112.0, 175.0], [113.0, 175.0], [114.0, 175.0], [115.0, 175.0], [116.0, 175.0], [117.0, 175.0], [117.0, 174.0], [118.0, 174.0], [119.0, 174.0], [120.0, 174.0], [121.0, 174.0], [121.0, 175.0], [121.0, 176.0], [122.0, 176.0], [123.0, 176.0], [124.0, 176.0], [125.0, 176.0], [126.0, 176.0], [127.0, 176.0], [128.0, 176.0], [128.0, 175.0], [129.0, 175.0], [130.0, 175.0], [131.0, 175.0], [131.0, 174.0], [131.0, 173.0], [132.0, 173.0], [133.0, 173.0], [134.0, 173.0], [134.0, 172.0], [135.0, 172.0], [135.0, 171.0], [135.0, 170.0], [136.0, 170.0], [136.0, 169.0], [136.0, 168.0], [136.0, 167.0], [137.0, 167.0], [138.0, 167.0], [138.0, 166.0], [139.0, 166.0], [140.0, 166.0], [141.0, 166.0], [142.0, 166.0], [142.0, 167.0], [142.0, 168.0], [143.0, 168.0], [144.0, 168.0], [144.0, 167.0], [144.0, 166.0], [144.0, 165.0], [144.0, 164.0], [144.0, 163.0], [143.0, 163.0], [143.0, 162.0], [143.0, 161.0], [142.0, 161.0], [142.0, 160.0], [143.0, 160.0], [143.0, 159.0], [144.0, 159.0], [144.0, 158.0], [144.0, 157.0], [144.0, 156.0], [144.0, 155.0], [144.0, 154.0], [144.0, 153.0], [144.0, 152.0], [144.0, 151.0], [144.0, 150.0], [144.0, 149.0], [144.0, 148.0], [144.0, 147.0], [143.0, 147.0], [142.0, 147.0], [141.0, 147.0], [141.0, 146.0], [140.0, 146.0], [140.0, 145.0], [139.0, 145.0], [139.0, 144.0], [139.0, 143.0], [139.0, 142.0], [139.0, 141.0], [139.0, 140.0], [140.0, 140.0], [140.0, 139.0], [141.0, 139.0], [141.0, 138.0], [142.0, 138.0], [143.0, 138.0], [144.0, 138.0], [144.0, 137.0], [144.0, 136.0], [144.0, 135.0], [144.0, 134.0], [144.0, 133.0], [144.0, 132.0], [144.0, 131.0], [143.0, 131.0], [143.0, 130.0], [143.0, 129.0], [142.0, 129.0], [142.0, 128.0], [143.0, 128.0], [143.0, 127.0], [143.0, 126.0], [143.0, 125.0], [144.0, 125.0], [144.0, 124.0], [145.0, 124.0], [146.0, 124.0], [147.0, 124.0], [148.0, 124.0], [149.0, 124.0], [150.0, 124.0], [150.0, 125.0], [151.0, 125.0], [151.0, 126.0], [151.0, 127.0], [152.0, 127.0], [152.0, 128.0], [153.0, 128.0], [154.0, 128.0], [155.0, 128.0], [156.0, 128.0], [157.0, 128.0], [158.0, 128.0], [159.0, 128.0], [159.0, 127.0], [160.0, 127.0], [161.0, 127.0], [162.0, 127.0], [163.0, 127.0], [163.0, 128.0], [164.0, 128.0], [165.0, 128.0], [166.0, 128.0], [167.0, 128.0], [168.0, 128.0], [169.0, 128.0], [170.0, 128.0], [171.0, 128.0], [172.0, 128.0], [173.0, 128.0], [174.0, 128.0], [175.0, 128.0], [175.0, 127.0], [176.0, 127.0], [176.0, 126.0], [177.0, 126.0], [177.0, 127.0], [178.0, 127.0], [179.0, 127.0], [179.0, 128.0], [180.0, 128.0], [181.0, 128.0], [182.0, 128.0], [183.0, 128.0], [184.0, 128.0], [185.0, 128.0], [186.0, 128.0], [187.0, 128.0], [188.0, 128.0], [189.0, 128.0], [190.0, 128.0], [190.0, 127.0], [191.0, 127.0], [191.0, 126.0], [192.0, 126.0], [192.0, 125.0], [193.0, 125.0], [194.0, 125.0], [195.0, 125.0], [196.0, 125.0], [197.0, 125.0], [197.0, 124.0], [196.0, 124.0], [196.0, 123.0], [195.0, 123.0], [195.0, 122.0], [194.0, 122.0], [194.0, 121.0], [193.0, 121.0], [193.0, 120.0], [193.0, 119.0], [192.0, 119.0], [192.0, 118.0], [192.0, 117.0], [191.0, 117.0], [190.0, 117.0], [190.0, 116.0], [190.0, 115.0], [190.0, 114.0], [191.0, 114.0], [191.0, 113.0], [190.0, 113.0], [189.0, 113.0], [188.0, 113.0], [187.0, 113.0], [187.0, 114.0], [187.0, 115.0], [187.0, 116.0], [187.0, 117.0], [186.0, 117.0], [186.0, 118.0], [185.0, 118.0], [184.0, 118.0], [183.0, 118.0], [182.0, 118.0], [181.0, 118.0], [180.0, 118.0], [180.0, 117.0], [179.0, 117.0], [179.0, 116.0], [179.0, 115.0], [179.0, 114.0], [178.0, 114.0], [178.0, 115.0], [177.0, 115.0], [176.0, 115.0], [175.0, 115.0], [175.0, 114.0], [175.0, 113.0], [174.0, 113.0], [173.0, 113.0], [172.0, 113.0], [171.0, 113.0], [170.0, 113.0], [169.0, 113.0], [168.0, 113.0], [167.0, 113.0], [166.0, 113.0], [165.0, 113.0], [164.0, 113.0], [163.0, 113.0], [162.0, 113.0], [162.0, 114.0], [161.0, 114.0], [160.0, 114.0], [159.0, 114.0], [158.0, 114.0], [157.0, 114.0], [156.0, 114.0], [156.0, 113.0], [155.0, 113.0], [155.0, 112.0], [154.0, 112.0], [153.0, 112.0], [153.0, 113.0], [152.0, 113.0], [152.0, 112.0], [151.0, 112.0], [150.0, 112.0], [149.0, 112.0], [149.0, 113.0], [148.0, 113.0], [147.0, 113.0], [146.0, 113.0], [146.0, 114.0], [145.0, 114.0], [145.0, 115.0], [144.0, 115.0], [144.0, 116.0], [143.0, 116.0], [142.0, 116.0], [142.0, 115.0], [141.0, 115.0], [141.0, 114.0], [140.0, 114.0], [139.0, 114.0], [139.0, 113.0], [139.0, 112.0], [139.0, 111.0], [139.0, 110.0], [139.0, 109.0], [139.0, 108.0], [138.0, 108.0], [138.0, 107.0], [138.0, 106.0], [138.0, 105.0], [138.0, 104.0], [138.0, 103.0], [138.0, 102.0], [138.0, 101.0], [138.0, 100.0], [137.0, 100.0], [137.0, 99.0], [138.0, 99.0], [138.0, 98.0], [138.0, 97.0], [139.0, 97.0], [139.0, 96.0], [140.0, 96.0], [141.0, 96.0], [141.0, 95.0], [142.0, 95.0], [142.0, 94.0], [143.0, 94.0], [144.0, 94.0], [144.0, 93.0], [145.0, 93.0], [145.0, 92.0], [146.0, 92.0], [147.0, 92.0], [147.0, 91.0], [148.0, 91.0], [148.0, 90.0], [149.0, 90.0], [149.0, 89.0], [150.0, 89.0], [151.0, 89.0], [151.0, 90.0], [152.0, 90.0], [152.0, 89.0], [153.0, 89.0], [154.0, 89.0], [155.0, 89.0], [156.0, 89.0], [157.0, 89.0], [157.0, 88.0], [158.0, 88.0], [159.0, 88.0], [159.0, 87.0], [159.0, 86.0], [159.0, 85.0], [159.0, 84.0], [158.0, 84.0], [157.0, 84.0], [156.0, 84.0], [156.0, 83.0], [156.0, 82.0], [156.0, 81.0], [157.0, 81.0], [158.0, 81.0], [159.0, 81.0], [159.0, 80.0], [160.0, 80.0], [161.0, 80.0], [162.0, 80.0], [163.0, 80.0], [164.0, 80.0], [165.0, 80.0], [166.0, 80.0], [167.0, 80.0], [168.0, 80.0], [169.0, 80.0], [170.0, 80.0], [171.0, 80.0], [172.0, 80.0], [173.0, 80.0], [174.0, 80.0], [175.0, 80.0], [175.0, 79.0], [176.0, 79.0], [177.0, 79.0], [178.0, 79.0], [179.0, 79.0], [179.0, 80.0], [180.0, 80.0], [181.0, 80.0], [182.0, 80.0], [183.0, 80.0], [184.0, 80.0], [185.0, 80.0], [186.0, 80.0], [187.0, 80.0], [188.0, 80.0], [189.0, 80.0], [190.0, 80.0], [191.0, 80.0], [191.0, 79.0], [192.0, 79.0], [193.0, 79.0], [194.0, 79.0], [195.0, 79.0], [196.0, 79.0], [197.0, 79.0], [198.0, 79.0], [199.0, 79.0], [200.0, 79.0], [201.0, 79.0], [202.0, 79.0], [203.0, 79.0], [204.0, 79.0], [205.0, 79.0], [206.0, 79.0], [207.0, 79.0], [207.0, 78.0], [206.0, 78.0], [205.0, 78.0], [205.0, 77.0], [205.0, 76.0], [205.0, 75.0], [205.0, 74.0], [205.0, 73.0], [205.0, 72.0], [205.0, 71.0], [205.0, 70.0], [205.0, 69.0], [204.0, 69.0], [204.0, 68.0], [204.0, 67.0], [204.0, 66.0], [205.0, 66.0], [205.0, 65.0], [204.0, 65.0], [203.0, 65.0], [202.0, 65.0], [201.0, 65.0], [200.0, 65.0], [199.0, 65.0], [198.0, 65.0], [197.0, 65.0], [196.0, 65.0], [195.0, 65.0], [195.0, 66.0], [194.0, 66.0], [194.0, 67.0], [193.0, 67.0], [192.0, 67.0], [191.0, 67.0], [191.0, 66.0], [191.0, 65.0], [190.0, 65.0], [189.0, 65.0], [188.0, 65.0], [187.0, 65.0], [186.0, 65.0], [185.0, 65.0], [185.0, 66.0], [185.0, 67.0], [185.0, 68.0], [185.0, 69.0], [185.0, 70.0], [184.0, 70.0], [184.0, 71.0], [183.0, 71.0], [182.0, 71.0], [181.0, 71.0], [180.0, 71.0], [179.0, 71.0], [179.0, 70.0], [178.0, 70.0], [177.0, 70.0], [176.0, 70.0], [175.0, 70.0], [175.0, 69.0], [175.0, 68.0], [175.0, 67.0], [175.0, 66.0], [175.0, 65.0], [174.0, 65.0], [173.0, 65.0], [172.0, 65.0], [171.0, 65.0], [170.0, 65.0], [169.0, 65.0], [168.0, 65.0], [167.0, 65.0], [166.0, 65.0], [165.0, 65.0], [164.0, 65.0], [163.0, 65.0], [162.0, 65.0], [162.0, 66.0], [161.0, 66.0], [160.0, 66.0], [159.0, 66.0], [158.0, 66.0], [157.0, 66.0], [156.0, 66.0], [156.0, 65.0], [156.0, 64.0], [156.0, 63.0], [157.0, 63.0], [158.0, 63.0], [159.0, 63.0], [159.0, 62.0], [160.0, 62.0], [160.0, 61.0], [160.0, 60.0], [160.0, 59.0], [160.0, 58.0], [160.0, 57.0], [160.0, 56.0], [160.0, 55.0], [160.0, 54.0], [160.0, 53.0], [160.0, 52.0], [160.0, 51.0], [159.0, 51.0], [159.0, 50.0], [158.0, 50.0], [157.0, 50.0], [156.0, 50.0], [155.0, 50.0], [154.0, 50.0], [153.0, 50.0], [152.0, 50.0], [151.0, 50.0], [150.0, 50.0], [149.0, 50.0], [148.0, 50.0], [147.0, 50.0], [147.0, 51.0], [146.0, 51.0], [146.0, 52.0], [145.0, 52.0], [144.0, 52.0], [143.0, 52.0], [143.0, 51.0], [142.0, 51.0], [141.0, 51.0], [140.0, 51.0], [139.0, 51.0], [139.0, 50.0], [139.0, 49.0], [139.0, 48.0], [139.0, 47.0], [140.0, 47.0], [141.0, 47.0], [142.0, 47.0], [143.0, 47.0], [143.0, 46.0], [144.0, 46.0], [145.0, 46.0], [146.0, 46.0], [147.0, 46.0], [147.0, 47.0], [148.0, 47.0], [149.0, 47.0], [150.0, 47.0], [151.0, 47.0], [152.0, 47.0], [153.0, 47.0], [154.0, 47.0], [155.0, 47.0], [156.0, 47.0], [157.0, 47.0], [157.0, 46.0], [157.0, 45.0], [157.0, 44.0], [157.0, 43.0], [157.0, 42.0], [157.0, 41.0], [157.0, 40.0], [157.0, 39.0], [157.0, 38.0], [156.0, 38.0], [156.0, 37.0], [156.0, 36.0], [156.0, 35.0], [157.0, 35.0], [157.0, 34.0], [158.0, 34.0], [159.0, 34.0], [159.0, 33.0], [159.0, 32.0], [159.0, 31.0], [160.0, 31.0], [160.0, 30.0], [160.0, 29.0], [160.0, 28.0], [160.0, 27.0], [160.0, 26.0], [160.0, 25.0], [160.0, 24.0], [160.0, 23.0], [160.0, 22.0], [160.0, 21.0], [160.0, 20.0], [160.0, 19.0], [159.0, 19.0], [159.0, 18.0], [159.0, 17.0], [159.0, 16.0], [159.0, 15.0], [160.0, 15.0], [160.0, 14.0], [160.0, 13.0], [160.0, 12.0], [160.0, 11.0], [160.0, 10.0], [160.0, 9.0], [160.0, 8.0], [159.0, 8.0], [158.0, 8.0], [158.0, 7.0], [159.0, 7.0], [160.0, 7.0], [160.0, 6.0], [160.0, 5.0], [160.0, 4.0], [160.0, 3.0], [159.0, 3.0], [159.0, 4.0], [158.0, 4.0], [158.0, 5.0], [157.0, 5.0], [157.0, 6.0], [157.0, 7.0], [156.0, 7.0], [155.0, 7.0], [154.0, 7.0], [153.0, 7.0], [152.0, 7.0], [151.0, 7.0], [151.0, 6.0], [150.0, 6.0], [150.0, 5.0], [149.0, 5.0], [148.0, 5.0], [147.0, 5.0], [147.0, 4.0], [146.0, 4.0], [146.0, 3.0], [145.0, 3.0], [144.0, 3.0], [143.0, 3.0], [143.0, 2.0], [142.0, 2.0], [141.0, 2.0], [140.0, 2.0], [140.0, 3.0], [139.0, 3.0], [139.0, 4.0], [139.0, 5.0], [138.0, 5.0], [137.0, 5.0], [136.0, 5.0], [136.0, 4.0], [135.0, 4.0], [135.0, 3.0], [134.0, 3.0], [134.0, 2.0], [133.0, 2.0], [133.0, 1.0], [132.0, 1.0], [131.0, 1.0], [131.0, 2.0], [130.0, 2.0], [130.0, 3.0], [129.0, 3.0], [128.0, 3.0], [127.0, 3.0], [127.0, 2.0], [127.0, 1.0], [126.0, 1.0], [125.0, 1.0], [124.0, 1.0], [123.0, 1.0], [122.0, 1.0], [121.0, 1.0], [120.0, 1.0], [119.0, 1.0], [118.0, 1.0], [117.0, 1.0], [116.0, 1.0], [115.0, 1.0], [115.0, 2.0], [114.0, 2.0], [114.0, 3.0], [115.0, 3.0], [115.0, 4.0], [115.0, 5.0], [115.0, 6.0], [115.0, 7.0], [114.0, 7.0], [113.0, 7.0], [113.0, 8.0], [113.0, 9.0], [113.0, 10.0], [113.0, 11.0], [113.0, 12.0], [113.0, 13.0], [113.0, 14.0], [113.0, 15.0], [114.0, 15.0], [115.0, 15.0], [115.0, 16.0], [115.0, 17.0], [116.0, 17.0], [116.0, 18.0], [117.0, 18.0], [117.0, 19.0], [117.0, 20.0], [117.0, 21.0], [116.0, 21.0], [116.0, 22.0], [116.0, 23.0], [116.0, 24.0], [116.0, 25.0], [117.0, 25.0], [117.0, 26.0], [117.0, 27.0], [117.0, 28.0], [117.0, 29.0], [117.0, 30.0], [116.0, 30.0], [115.0, 30.0], [115.0, 31.0], [115.0, 32.0], [115.0, 33.0], [115.0, 34.0], [114.0, 34.0], [114.0, 35.0], [113.0, 35.0], [113.0, 36.0], [113.0, 37.0], [113.0, 38.0], [113.0, 39.0], [113.0, 40.0], [113.0, 41.0], [113.0, 42.0], [114.0, 42.0], [115.0, 42.0], [115.0, 43.0], [115.0, 44.0], [115.0, 45.0], [115.0, 46.0], [114.0, 46.0], [114.0, 45.0], [113.0, 45.0], [113.0, 46.0], [113.0, 47.0], [114.0, 47.0], [115.0, 47.0], [116.0, 47.0], [117.0, 47.0], [118.0, 47.0], [119.0, 47.0], [120.0, 47.0], [121.0, 47.0], [122.0, 47.0], [123.0, 47.0], [124.0, 47.0], [125.0, 47.0], [126.0, 47.0], [127.0, 47.0], [128.0, 47.0], [128.0, 46.0], [129.0, 46.0], [130.0, 46.0], [131.0, 46.0], [132.0, 46.0], [132.0, 47.0], [133.0, 47.0], [134.0, 47.0], [135.0, 47.0], [135.0, 48.0], [135.0, 49.0], [134.0, 49.0], [134.0, 50.0], [134.0, 51.0], [133.0, 51.0], [132.0, 51.0], [131.0, 51.0], [131.0, 52.0], [130.0, 52.0], [129.0, 52.0], [128.0, 52.0], [128.0, 51.0], [128.0, 50.0], [127.0, 50.0], [126.0, 50.0], [125.0, 50.0], [124.0, 50.0], [123.0, 50.0], [123.0, 51.0], [124.0, 51.0], [124.0, 52.0], [124.0, 53.0], [124.0, 54.0], [125.0, 54.0], [125.0, 55.0], [126.0, 55.0], [126.0, 56.0], [126.0, 57.0], [126.0, 58.0], [126.0, 59.0], [126.0, 60.0], [125.0, 60.0], [125.0, 61.0], [124.0, 61.0], [124.0, 62.0], [123.0, 62.0], [122.0, 62.0], [121.0, 62.0], [120.0, 62.0], [119.0, 62.0], [118.0, 62.0], [118.0, 61.0], [117.0, 61.0], [117.0, 60.0], [116.0, 60.0], [116.0, 59.0], [116.0, 58.0], [116.0, 57.0], [115.0, 57.0], [115.0, 56.0], [115.0, 55.0], [115.0, 54.0], [115.0, 53.0], [114.0, 53.0], [114.0, 52.0], [115.0, 52.0], [115.0, 51.0], [116.0, 51.0], [117.0, 51.0], [118.0, 51.0], [119.0, 51.0], [120.0, 51.0], [120.0, 50.0], [119.0, 50.0], [118.0, 50.0], [117.0, 50.0], [116.0, 50.0], [115.0, 50.0], [114.0, 50.0], [114.0, 51.0], [113.0, 51.0], [113.0, 52.0], [113.0, 53.0], [113.0, 54.0], [113.0, 55.0], [113.0, 56.0], [113.0, 57.0], [113.0, 58.0], [113.0, 59.0], [113.0, 60.0], [113.0, 61.0], [113.0, 62.0], [114.0, 62.0], [115.0, 62.0], [115.0, 63.0], [116.0, 63.0], [117.0, 63.0], [117.0, 64.0], [117.0, 65.0], [117.0, 66.0], [116.0, 66.0], [115.0, 66.0], [114.0, 66.0], [113.0, 66.0], [112.0, 66.0], [111.0, 66.0], [110.0, 66.0], [109.0, 66.0], [108.0, 66.0], [107.0, 66.0], [106.0, 66.0], [105.0, 66.0], [104.0, 66.0], [103.0, 66.0], [102.0, 66.0], [101.0, 66.0], [100.0, 66.0], [99.0, 66.0], [98.0, 66.0], [98.0, 67.0], [97.0, 67.0], [96.0, 67.0], [95.0, 67.0], [95.0, 66.0], [94.0, 66.0], [93.0, 66.0], [92.0, 66.0], [91.0, 66.0], [90.0, 66.0], [89.0, 66.0], [88.0, 66.0], [87.0, 66.0], [86.0, 66.0], [85.0, 66.0], [84.0, 66.0], [83.0, 66.0], [83.0, 67.0], [83.0, 68.0], [82.0, 68.0], [81.0, 68.0], [80.0, 68.0], [79.0, 68.0], [79.0, 67.0], [79.0, 66.0], [79.0, 65.0], [78.0, 65.0], [77.0, 65.0], [76.0, 65.0], [75.0, 65.0], [74.0, 65.0], [73.0, 65.0], [72.0, 65.0], [71.0, 65.0], [70.0, 65.0], [69.0, 65.0], [68.0, 65.0], [67.0, 65.0], [67.0, 66.0], [66.0, 66.0], [66.0, 67.0], [67.0, 67.0], [68.0, 67.0], [68.0, 68.0], [69.0, 68.0], [69.0, 69.0], [69.0, 70.0], [69.0, 71.0], [68.0, 71.0], [68.0, 72.0], [68.0, 73.0], [68.0, 74.0], [68.0, 75.0], [69.0, 75.0], [69.0, 76.0], [69.0, 77.0], [69.0, 78.0], [69.0, 79.0], [69.0, 80.0], [70.0, 80.0], [71.0, 80.0], [72.0, 80.0], [73.0, 80.0], [74.0, 80.0], [75.0, 80.0], [75.0, 79.0], [74.0, 79.0], [74.0, 78.0], [74.0, 77.0], [74.0, 76.0], [75.0, 76.0], [76.0, 76.0], [76.0, 77.0], [77.0, 77.0], [77.0, 78.0], [76.0, 78.0], [76.0, 79.0], [77.0, 79.0], [77.0, 80.0], [78.0, 80.0], [79.0, 80.0], [79.0, 79.0], [80.0, 79.0], [81.0, 79.0], [82.0, 79.0], [83.0, 79.0], [83.0, 80.0], [84.0, 80.0], [85.0, 80.0], [86.0, 80.0], [87.0, 80.0], [88.0, 80.0], [89.0, 80.0], [90.0, 80.0], [91.0, 80.0], [92.0, 80.0], [93.0, 80.0], [94.0, 80.0], [95.0, 80.0], [95.0, 79.0], [96.0, 79.0], [96.0, 78.0], [97.0, 78.0], [97.0, 79.0], [98.0, 79.0], [99.0, 79.0], [99.0, 80.0], [100.0, 80.0], [101.0, 80.0], [102.0, 80.0], [103.0, 80.0], [104.0, 80.0], [105.0, 80.0], [106.0, 80.0], [107.0, 80.0], [108.0, 80.0], [109.0, 80.0], [110.0, 80.0], [111.0, 80.0], [112.0, 80.0], [113.0, 80.0], [114.0, 80.0], [115.0, 80.0], [115.0, 79.0], [116.0, 79.0], [117.0, 79.0], [118.0, 79.0], [119.0, 79.0], [120.0, 79.0], [121.0, 79.0], [122.0, 79.0], [123.0, 79.0], [124.0, 79.0], [125.0, 79.0], [126.0, 79.0], [127.0, 79.0], [128.0, 79.0], [129.0, 79.0], [130.0, 79.0], [131.0, 79.0], [132.0, 79.0], [133.0, 79.0], [134.0, 79.0], [135.0, 79.0], [136.0, 79.0], [137.0, 79.0], [138.0, 79.0], [139.0, 79.0], [140.0, 79.0], [141.0, 79.0], [142.0, 79.0], [143.0, 79.0], [144.0, 79.0], [145.0, 79.0], [146.0, 79.0], [146.0, 80.0], [146.0, 81.0], [146.0, 82.0], [145.0, 82.0], [145.0, 83.0], [144.0, 83.0], [144.0, 84.0], [143.0, 84.0], [142.0, 84.0], [142.0, 85.0], [143.0, 85.0], [143.0, 86.0], [142.0, 86.0], [141.0, 86.0], [141.0, 87.0], [140.0, 87.0], [140.0, 88.0], [139.0, 88.0], [139.0, 89.0], [138.0, 89.0], [138.0, 90.0], [137.0, 90.0], [137.0, 91.0], [136.0, 91.0], [136.0, 92.0], [135.0, 92.0], [135.0, 93.0], [134.0, 93.0], [134.0, 94.0], [133.0, 94.0], [133.0, 95.0], [132.0, 95.0], [132.0, 96.0], [132.0, 97.0], [132.0, 98.0], [131.0, 98.0], [131.0, 99.0], [130.0, 99.0], [129.0, 99.0], [129.0, 100.0], [129.0, 101.0], [128.0, 101.0], [127.0, 101.0], [126.0, 101.0], [125.0, 101.0], [124.0, 101.0], [124.0, 100.0], [123.0, 100.0], [122.0, 100.0], [122.0, 101.0], [121.0, 101.0], [120.0, 101.0], [120.0, 102.0], [119.0, 102.0], [118.0, 102.0], [118.0, 103.0], [117.0, 103.0], [116.0, 103.0], [115.0, 103.0], [115.0, 104.0], [114.0, 104.0], [113.0, 104.0], [113.0, 105.0], [113.0, 106.0], [113.0, 107.0], [113.0, 108.0], [113.0, 109.0], [113.0, 110.0], [114.0, 110.0], [114.0, 109.0], [115.0, 109.0], [116.0, 109.0], [117.0, 109.0], [117.0, 110.0], [118.0, 110.0], [118.0, 111.0], [118.0, 112.0], [117.0, 112.0], [117.0, 113.0], [116.0, 113.0], [115.0, 113.0], [114.0, 113.0], [114.0, 114.0], [113.0, 114.0], [112.0, 114.0], [111.0, 114.0], [110.0, 114.0], [109.0, 114.0], [108.0, 114.0], [107.0, 114.0], [106.0, 114.0], [105.0, 114.0], [104.0, 114.0], [103.0, 114.0], [102.0, 114.0], [101.0, 114.0], [100.0, 114.0], [99.0, 114.0], [98.0, 114.0], [98.0, 115.0], [97.0, 115.0], [96.0, 115.0], [95.0, 115.0], [95.0, 114.0], [94.0, 114.0], [93.0, 114.0], [92.0, 114.0], [91.0, 114.0], [90.0, 114.0], [89.0, 114.0], [88.0, 114.0], [87.0, 114.0], [86.0, 114.0], [85.0, 114.0], [84.0, 114.0], [83.0, 114.0], [82.0, 114.0], [82.0, 115.0], [81.0, 115.0], [80.0, 115.0], [79.0, 115.0], [79.0, 116.0], [78.0, 116.0], [77.0, 116.0], [76.0, 116.0], [76.0, 115.0], [76.0, 114.0], [76.0, 113.0], [75.0, 113.0], [74.0, 113.0], [73.0, 113.0], [72.0, 113.0], [71.0, 113.0], [70.0, 113.0], [69.0, 113.0], [68.0, 113.0], [67.0, 113.0], [67.0, 114.0], [66.0, 114.0], [66.0, 115.0], [67.0, 115.0], [68.0, 115.0], [68.0, 116.0], [69.0, 116.0], [69.0, 117.0], [69.0, 118.0], [69.0, 119.0], [68.0, 119.0], [68.0, 120.0], [68.0, 121.0], [68.0, 122.0], [68.0, 123.0], [69.0, 123.0], [69.0, 124.0], [69.0, 125.0], [69.0, 126.0], [69.0, 127.0], [69.0, 128.0], [70.0, 128.0], [71.0, 128.0], [72.0, 128.0], [73.0, 128.0], [74.0, 128.0], [74.0, 127.0], [73.0, 127.0], [73.0, 126.0], [73.0, 125.0], [73.0, 124.0], [74.0, 124.0], [74.0, 123.0], [75.0, 123.0], [76.0, 123.0], [77.0, 123.0], [78.0, 123.0], [78.0, 124.0], [78.0, 125.0], [79.0, 125.0], [80.0, 125.0], [80.0, 126.0], [80.0, 127.0], [81.0, 127.0], [82.0, 127.0], [83.0, 127.0], [83.0, 128.0], [84.0, 128.0], [85.0, 128.0], [86.0, 128.0], [87.0, 128.0], [88.0, 128.0], [88.0, 127.0], [88.0, 126.0], [88.0, 125.0], [88.0, 124.0], [89.0, 124.0], [90.0, 124.0], [90.0, 123.0], [91.0, 123.0], [92.0, 123.0], [93.0, 123.0], [94.0, 123.0], [94.0, 124.0], [95.0, 124.0], [96.0, 124.0], [96.0, 125.0], [96.0, 126.0], [96.0, 127.0], [97.0, 127.0], [98.0, 127.0], [99.0, 127.0], [99.0, 128.0], [100.0, 128.0], [101.0, 128.0], [102.0, 128.0], [103.0, 128.0], [104.0, 128.0], [105.0, 128.0], [106.0, 128.0], [107.0, 128.0], [108.0, 128.0], [109.0, 128.0], [110.0, 128.0], [111.0, 128.0], [111.0, 127.0], [112.0, 127.0], [112.0, 126.0], [113.0, 126.0], [113.0, 127.0], [114.0, 127.0], [115.0, 127.0], [115.0, 128.0], [116.0, 128.0], [117.0, 128.0], [118.0, 128.0], [119.0, 128.0], [120.0, 128.0], [121.0, 128.0], [122.0, 128.0], [123.0, 128.0], [124.0, 128.0], [125.0, 128.0], [126.0, 128.0], [127.0, 128.0], [127.0, 127.0], [128.0, 127.0], [129.0, 127.0], [130.0, 127.0], [131.0, 127.0], [131.0, 128.0], [131.0, 129.0], [131.0, 130.0], [130.0, 130.0], [130.0, 131.0], [130.0, 132.0], [130.0, 133.0], [130.0, 134.0], [130.0, 135.0], [130.0, 136.0], [130.0, 137.0], [130.0, 138.0], [130.0, 139.0], [130.0, 140.0], [130.0, 141.0], [130.0, 142.0], [130.0, 143.0], [131.0, 143.0], [131.0, 144.0], [131.0, 145.0], [131.0, 146.0], [130.0, 146.0], [130.0, 147.0], [130.0, 148.0], [130.0, 149.0], [130.0, 150.0], [130.0, 151.0], [130.0, 152.0], [130.0, 153.0], [130.0, 154.0], [130.0, 155.0], [130.0, 156.0], [131.0, 156.0], [131.0, 157.0], [132.0, 157.0], [132.0, 158.0], [132.0, 159.0], [132.0, 160.0], [131.0, 160.0], [131.0, 161.0], [131.0, 162.0], [130.0, 162.0], [130.0, 163.0], [129.0, 163.0], [128.0, 163.0], [127.0, 163.0], [127.0, 162.0], [126.0, 162.0], [125.0, 162.0], [124.0, 162.0], [123.0, 162.0], [122.0, 162.0], [121.0, 162.0], [120.0, 162.0], [119.0, 162.0], [118.0, 162.0], [117.0, 162.0], [116.0, 162.0], [115.0, 162.0], [114.0, 162.0], [114.0, 163.0], [113.0, 163.0], [112.0, 163.0], [111.0, 163.0], [111.0, 162.0], [111.0, 161.0], [110.0, 161.0], [109.0, 161.0], [108.0, 161.0], [107.0, 161.0], [106.0, 161.0], [105.0, 161.0], [105.0, 162.0], [106.0, 162.0], [106.0, 163.0], [106.0, 164.0], [106.0, 165.0], [106.0, 166.0], [105.0, 166.0], [105.0, 167.0], [104.0, 167.0], [103.0, 167.0], [103.0, 168.0], [102.0, 168.0], [102.0, 167.0], [101.0, 167.0], [100.0, 167.0], [100.0, 168.0], [99.0, 168.0], [98.0, 168.0], [98.0, 169.0], [98.0, 170.0], [98.0, 171.0], [99.0, 171.0], [100.0, 171.0], [100.0, 172.0], [100.0, 173.0], [100.0, 174.0], [99.0, 174.0], [98.0, 174.0], [98.0, 175.0], [99.0, 175.0], [100.0, 175.0], [100.0, 176.0], [100.0, 177.0], [100.0, 178.0], [99.0, 178.0], [99.0, 179.0], [99.0, 180.0], [99.0, 181.0], [99.0, 182.0], [99.0, 183.0], [99.0, 184.0], [99.0, 185.0], [99.0, 186.0], [99.0, 187.0], [99.0, 188.0], [99.0, 189.0], [100.0, 189.0], [101.0, 189.0], [101.0, 190.0], [102.0, 190.0], [103.0, 190.0], [103.0, 191.0], [103.0, 192.0], [103.0, 193.0], [104.0, 193.0], [104.0, 194.0], [103.0, 194.0], [102.0, 194.0], [102.0, 195.0], [102.0, 196.0], [101.0, 196.0], [101.0, 197.0], [100.0, 197.0], [99.0, 197.0], [99.0, 196.0], [98.0, 196.0], [98.0, 195.0], [97.0, 195.0], [96.0, 195.0], [95.0, 195.0], [95.0, 194.0], [94.0, 194.0], [93.0, 194.0], [93.0, 193.0], [93.0, 192.0], [93.0, 191.0], [92.0, 191.0], [92.0, 190.0], [91.0, 190.0], [91.0, 191.0], [91.0, 192.0], [91.0, 193.0], [91.0, 194.0], [90.0, 194.0], [89.0, 194.0], [88.0, 194.0], [88.0, 193.0], [87.0, 193.0], [87.0, 192.0], [86.0, 192.0], [86.0, 193.0], [86.0, 194.0], [85.0, 194.0], [84.0, 194.0], [83.0, 194.0], [82.0, 194.0], [82.0, 195.0], [81.0, 195.0], [80.0, 195.0], [79.0, 195.0], [79.0, 194.0], [78.0, 194.0], [77.0, 194.0], [76.0, 194.0], [75.0, 194.0], [74.0, 194.0], [73.0, 194.0], [72.0, 194.0], [71.0, 194.0], [70.0, 194.0], [69.0, 194.0], [68.0, 194.0], [67.0, 194.0], [66.0, 194.0], [66.0, 195.0], [66.0, 196.0], [66.0, 197.0], [66.0, 198.0], [66.0, 199.0], [66.0, 200.0], [66.0, 201.0], [66.0, 202.0], [67.0, 202.0], [68.0, 202.0], [69.0, 202.0], [70.0, 202.0], [70.0, 203.0], [71.0, 203.0], [71.0, 204.0], [71.0, 205.0], [71.0, 206.0], [70.0, 206.0], [70.0, 207.0], [69.0, 207.0], [69.0, 208.0], [68.0, 208.0], [67.0, 208.0], [67.0, 209.0], [68.0, 209.0], [69.0, 209.0], [70.0, 209.0], [70.0, 210.0], [70.0, 211.0], [71.0, 211.0], [72.0, 211.0], [72.0, 212.0], [72.0, 213.0], [71.0, 213.0], [71.0, 214.0], [71.0, 215.0], [71.0, 216.0], [70.0, 216.0], [69.0, 216.0], [68.0, 216.0], [67.0, 216.0], [67.0, 215.0], [66.0, 215.0], [65.0, 215.0], [64.0, 215.0], [64.0, 214.0], [64.0, 213.0], [64.0, 212.0], [64.0, 211.0], [63.0, 211.0], [63.0, 210.0], [62.0, 210.0], [62.0, 211.0], [61.0, 211.0], [60.0, 211.0], [59.0, 211.0], [58.0, 211.0], [58.0, 210.0], [58.0, 209.0], [57.0, 209.0], [56.0, 209.0], [55.0, 209.0], [54.0, 209.0], [53.0, 209.0], [52.0, 209.0], [51.0, 209.0], [50.0, 209.0], [50.0, 210.0], [49.0, 210.0], [48.0, 210.0], [47.0, 210.0], [47.0, 209.0], [47.0, 208.0], [46.0, 208.0], [46.0, 207.0], [47.0, 207.0], [47.0, 206.0], [46.0, 206.0], [46.0, 205.0], [46.0, 204.0], [47.0, 204.0], [47.0, 203.0], [46.0, 203.0], [46.0, 202.0], [46.0, 201.0], [45.0, 201.0], [45.0, 200.0], [46.0, 200.0], [46.0, 199.0], [47.0, 199.0], [47.0, 198.0], [48.0, 198.0], [48.0, 197.0], [48.0, 196.0], [47.0, 196.0], [47.0, 195.0], [46.0, 195.0], [45.0, 195.0], [44.0, 195.0], [43.0, 195.0], [43.0, 194.0], [43.0, 193.0], [43.0, 192.0], [43.0, 191.0], [44.0, 191.0], [45.0, 191.0], [46.0, 191.0], [46.0, 190.0], [45.0, 190.0], [45.0, 189.0], [44.0, 189.0], [43.0, 189.0], [43.0, 188.0], [43.0, 187.0], [43.0, 186.0], [43.0, 185.0], [43.0, 184.0], [43.0, 183.0], [44.0, 183.0], [44.0, 182.0], [45.0, 182.0], [45.0, 181.0], [46.0, 181.0], [47.0, 181.0], [48.0, 181.0], [48.0, 180.0], [48.0, 179.0], [47.0, 179.0], [47.0, 178.0], [47.0, 177.0], [46.0, 177.0], [46.0, 176.0], [45.0, 176.0], [45.0, 175.0], [44.0, 175.0], [44.0, 174.0], [45.0, 174.0], [45.0, 173.0], [45.0, 172.0], [45.0, 171.0], [44.0, 171.0], [44.0, 170.0], [45.0, 170.0], [45.0, 169.0], [44.0, 169.0], [43.0, 169.0], [42.0, 169.0], [41.0, 169.0], [40.0, 169.0], [39.0, 169.0], [38.0, 169.0], [38.0, 168.0], [39.0, 168.0], [40.0, 168.0], [41.0, 168.0], [42.0, 168.0], [43.0, 168.0], [44.0, 168.0], [45.0, 168.0], [46.0, 168.0], [46.0, 167.0], [46.0, 166.0], [45.0, 166.0], [45.0, 165.0], [45.0, 164.0], [46.0, 164.0], [46.0, 163.0], [45.0, 163.0], [45.0, 162.0], [46.0, 162.0], [47.0, 162.0], [47.0, 161.0], [46.0, 161.0], [45.0, 161.0], [44.0, 161.0], [43.0, 161.0], [42.0, 161.0], [41.0, 161.0], [40.0, 161.0], [39.0, 161.0], [38.0, 161.0], [37.0, 161.0], [36.0, 161.0], [35.0, 161.0], [35.0, 162.0], [34.0, 162.0], [34.0, 163.0], [35.0, 163.0], [35.0, 164.0], [35.0, 165.0], [35.0, 166.0], [35.0, 167.0], [35.0, 168.0], [35.0, 169.0], [35.0, 170.0], [35.0, 171.0], [34.0, 171.0], [33.0, 171.0], [32.0, 171.0], [31.0, 171.0], [30.0, 171.0], [30.0, 170.0], [30.0, 169.0], [30.0, 168.0], [30.0, 167.0], [30.0, 166.0], [30.0, 165.0], [30.0, 164.0], [30.0, 163.0], [31.0, 163.0], [31.0, 162.0], [31.0, 161.0], [30.0, 161.0], [29.0, 161.0], [28.0, 161.0], [28.0, 162.0], [28.0, 163.0], [28.0, 164.0], [28.0, 165.0], [28.0, 166.0], [28.0, 167.0], [28.0, 168.0], [28.0, 169.0], [28.0, 170.0], [27.0, 170.0], [27.0, 169.0], [27.0, 168.0], [27.0, 167.0], [27.0, 166.0], [26.0, 166.0], [26.0, 165.0], [26.0, 164.0], [26.0, 163.0], [26.0, 162.0], [26.0, 161.0], [25.0, 161.0], [24.0, 161.0], [23.0, 161.0], [22.0, 161.0], [21.0, 161.0], [20.0, 161.0], [19.0, 161.0], [19.0, 162.0], [20.0, 162.0], [21.0, 162.0], [22.0, 162.0], [23.0, 162.0]]
    
    brc203_holes = [[[79.0, 339.0], [79.0, 340.0], [78.0, 340.0], [78.0, 339.0], [79.0, 339.0]], [[80.0, 363.0], [80.0, 362.0], [81.0, 362.0], [81.0, 363.0], [80.0, 363.0]], [[115.0, 340.0], [115.0, 341.0], [114.0, 341.0], [114.0, 340.0], [115.0, 340.0]], [[178.0, 164.0], [179.0, 164.0], [180.0, 164.0], [180.0, 165.0], [180.0, 166.0], [179.0, 166.0], [179.0, 167.0], [178.0, 167.0], [177.0, 167.0], [177.0, 166.0], [177.0, 165.0], [178.0, 165.0], [178.0, 164.0]], [[143.0, 3.0], [143.0, 4.0], [142.0, 4.0], [142.0, 3.0], [143.0, 3.0]], [[76.0, 116.0], [76.0, 117.0], [77.0, 117.0], [77.0, 118.0], [77.0, 119.0], [77.0, 120.0], [77.0, 121.0], [77.0, 122.0], [76.0, 122.0], [75.0, 122.0], [75.0, 121.0], [74.0, 121.0], [74.0, 120.0], [74.0, 119.0], [73.0, 119.0], [73.0, 118.0], [73.0, 117.0], [73.0, 116.0], [74.0, 116.0], [75.0, 116.0], [76.0, 116.0]], [[198.0, 68.0], [199.0, 68.0], [199.0, 69.0], [198.0, 69.0], [198.0, 68.0]], [[193.0, 70.0], [194.0, 70.0], [195.0, 70.0], [196.0, 70.0], [196.0, 71.0], [196.0, 72.0], [196.0, 73.0], [196.0, 74.0], [196.0, 75.0], [196.0, 76.0], [195.0, 76.0], [194.0, 76.0], [193.0, 76.0], [193.0, 75.0], [193.0, 74.0], [192.0, 74.0], [192.0, 73.0], [192.0, 72.0], [192.0, 71.0], [193.0, 71.0], [193.0, 70.0]], [[189.0, 177.0], [190.0, 177.0], [190.0, 178.0], [190.0, 179.0], [189.0, 179.0], [188.0, 179.0], [187.0, 179.0], [186.0, 179.0], [186.0, 178.0], [186.0, 177.0], [187.0, 177.0], [188.0, 177.0], [189.0, 177.0]], [[188.0, 181.0], [189.0, 181.0], [190.0, 181.0], [190.0, 182.0], [190.0, 183.0], [189.0, 183.0], [189.0, 184.0], [188.0, 184.0], [187.0, 184.0], [186.0, 184.0], [186.0, 183.0], [186.0, 182.0], [187.0, 182.0], [187.0, 181.0], [188.0, 181.0]], [[184.0, 167.0], [183.0, 167.0], [182.0, 167.0], [182.0, 166.0], [181.0, 166.0], [181.0, 165.0], [181.0, 164.0], [182.0, 164.0], [182.0, 165.0], [183.0, 165.0], [184.0, 165.0], [184.0, 166.0], [184.0, 167.0]], [[179.0, 121.0], [178.0, 121.0], [178.0, 120.0], [179.0, 120.0], [179.0, 121.0]], [[177.0, 119.0], [177.0, 120.0], [176.0, 120.0], [176.0, 119.0], [177.0, 119.0]], [[173.0, 225.0], [174.0, 225.0], [174.0, 226.0], [174.0, 227.0], [174.0, 228.0], [173.0, 228.0], [172.0, 228.0], [171.0, 228.0], [171.0, 227.0], [171.0, 226.0], [171.0, 225.0], [172.0, 225.0], [173.0, 225.0]], [[153.0, 30.0], [152.0, 30.0], [151.0, 30.0], [150.0, 30.0], [150.0, 29.0], [149.0, 29.0], [149.0, 28.0], [148.0, 28.0], [148.0, 27.0], [148.0, 26.0], [148.0, 25.0], [147.0, 25.0], [147.0, 24.0], [147.0, 23.0], [147.0, 22.0], [147.0, 21.0], [146.0, 21.0], [146.0, 20.0], [147.0, 20.0], [147.0, 19.0], [148.0, 19.0], [149.0, 19.0], [150.0, 19.0], [151.0, 19.0], [152.0, 19.0], [152.0, 18.0], [153.0, 18.0], [154.0, 18.0], [155.0, 18.0], [155.0, 19.0], [156.0, 19.0], [156.0, 20.0], [156.0, 21.0], [156.0, 22.0], [157.0, 22.0], [157.0, 23.0], [158.0, 23.0], [158.0, 24.0], [158.0, 25.0], [158.0, 26.0], [158.0, 27.0], [158.0, 28.0], [157.0, 28.0], [157.0, 29.0], [156.0, 29.0], [156.0, 30.0], [155.0, 30.0], [154.0, 30.0], [153.0, 30.0]], [[152.0, 33.0], [153.0, 33.0], [153.0, 34.0], [152.0, 34.0], [152.0, 33.0]], [[151.0, 42.0], [152.0, 42.0], [152.0, 43.0], [152.0, 44.0], [152.0, 45.0], [151.0, 45.0], [150.0, 45.0], [149.0, 45.0], [149.0, 44.0], [149.0, 43.0], [150.0, 43.0], [150.0, 42.0], [151.0, 42.0]], [[143.0, 15.0], [143.0, 14.0], [143.0, 13.0], [143.0, 12.0], [144.0, 12.0], [144.0, 11.0], [145.0, 11.0], [146.0, 11.0], [146.0, 12.0], [147.0, 12.0], [147.0, 13.0], [147.0, 14.0], [147.0, 15.0], [147.0, 16.0], [147.0, 17.0], [147.0, 18.0], [146.0, 18.0], [146.0, 19.0], [145.0, 19.0], [144.0, 19.0], [143.0, 19.0], [143.0, 18.0], [143.0, 17.0], [143.0, 16.0], [143.0, 15.0]], [[142.0, 32.0], [143.0, 32.0], [143.0, 31.0], [144.0, 31.0], [144.0, 30.0], [143.0, 30.0], [143.0, 29.0], [143.0, 28.0], [144.0, 28.0], [144.0, 27.0], [145.0, 27.0], [146.0, 27.0], [146.0, 28.0], [146.0, 29.0], [146.0, 30.0], [146.0, 31.0], [147.0, 31.0], [147.0, 32.0], [147.0, 33.0], [147.0, 34.0], [146.0, 34.0], [146.0, 35.0], [145.0, 35.0], [144.0, 35.0], [143.0, 35.0], [143.0, 34.0], [143.0, 33.0], [142.0, 33.0], [142.0, 32.0]], [[141.0, 65.0], [141.0, 64.0], [142.0, 64.0], [142.0, 63.0], [143.0, 63.0], [144.0, 63.0], [145.0, 63.0], [146.0, 63.0], [146.0, 64.0], [146.0, 65.0], [146.0, 66.0], [145.0, 66.0], [145.0, 67.0], [144.0, 67.0], [143.0, 67.0], [142.0, 67.0], [142.0, 66.0], [142.0, 65.0], [141.0, 65.0]], [[128.0, 65.0], [127.0, 65.0], [127.0, 64.0], [128.0, 64.0], [128.0, 63.0], [129.0, 63.0], [130.0, 63.0], [131.0, 63.0], [132.0, 63.0], [132.0, 64.0], [132.0, 65.0], [132.0, 66.0], [131.0, 66.0], [131.0, 67.0], [130.0, 67.0], [129.0, 67.0], [128.0, 67.0], [128.0, 66.0], [128.0, 65.0]], [[127.0, 113.0], [127.0, 112.0], [128.0, 112.0], [128.0, 111.0], [129.0, 111.0], [130.0, 111.0], [131.0, 111.0], [132.0, 111.0], [132.0, 112.0], [132.0, 113.0], [132.0, 114.0], [131.0, 114.0], [131.0, 115.0], [130.0, 115.0], [129.0, 115.0], [128.0, 115.0], [128.0, 114.0], [128.0, 113.0], [127.0, 113.0]], [[127.0, 34.0], [127.0, 33.0], [127.0, 32.0], [127.0, 31.0], [127.0, 30.0], [127.0, 29.0], [127.0, 28.0], [128.0, 28.0], [129.0, 28.0], [130.0, 28.0], [130.0, 29.0], [130.0, 30.0], [130.0, 31.0], [131.0, 31.0], [131.0, 32.0], [131.0, 33.0], [131.0, 34.0], [130.0, 34.0], [130.0, 35.0], [129.0, 35.0], [128.0, 35.0], [127.0, 35.0], [127.0, 34.0]], [[127.0, 18.0], [127.0, 17.0], [127.0, 16.0], [127.0, 15.0], [127.0, 14.0], [127.0, 13.0], [127.0, 12.0], [128.0, 12.0], [128.0, 11.0], [129.0, 11.0], [130.0, 11.0], [130.0, 12.0], [131.0, 12.0], [131.0, 13.0], [131.0, 14.0], [131.0, 15.0], [131.0, 16.0], [131.0, 17.0], [131.0, 18.0], [130.0, 18.0], [130.0, 19.0], [129.0, 19.0], [128.0, 19.0], [127.0, 19.0], [127.0, 18.0]], [[122.0, 27.0], [122.0, 26.0], [122.0, 25.0], [123.0, 25.0], [124.0, 25.0], [125.0, 25.0], [125.0, 26.0], [125.0, 27.0], [125.0, 28.0], [126.0, 28.0], [126.0, 29.0], [126.0, 30.0], [126.0, 31.0], [125.0, 31.0], [124.0, 31.0], [123.0, 31.0], [123.0, 30.0], [123.0, 29.0], [123.0, 28.0], [122.0, 28.0], [122.0, 27.0]], [[115.0, 12.0], [115.0, 11.0], [116.0, 11.0], [116.0, 12.0], [115.0, 12.0]], [[110.0, 218.0], [110.0, 217.0], [111.0, 217.0], [111.0, 218.0], [110.0, 218.0]], [[110.0, 212.0], [110.0, 211.0], [110.0, 210.0], [111.0, 210.0], [111.0, 211.0], [111.0, 212.0], [111.0, 213.0], [111.0, 214.0], [111.0, 215.0], [111.0, 216.0], [110.0, 216.0], [110.0, 215.0], [110.0, 214.0], [110.0, 213.0], [110.0, 212.0]], [[108.0, 354.0], [108.0, 353.0], [109.0, 353.0], [109.0, 354.0], [108.0, 354.0]], [[102.0, 221.0], [102.0, 222.0], [102.0, 223.0], [101.0, 223.0], [101.0, 222.0], [101.0, 221.0], [101.0, 220.0], [102.0, 220.0], [102.0, 221.0]], [[102.0, 217.0], [102.0, 218.0], [102.0, 219.0], [101.0, 219.0], [101.0, 218.0], [101.0, 217.0], [101.0, 216.0], [101.0, 215.0], [102.0, 215.0], [102.0, 216.0], [102.0, 217.0]], [[102.0, 213.0], [101.0, 213.0], [101.0, 212.0], [102.0, 212.0], [102.0, 213.0]], [[89.0, 231.0], [89.0, 230.0], [89.0, 229.0], [89.0, 228.0], [89.0, 227.0], [90.0, 227.0], [90.0, 226.0], [91.0, 226.0], [91.0, 225.0], [92.0, 225.0], [92.0, 226.0], [93.0, 226.0], [94.0, 226.0], [95.0, 226.0], [95.0, 227.0], [96.0, 227.0], [96.0, 228.0], [96.0, 229.0], [96.0, 230.0], [96.0, 231.0], [95.0, 231.0], [95.0, 232.0], [94.0, 232.0], [94.0, 233.0], [93.0, 233.0], [92.0, 233.0], [91.0, 233.0], [91.0, 232.0], [90.0, 232.0], [90.0, 231.0], [89.0, 231.0]], [[89.0, 235.0], [89.0, 234.0], [89.0, 233.0], [90.0, 233.0], [90.0, 234.0], [91.0, 234.0], [91.0, 235.0], [90.0, 235.0], [89.0, 235.0]], [[88.0, 121.0], [88.0, 120.0], [89.0, 120.0], [89.0, 121.0], [88.0, 121.0]], [[85.0, 341.0], [85.0, 340.0], [86.0, 340.0], [86.0, 341.0], [85.0, 341.0]], [[85.0, 122.0], [85.0, 121.0], [86.0, 121.0], [86.0, 122.0], [85.0, 122.0]], [[77.0, 350.0], [78.0, 350.0], [78.0, 349.0], [79.0, 349.0], [80.0, 349.0], [81.0, 349.0], [82.0, 349.0], [83.0, 349.0], [84.0, 349.0], [84.0, 350.0], [84.0, 351.0], [84.0, 352.0], [84.0, 353.0], [84.0, 354.0], [84.0, 355.0], [83.0, 355.0], [82.0, 355.0], [82.0, 356.0], [81.0, 356.0], [80.0, 356.0], [79.0, 356.0], [79.0, 355.0], [78.0, 355.0], [78.0, 354.0], [77.0, 354.0], [77.0, 353.0], [77.0, 352.0], [77.0, 351.0], [77.0, 350.0]], [[79.0, 123.0], [79.0, 122.0], [80.0, 122.0], [80.0, 123.0], [79.0, 123.0]], [[78.0, 345.0], [79.0, 345.0], [79.0, 346.0], [78.0, 346.0], [78.0, 345.0]], [[73.0, 70.0], [73.0, 69.0], [74.0, 69.0], [74.0, 68.0], [75.0, 68.0], [76.0, 68.0], [76.0, 69.0], [77.0, 69.0], [78.0, 69.0], [79.0, 69.0], [79.0, 70.0], [79.0, 71.0], [79.0, 72.0], [78.0, 72.0], [77.0, 72.0], [76.0, 72.0], [76.0, 71.0], [75.0, 71.0], [74.0, 71.0], [73.0, 71.0], [73.0, 70.0]], [[38.0, 163.0], [37.0, 163.0], [37.0, 162.0], [38.0, 162.0], [38.0, 163.0]], [[28.0, 346.0], [28.0, 345.0], [29.0, 345.0], [29.0, 346.0], [28.0, 346.0]]]
    
    
    polygon = Polygon(outer_boundary3, holes_cathedral) 
    start_pos = (125,500)
    
    polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError("Polygon is empty after cleaning, check input geometry.")
        
    print(f"--- Using start_pos: {start_pos} ---")
    fleet_size = 10
    num_stationary_goal = 10
    
    print("--- PRE-COMPUTATION: Triangulating Environment ---")
    
    # ... (Triangulation setup, identical) ...
    vertices = []
    segments = []
    holes_list = []
    
    if polygon.geom_type == 'Polygon':
        polygons_to_process = [polygon]
    elif polygon.geom_type == 'MultiPolygon':
        polygons_to_process = list(polygon.geoms)
    else:
        polygons_to_process = []
    
    current_vertex_index = 0
    
    for poly in polygons_to_process:
        if poly.is_empty: continue
        
        exterior_coords = list(poly.exterior.coords[:-1])
        vertices.extend(exterior_coords)
        ext_len = len(exterior_coords)
        segments.extend([[current_vertex_index + i, current_vertex_index + (i + 1) % ext_len] for i in range(ext_len)])
        current_vertex_index += ext_len
        
        for interior in poly.interiors:
            hole_coords = list(interior.coords[:-1])
            if not hole_coords: continue
            
            vertices.extend(hole_coords)
            hole_len = len(hole_coords)
            segments.extend([[current_vertex_index + i, current_vertex_index + (i + 1) % hole_len] for i in range(hole_len)])
            
            hole_poly = Polygon(interior)
            clean_hole_poly = hole_poly.buffer(0)
            if not clean_hole_poly.is_empty:
                holes_list.append(list(clean_hole_poly.representative_point().coords)[0])
            
            current_vertex_index += hole_len

    scene = { 'vertices': np.array(vertices), 'segments': np.array(segments) }
    if holes_list:
        scene['holes'] = np.array(holes_list)
        
    B = tr.triangulate(scene, 'pA') 
    
    halfedges = build_halfedges(B['triangles'])
    delaunator_like_data = {
        'coords': B['vertices'].flatten(),
        'triangles': B['triangles'].flatten(),
        'halfedges': halfedges,
    }
    
    obstructing_segments = {
        tuple(sorted(B['segments'][i])) 
        for i, m in enumerate(B['segment_markers']) if m == 1
    }
    
    def obstructs(edge_idx: int) -> bool:
        p1 = delaunator_like_data['triangles'][edge_idx]
        p2 = delaunator_like_data['triangles'][next_edge(edge_idx)]
        return tuple(sorted((p1, p2))) in obstructing_segments

    print(f"Triangulation complete. Total vertices: {len(B['vertices'])}")
    
    # --- NEW: Build the A* graph ONE TIME ---
    a_star_graph, tri_centroids = build_a_star_graph(delaunator_like_data, obstructs)
    # ---
    
    print("--- Simulation Start (FINAL: Chained-Auction + A* Paths) ---")
    
    # 6. Initialize Drone Fleet
    drone_fleet = [Drone(i, start_pos) for i in range(fleet_size)]
    
    first_drone = drone_fleet[0]
    first_drone.status = 'stationary'
    
    first_drone.vis_poly = get_visibility_from_triangulation(
        first_drone.pos, delaunator_like_data, obstructs
    )
    
    stationary_drones = [first_drone]
    shared_map = first_drone.vis_poly
    
    root_node = first_drone
    gathering_spot_node = root_node # Starts at the root
    
    # Set parent for all drones (for tree building)
    for drone in drone_fleet:
        if drone.id != root_node.id:
            drone.parent = root_node
    
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.ion()
    update_plot(ax, polygon, drone_fleet, shared_map, "Round 0: Initial State")
    time.sleep(0.1)

    # 7. Run Main Simulation Loop
    round_num = 1
    while len(stationary_drones) < num_stationary_goal:
        print(f"\n{'='*15} ROUND {round_num} (Chained-Auction + A*) {'='*15}")
        print(f"     - Current Gathering Spot: Drone {gathering_spot_node.id} at {np.round(gathering_spot_node.pos, 1)}")
        
        # --- 1. SENSE ---
        frontiers_geom = get_windows(polygon, shared_map)
        if frontiers_geom.is_empty:
            print("--- 1. SENSE: No frontiers found. Mission complete.")
            break
            
        tasks_geom = list(frontiers_geom.geoms) if frontiers_geom.geom_type == 'MultiLineString' else [frontiers_geom]
        task_list_raw = [f for f in tasks_geom if f.length > MIN_FRONTIER_LENGTH]
        
        if not task_list_raw:
            print("--- 1. SENSE: Frontiers found, but all are too small. Mission complete.")
            break

        # --- 1b. Assign Parents to Frontiers ---
        print("     - Assigning parent nodes to frontiers...")
        task_list = [] # This will be our list of dicts
        for frontier in task_list_raw:
            frontier_midpoint_obj = frontier.interpolate(0.5, normalized=True)
            frontier_midpoint = frontier_midpoint_obj.coords[0]
            best_parent = None
            min_parent_dist = float('inf')

            # Check *all* stationary drones to find the one that can see the midpoint
            for s_drone in stationary_drones:
                if s_drone.vis_poly.buffer(TOLERANCE).contains(frontier_midpoint_obj):
                    dist = Point(s_drone.pos).distance(frontier_midpoint_obj)
                    if dist < min_parent_dist:
                        min_parent_dist = dist
                        best_parent = s_drone
            
            if best_parent:
                task_list.append({
                    'frontier': frontier,
                    'parent_drone': best_parent,
                    'target_pos': frontier_midpoint # Store the (x,y) coord
                })
        
        if not task_list:
            print("     - No frontiers could be linked to a parent node.")
            round_num += 1
            continue # Skip to next round

        print(f"--- 1. SENSE: Found {len(task_list)} valid frontiers.")
        update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Found {len(task_list)} Frontiers", frontiers=frontiers_geom)
        time.sleep(0.1)

        # --- 2. EVALUATE (NEW: "Chained Auction" Logic) ---
        print("--- 2. EVALUATE: Drones scanning all frontiers in chained batches.")
        
        broadcast_channel = []
        drones_in_field = [] # Drones that leave the base
        
        shared_map_simple = shared_map.simplify(TOLERANCE, preserve_topology=True)
        try:
            shared_map_simple = shared_map_simple.buffer(0)
        except Exception as e:
            print(f"     Warning: Failed to clean shared_map_simple, continuing with simplified: {e}")
            pass 
        
        batch_num = 1
        while task_list:
            mobile_drones = [d for d in drone_fleet if d.status == 'available']
            if not mobile_drones:
                print("     - No mobile drones available to evaluate (waiting for others).")
                # This should not happen if all drones return at the end
                break
            
            # Track drones that are leaving the base *this round*
            for d in mobile_drones:
                if d.id not in [f.id for f in drones_in_field]:
                    drones_in_field.append(d)
            
            num_drones = len(mobile_drones)
            num_tasks_in_batch = min(num_drones, len(task_list))
            
            tasks_for_this_batch = task_list[:num_tasks_in_batch]
            drones_for_this_batch = mobile_drones[:num_tasks_in_batch]
            
            print(f"     - Processing Batch {batch_num}: {len(drones_for_this_batch)} drones vs {len(tasks_for_this_batch)} tasks.")

            # --- 2b. Build Cost Matrix (A* from Drone's CURRENT pos) ---
            cost_matrix = np.zeros((len(drones_for_this_batch), len(tasks_for_this_batch)))
            animation_paths = {} 
            
            for i in range(len(drones_for_this_batch)):
                drone = drones_for_this_batch[i]
                start_pos = drone.pos # <-- KEY: From drone's current location

                for j in range(len(tasks_for_this_batch)):
                    task = tasks_for_this_batch[j]
                    end_pos = task['target_pos']
                    
                    # --- A* PATHFINDING ---
                    waypoints, cost = get_a_star_waypoint_path(
                        start_pos, end_pos, 
                        delaunator_like_data, a_star_graph, tri_centroids
                    )
                    # ---
                    
                    cost_matrix[i, j] = cost
                    animation_paths[(i, j)] = waypoints

            # 2c. Run the optimized assignment solver
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # 2d. Process the assignments
            animation_data_list_out = []
            assigned_task_indices_in_batch = set()

            for i in range(len(row_ind)):
                drone_index = row_ind[i]
                task_index = col_ind[i]
                
                drone = drones_for_this_batch[drone_index]
                task_data = tasks_for_this_batch[task_index]
                travel_cost = cost_matrix[drone_index, task_index]
                path_waypoints = animation_paths[(drone_index, task_index)]
                
                # This task (by global index) is now assigned
                global_task_index = task_list.index(task_data)
                assigned_task_indices_in_batch.add(global_task_index)
                
                drone.total_distance_traveled += travel_cost
                
                # --- Evaluate gain at the target point ---
                best_point = task_data['target_pos'] # Use the pre-calculated midpoint
                if N_FRONTIER_SAMPLES > 0:
                    # (Optional: Re-run N-sample logic here if needed)
                    pass 
                
                view = get_visibility_from_triangulation(
                    best_point, delaunator_like_data, obstructs
                )
                try:
                    view_clean = view.buffer(0)
                except Exception as e:
                    view_clean = view 
                gain = view_clean.difference(shared_map_simple).area
                # ---
                
                if gain > 0 or N_FRONTIER_SAMPLES == 0: # Store it even if gain is 0
                    broadcast_channel.append({
                        'pos': best_point, 
                        'gain': gain,
                        'parent_drone': task_data['parent_drone']
                    })
                    
                    animation_data_list_out.append({
                        'drone': drone,
                        'out_path_points': path_waypoints,
                    })
                    
                    drone.status = 'exploring' # This drone is now busy
                    drone.current_task_target_pos = best_point # Store where it's going
                    
                    print_gain = f"gain of {gain:.2f}"
                    print_pos = f"at {np.round(best_point,1)}"
                    print_cost = f"Cost: {travel_cost:.2f} units (A*)"
                    print(f"         - D{drone.id} deploying. {print_cost}. Finds {print_gain} {print_pos}")
                
                else:
                    print(f"         - D{drone.id} traveled {travel_cost:.2f} but found no valid gain. Staying put.")
                    drone.status = 'available' # This drone is free for next batch
            
            # --- PARALLEL ANIMATION BLOCK (OUTBOUND) ---
            if animation_data_list_out:
                print("     - Animating drone deployment...")
                animate_drones_parallel(
                    animation_data_list_out, f"Drones deploying (Batch {batch_num})", 
                    path_key='out_path_points',
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                    shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num
                )
                # Finalize drone positions and mark them as 'available' for the *next* batch
                for data in animation_data_list_out:
                    data['drone'].pos = data['out_path_points'][-1]
                    data['drone'].status = 'available' # Ready for next auction
            
            # Remove all assigned tasks from the main list
            # We must iterate backwards to not mess up indices
            for index in sorted(list(assigned_task_indices_in_batch), reverse=True):
                del task_list[index]
            
            batch_num += 1
            update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Batch {batch_num} complete, {len(task_list)} tasks left", frontiers=frontiers_geom, evaluated_tasks=broadcast_channel)
            time.sleep(0.1) 
        
        # --- End of 'while task_list:' loop ---
        # ALL frontiers are evaluated. 
        # All 'drones_in_field' are at their *last* evaluation spot.
        
        # --- NEW: RECALL ALL DRONES to GATHERING SPOT ---
        animation_data_list_in = []
        if drones_in_field:
            print(f"     - Recalling {len(drones_in_field)} drones to gathering spot (D{gathering_spot_node.id})...")
            for drone in drones_in_field:
                
                start_pos = drone.pos
                end_pos = gathering_spot_node.pos
                
                waypoints, cost = get_a_star_waypoint_path(
                    start_pos, end_pos,
                    delaunator_like_data, a_star_graph, tri_centroids
                )
                
                animation_data_list_in.append({
                    'drone': drone,
                    'in_path_points': waypoints
                })
                
                print(f"         - D{drone.id} returning. Cost: {cost:.2f} units (A*).")
                drone.total_distance_traveled += cost
            
            # --- ANIMATE DRONES RETURNING ---
            if animation_data_list_in:
                animate_drones_parallel(
                    animation_data_list_in, f"Drones returning to D{gathering_spot_node.id}", 
                    path_key='in_path_points',
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                    shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num
                )
            
            # --- Reset Drones' State ---
            for drone in drones_in_field:
                drone.status = 'available'
                drone.pos = gathering_spot_node.pos
                drone.current_task_target_pos = None
        # --- *** END RETURN-TO-SPOT BLOCK *** ---

        # --- 3. CONSENSUS PHASE ---
        print("--- 3. CONSENSUS: All drones independently decide the winner.")
        if not broadcast_channel: 
            print("     - No valid tasks found this round. Ending.")
            break
        else:
            best_task = max(broadcast_channel, key=lambda x: x['gain'])
            print(f"     - Consensus reached: Best frontier is at {np.round(best_task['pos'], 2)} with a gain of {best_task['gain']:.2f}")

        # --- 4. UPDATE PHASE ---
        print("--- 4. UPDATE: Re-tasking drone from gathering spot.")
        
        # --- 4a. Get an available drone ---
        mobile_drones = [d for d in drone_fleet if d.status == 'available']
        if not mobile_drones:
             print("     - No mobile drones left to place. Mission complete.")
             break

        # Since all are at the gathering spot, just pick the first one.
        winner_drone = mobile_drones[0]

        target_pos = best_task['pos']
        target_parent = best_task['parent_drone'] 
        
        # --- 4b. Animate the single winner deploying ---
        start_pos = gathering_spot_node.pos
        
        waypoints, deployment_cost = get_a_star_waypoint_path(
            start_pos, target_pos,
            delaunator_like_data, a_star_graph, tri_centroids
        )
        
        print(f"     - D{winner_drone.id} deploying (Cost: {deployment_cost:.2f})...")

        animate_single_drone_path(
            winner_drone, waypoints, f"D{winner_drone.id} deploying to new site",
            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, 
            frontiers=frontiers_geom, round_num=round_num,
            evaluated_tasks=broadcast_channel, winning_task=best_task
        )
        
        # --- 4c. Finalize winner's state ---
        winner_drone.pos = target_pos # Set final pos
        winner_drone.status = 'stationary'
        winner_drone.parent = target_parent # Set tree parent
        winner_drone.current_task_target_pos = None
        stationary_drones.append(winner_drone)
        
        winner_drone.total_distance_traveled += deployment_cost
        
        # --- 4d. MIGRATE BASE: Move all "loser" drones to the new spot ---
        old_gathering_spot_node = gathering_spot_node
        gathering_spot_node = winner_drone # The winner is the new FOB!
        
        print(f"     - Migrating base. All available drones moving to D{gathering_spot_node.id}.")
        
        animation_data_list_migrate = []
        migrating_drones = [d for d in mobile_drones if d.id != winner_drone.id]
        
        if migrating_drones:
            for drone in migrating_drones:
                start_pos = old_gathering_spot_node.pos
                end_pos = gathering_spot_node.pos
                
                waypoints, migration_cost = get_a_star_waypoint_path(
                    start_pos, end_pos,
                    delaunator_like_data, a_star_graph, tri_centroids
                )
                
                animation_data_list_migrate.append({
                    'drone': drone,
                    'out_path_points': waypoints
                })
                
                print(f"         - D{drone.id} migrating to new base. Cost: {migration_cost:.2f} units (A*).")
                drone.total_distance_traveled += migration_cost

            # --- Animate the migration ---
            animate_drones_parallel(
                animation_data_list_migrate, f"Swarm migrating to new base (D{gathering_spot_node.id})", 
                path_key='out_path_points',
                ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num
            )
            
            # --- Set final state for migrating drones ---
            for drone in migrating_drones:
                drone.pos = gathering_spot_node.pos
                drone.status = 'available' 
        
        # --- 4e. Update Map with Winner's View ---
        winner_view = get_visibility_from_triangulation(
            winner_drone.pos, delaunator_like_data, obstructs
        )
        
        winner_drone.vis_poly = winner_view
        shared_map = shared_map.union(winner_view).buffer(0).simplify(TOLERANCE, preserve_topology=True)
        
        update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Drone {winner_drone.id} Placed", frontiers=frontiers_geom, evaluated_tasks=broadcast_channel, winning_task=best_task)
        
        round_num += 1
        time.sleep(0.1)
        
    # --- End of the main while loop ---
        
    simulation_end_time = time.perf_counter() 
    total_runtime = simulation_end_time - simulation_start_time 
        
    print(f"\n{'='*15} MISSION COMPLETE {'='*15}")
    
    total_area = polygon.area
    visible_area = shared_map.area
    coverage_percentage = 0.0
    if total_area > 0:
        coverage_percentage = (visible_area / total_area) * 100
    
    print(f"Algorithm:           Chained-Auction + A* Paths")
    print(f"Total Free Space:    {total_area:,.2f} sq. units")
    print(f"Total Visible Area:  {visible_area:,.2f} sq. units")
    print(f"Final Coverage:      {coverage_percentage:.2f}%")
    print(f"Total Runtime:       {total_runtime:.2f} seconds")

    print(f"\n{'='*15} FINAL TRAVEL LOG {'='*15}")
    total_fleet_travel = 0.0
    for drone in drone_fleet:
        print(f" - Drone {drone.id} (Status: {drone.status}) Traveled: {drone.total_distance_traveled:.2f} units")
        total_fleet_travel += drone.total_distance_traveled
    print(f"--- Total Fleet Travel: {total_fleet_travel:.2f} units ---")
    
    final_title = f"Coverage Complete! (Chained-Auction + A*) ({len(stationary_drones)} Placed) - {coverage_percentage:.2f}% Coverage"
    update_plot(ax, polygon, drone_fleet, shared_map, final_title)
    plt.ioff()
    plt.show()