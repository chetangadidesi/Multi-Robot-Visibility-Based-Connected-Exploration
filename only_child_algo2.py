import math
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon
from shapely.ops import unary_union, nearest_points
import triangle as tr
from typing import List, Tuple, Callable, Dict, Optional
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree # For fast centroid lookups
import heapq 

# ==============================================================================
# CONSTANTS
# ==============================================================================
TOLERANCE = 0.001
NAV_BUFFER = 0.05 # Inflates nav mesh to capture boundary points
N_FRONTIER_SAMPLES = 0 
MIN_FRONTIER_LENGTH = 1.0 
TOTAL_ANIMATION_STEPS = 10 

# ==============================================================================
# ROBUST MATH HELPERS (unchanged)
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
# Type Aliases
# ==============================================================================
PointT = Tuple[float, float]
Segment = Tuple[float, float, float, float]
DelaunatorLike = Dict[str, np.ndarray]
AStarGraph = Dict[int, List[Tuple[int, float]]]

# ==============================================================================
# TRIANGULATION HELPERS
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
    coords = d['coords'].reshape(-1, 2)
    q = (qx, qy)
    num_triangles = len(d['triangles']) // 3
    for t_idx in range(num_triangles):
        p_indices = points_of_tri(d, t_idx)
        p1, p2, p3 = coords[p_indices]
        if orient2d(p1, p2, q) >= 0 and orient2d(p2, p3, q) >= 0 and orient2d(p3, p1, q) >= 0:
            return t_idx
    return -1

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
# TRIANGULAR EXPANSION (VISIBILITY)
# ==============================================================================
def triangular_expansion(d: DelaunatorLike, qx: float, qy: float, obstructs: Callable[[int], bool]) -> List[Segment]:
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
        print(f"       ! Warning: Query point {qx, qy} is outside the VISIBILITY triangulation.")
        return []

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

def get_visibility_from_triangulation(viewpoint: Tuple[float, float], delaunator_data: DelaunatorLike, obstructs_func: Callable[[int], bool]) -> Polygon:
    qx, qy = viewpoint
    visibility_segments = triangular_expansion(delaunator_data, qx, qy, obstructs_func)
    if not visibility_segments: return Polygon()
    visible_triangles = []
    for seg in visibility_segments:
        triangle_coords = [viewpoint, (seg[0], seg[1]), (seg[2], seg[3])]
        visible_triangles.append(Polygon(triangle_coords))
    if not visible_triangles: return Polygon()
    raw_polygon = unary_union(visible_triangles)
    return raw_polygon.buffer(0)

# ==============================================================================
# FRONTIER CALCULATION
# ==============================================================================
def get_windows(polygon, vis_polygon):
    clean_vis_polygon = vis_polygon.buffer(TOLERANCE).buffer(-TOLERANCE)
    if clean_vis_polygon.is_empty: return LineString()
    
    exterior_boundaries = []
    if clean_vis_polygon.geom_type == 'Polygon':
        if not clean_vis_polygon.exterior.is_empty:
            exterior_boundaries.append(clean_vis_polygon.exterior)
    elif clean_vis_polygon.geom_type == 'MultiPolygon':
        for poly in clean_vis_polygon.geoms:
            if not poly.is_empty and not poly.exterior.is_empty:
                exterior_boundaries.append(poly.exterior)
            
    if not exterior_boundaries: return LineString()
    all_exterior_lines = unary_union(exterior_boundaries)
    if all_exterior_lines.is_empty: return LineString()

    environment_lines = polygon.boundary
    buffered_environment = environment_lines.buffer(TOLERANCE)
    window_segments = all_exterior_lines.difference(buffered_environment)
    return window_segments

# ==============================================================================
# DRONE CLASS
# ==============================================================================
class Drone:
    def __init__(self, drone_id, initial_pos):
        self.id = drone_id
        self.pos = initial_pos
        self.status = 'available'
        self.parent = None
        self.vis_poly = None
        self.total_distance_traveled = 0.0
        self.current_task_target_pos = None 

    def __repr__(self):
        return f"Drone {self.id} @ ({self.pos[0]:.2f}, {self.pos[1]:.2f})"

# ==============================================================================
# ANIMATION HELPERS
# ==============================================================================
def path_distance(p1: Tuple, p2: Tuple) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_path_total_length(path_points: List[Tuple]) -> float:
    total_len = 0.0
    for i in range(len(path_points) - 1):
        total_len += path_distance(path_points[i], path_points[i+1])
    return total_len

def get_point_at_distance(path_points: List[Tuple], target_dist: float) -> Tuple[float, float]:
    if not path_points: return (0, 0)
    if target_dist <= 0: return path_points[0]
    total_dist = 0.0
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i+1]
        segment_len = path_distance(p1, p2)
        if total_dist + segment_len >= target_dist - TOLERANCE:
            dist_on_segment = target_dist - total_dist
            t = 0.0
            if segment_len > 0: t = dist_on_segment / segment_len
            return (p1[0] * (1 - t) + p2[0] * t, p1[1] * (1 - t) + p2[1] * t)
        total_dist += segment_len
    return path_points[-1]

# --- (DELETED: get_frontier_key) ---

# ==============================================================================
# A* GRAPH FUNCTIONS (ENHANCED)
# ==============================================================================
def get_triangle_centroid(t_idx: int, coords_flat: np.ndarray, triangles_flat: np.ndarray) -> PointT:
    coords = coords_flat.reshape(-1, 2)
    p_indices = triangles_flat[t_idx*3 : t_idx*3 + 3]
    p1, p2, p3 = coords[p_indices]
    return ((p1[0] + p2[0] + p3[0]) / 3.0, (p1[1] + p2[1] + p3[1]) / 3.0)

def build_a_star_graph(d: DelaunatorLike, obstructs_func: Callable[[int], bool]) -> Tuple[AStarGraph, List[PointT]]:
    graph: AStarGraph = {}
    num_triangles = len(d['triangles']) // 3
    coords_flat = d['coords']
    triangles_flat = d['triangles']
    halfedges = d['halfedges']
    
    centroids = [get_triangle_centroid(i, coords_flat, triangles_flat) for i in range(num_triangles)]
    
    for t_idx in range(num_triangles):
        if t_idx not in graph: graph[t_idx] = []
        for edge_idx in [t_idx * 3, t_idx * 3 + 1, t_idx * 3 + 2]:
            neighbor_edge_idx = halfedges[edge_idx]
            if neighbor_edge_idx != -1 and not obstructs_func(edge_idx):
                neighbor_t_idx = neighbor_edge_idx // 3
                cost = path_distance(centroids[t_idx], centroids[neighbor_t_idx])
                graph[t_idx].append((neighbor_t_idx, cost))
    return graph, centroids

def snap_point_to_polygon(point: PointT, poly: Polygon) -> PointT:
    """Snap a point to the polygon if it's slightly outside."""
    pt_obj = Point(point)
    if poly.contains(pt_obj): return point
    p1, _ = nearest_points(poly, pt_obj)
    return (p1.x, p1.y)

def find_triangle_for_point(d: DelaunatorLike, qx: float, qy: float, centroid_tree: Optional[cKDTree] = None) -> int:
    """Finds triangle. Uses KDTree for O(log n) fallback if point is slightly off."""
    # 1. Exact Geometric Check (Standard)
    coords = d['coords'].reshape(-1, 2)
    q = (qx, qy)
    num_triangles = len(d['triangles']) // 3

    for t_idx in range(num_triangles):
        p_indices = points_of_tri(d, t_idx)
        p1, p2, p3 = coords[p_indices]
        if orient2d(p1, p2, q) >= 0 and orient2d(p2, p3, q) >= 0 and orient2d(p3, p1, q) >= 0:
            return t_idx
    
    # 2. Fast Fallback with KDTree (Optimized)
    if centroid_tree is not None:
        print(f"       ! Warning: Point {q} is outside triangulation. Finding closest centroid (fast).")
        _, closest_t_idx = centroid_tree.query((qx, qy))
        return int(closest_t_idx)

    # 3. Slow Fallback (Legacy)
    print(f"       ! Warning: Point {q} is outside triangulation. Finding closest centroid (SLOW).")
    min_dist = float('inf')
    closest_t_idx = -1
    centroids = [get_triangle_centroid(i, d['coords'], d['triangles']) for i in range(num_triangles)]
    for t_idx, centroid in enumerate(centroids):
        dist = path_distance(q, centroid)
        if dist < min_dist:
            min_dist = dist
            closest_t_idx = t_idx
    return closest_t_idx

def a_star_search(start_t: int, end_t: int, graph: AStarGraph, centroids: List[PointT]) -> Tuple[List[int], float]:
    if start_t == -1 or end_t == -1: return ([], 0.0)
    if start_t == end_t: return ([start_t], 0.0)
            
    end_centroid = centroids[end_t]
    pq: List[Tuple[float, float, int, List[int]]] = []
    heapq.heappush(pq, (0.0, 0.0, start_t, [start_t]))
    visited = set()

    while pq:
        _, cost_g, current_t, path = heapq.heappop(pq)
        if current_t in visited: continue
        visited.add(current_t)
        
        if current_t == end_t: return (path, cost_g)

        for neighbor_t, move_cost in graph.get(current_t, []):
            if neighbor_t not in visited:
                new_cost_g = cost_g + move_cost
                cost_h = path_distance(centroids[neighbor_t], end_centroid)
                new_path = path + [neighbor_t]
                heapq.heappush(pq, (new_cost_g + cost_h, new_cost_g, neighbor_t, new_path))
    
    print(f"       ! A* Error: No path found from {start_t} to {end_t}.")
    return ([], 0.0)

def get_a_star_waypoint_path(
    start_pos: PointT, 
    end_pos: PointT,
    d: DelaunatorLike,
    graph: AStarGraph,
    centroids: List[PointT],
    nav_poly: Polygon,      # <-- NEW
    centroid_tree: cKDTree  # <-- NEW
) -> Tuple[List[PointT], float]:
    
    # --- 1. CLAMP POINTS ---
    start_pos_safe = snap_point_to_polygon(start_pos, nav_poly)
    end_pos_safe = snap_point_to_polygon(end_pos, nav_poly)

    # --- 2. FIND TRIANGLES (Fast) ---
    start_t = find_triangle_for_point(d, start_pos_safe[0], start_pos_safe[1], centroid_tree)
    end_t = find_triangle_for_point(d, end_pos_safe[0], end_pos_safe[1], centroid_tree)
    
    if start_t == -1 or end_t == -1:
        print("       ! A* Warning: Start/End point outside triangulation.")
        return ([start_pos, end_pos], path_distance(start_pos, end_pos))

    triangle_path, _ = a_star_search(start_t, end_t, graph, centroids)
    
    if not triangle_path:
        print("       ! A* Warning: No path found. Returning straight line.")
        return ([start_pos, end_pos], path_distance(start_pos, end_pos))
        
    waypoints = [start_pos]
    start_t_centroid = centroids[triangle_path[0]]
    if not np.array_equal(start_pos, start_t_centroid):
        waypoints.append(start_t_centroid)
    waypoints.extend([centroids[t] for t in triangle_path[1:]]) 
    if not np.array_equal(waypoints[-1], end_pos):
        waypoints.append(end_pos)
    
    total_length = get_path_total_length(waypoints)
    return (waypoints, total_length)

def generate_a_star_graph_from_polygon(
    polygon_to_graph: Polygon
) -> Tuple[DelaunatorLike, AStarGraph, List[PointT], Optional[cKDTree], Polygon]:
    
    print("       - Generating new A* graph from current shared_map...")
    
    try:
        if not polygon_to_graph.is_valid:
            print("       - Warning: Input polygon is not valid. Attempting buffer(0) fix...")
            polygon_to_graph = polygon_to_graph.buffer(0)
        if polygon_to_graph.is_empty:
            print("       ! A* Graph Gen Error: Polygon is empty. Cannot graph.")
            return ({}, {}, [], None, Polygon())
        # print(f"       - Polygon is valid. Type: {polygon_to_graph.geom_type}. Area: {polygon_to_graph.area:.2f}")
    except Exception as e:
        print(f"       ! A* Graph Gen Error: Failed during initial validation: {e}")
        return ({}, {}, [], None, Polygon())

    # --- 1. CREATE NAVIGATION BUFFER ---
    try:
        poly_for_nav = polygon_to_graph.buffer(NAV_BUFFER).simplify(TOLERANCE, preserve_topology=True)
    except Exception as e:
        print(f"       ! Warning: NAV_BUFFER failed: {e}. Using original polygon.")
        poly_for_nav = polygon_to_graph

    if poly_for_nav.is_empty:
        print("       ! A* Graph Gen Error: Polygon became empty after buffer. Cannot graph.")
        return ({}, {}, [], None, Polygon())

    # --- 2. DECONSTRUCTION (On Buffered Poly) ---
    master_vertices_map = {}
    master_segments_set = set()
    master_holes_list = []
    current_vertex_index = 0

    polygons_to_process = []
    if poly_for_nav.geom_type == 'Polygon':
        polygons_to_process = [poly_for_nav]
    elif poly_for_nav.geom_type == 'MultiPolygon':
        polygons_to_process = list(poly_for_nav.geoms)
    else:
        print(f"       ! A* Graph Gen Error: Unknown geometry type: {poly_for_nav.geom_type}")
        return ({}, {}, [], None, poly_for_nav)

    # print(f"       - Deconstructing {len(polygons_to_process)} polygon(s)...")

    try:
        for poly in polygons_to_process:
            if poly.is_empty: continue
            exterior_coords = list(poly.exterior.coords[:-1])
            if not exterior_coords: continue
            segment_indices = []
            for coord in exterior_coords:
                coord_tuple = tuple(coord)
                if coord_tuple not in master_vertices_map:
                    master_vertices_map[coord_tuple] = current_vertex_index
                    current_vertex_index += 1
                segment_indices.append(master_vertices_map[coord_tuple])
            for i in range(len(segment_indices)):
                p1, p2 = segment_indices[i], segment_indices[(i + 1) % len(segment_indices)]
                master_segments_set.add(tuple(sorted((p1, p2))))
            for interior in poly.interiors:
                hole_coords = list(interior.coords[:-1])
                if not hole_coords: continue
                hole_segment_indices = []
                for coord in hole_coords:
                    coord_tuple = tuple(coord)
                    if coord_tuple not in master_vertices_map:
                        master_vertices_map[coord_tuple] = current_vertex_index
                        current_vertex_index += 1
                    hole_segment_indices.append(master_vertices_map[coord_tuple])
                for i in range(len(hole_segment_indices)):
                    p1, p2 = hole_segment_indices[i], hole_segment_indices[(i + 1) % len(hole_segment_indices)]
                    master_segments_set.add(tuple(sorted((p1, p2))))
                hole_poly = Polygon(interior).buffer(0)
                if not hole_poly.is_empty and hole_poly.is_valid:
                    master_holes_list.append(list(hole_poly.representative_point().coords)[0])
        # print("       - Polygon deconstruction complete.")
        final_vertices_list = [None] * len(master_vertices_map)
        for coord, index in master_vertices_map.items():
            final_vertices_list[index] = coord
        vertices_np = np.array(final_vertices_list)
        segments_np = np.array(list(master_segments_set))
        holes_np = np.array(master_holes_list) if master_holes_list else None
    except Exception as e:
        print(f"       ! A* Graph Gen Error: Failed during robust deconstruction: {e}")
        return ({}, {}, [], None, poly_for_nav)

    if len(final_vertices_list) == 0:
        print("       ! A* Graph Gen Error: No vertices found in shared_map.")
        return ({}, {}, [], None, poly_for_nav)

    # 3. Triangulate
    scene = { 'vertices': vertices_np, 'segments': segments_np }
    if holes_np is not None and holes_np.size > 0: scene['holes'] = holes_np
    
    try:
        num_holes = len(holes_np) if holes_np is not None else 0
        print(f"       - Triangulating scene with {len(vertices_np)} vertices, {len(segments_np)} segments, {num_holes} holes...")
        B = tr.triangulate(scene, 'p')
        print("       - Triangulation complete.")
    except Exception as e:
        print(f"       ! A* Graph Gen Error: Triangulation failed: {e}")
        return ({}, {}, [], None, poly_for_nav)
        
    if 'triangles' not in B or len(B['triangles']) == 0:
        print("       ! A* Graph Gen Error: Triangulation produced no triangles.")
        return ({}, {}, [], None, poly_for_nav)
        
    print("       - Building half-edges...")
    halfedges = build_halfedges(B['triangles'])
    d_safe = {
        'coords': B['vertices'].flatten(),
        'triangles': B['triangles'].flatten(),
        'halfedges': halfedges,
    }
    
    # 4. Build A* graph
    print("       - Building A* graph from safe triangulation...")
    obstructs_safe = lambda e: False
    safe_graph, safe_centroids = build_a_star_graph(d_safe, obstructs_safe)
    
    # 5. Build KDTree
    centroid_tree = cKDTree(safe_centroids) if safe_centroids else None
    
    print(f"       - New safe graph generated: {len(safe_centroids)} triangles.")
    return (d_safe, safe_graph, safe_centroids, centroid_tree, poly_for_nav)

# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================
def update_plot(ax, polygon, drone_fleet, shared_map, title_text, frontiers=None, evaluated_tasks=None, winning_task=None):
    ax.clear()
    ox, oy = polygon.exterior.xy
    ax.fill(ox, oy, color='lightgray', zorder=0)
    
    polygons_to_plot = []
    if shared_map.is_empty: pass
    elif shared_map.geom_type == 'Polygon': polygons_to_plot = [shared_map]
    elif shared_map.geom_type == 'MultiPolygon': polygons_to_plot = list(shared_map.geoms)
    
    for i, poly in enumerate(polygons_to_plot):
        if poly.is_empty or poly.geom_type != 'Polygon': continue
        vx, vy = poly.exterior.xy
        ax.fill(vx, vy, color='gold', alpha=0.5, label='Total Coverage' if i == 0 else '_nolegend_', zorder=1)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, color='lightgray', zorder=2)
            
    for interior in polygon.interiors:
        hx, hy = interior.xy
        ax.fill(hx, hy, color='dimgray', zorder=3)
        
    ax.plot(ox, oy, 'k-', linewidth=1, zorder=4)
    for interior in polygon.interiors:
        hx, hy = interior.xy
        ax.plot(hx, hy, 'k-', linewidth=1, zorder=4)
        
    for i, drone in enumerate(drone_fleet):
        if drone.status == 'stationary' and drone.parent:
            p_pos, c_pos = drone.parent.pos, drone.pos
            ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], 'k-', linewidth=1.5, zorder=5, label='Network Link' if i == 1 else '_nolegend_')
            
    if frontiers:
        if frontiers.geom_type == 'MultiLineString':
            for i, frontier in enumerate(frontiers.geoms):
                x, y = frontier.xy
                ax.plot(x, y, 'c--', linewidth=2, label='Frontiers' if i == 0 else '_nolegend_', zorder=6)
        elif frontiers.geom_type == 'LineString':
                 x, y = frontiers.xy
                 ax.plot(x, y, 'c--', linewidth=2, label='Frontiers', zorder=6)
                 
    if evaluated_tasks:
        for task in evaluated_tasks:
            if 'pos' in task:
                ax.text(task['pos'][0], task['pos'][1] + 0.3, f"{task['gain']:.1f}", color='blue', fontsize=9, ha='center', fontweight='bold', zorder=7)
                
    if winning_task and 'pos' in winning_task:
       ax.plot(winning_task['pos'][0], winning_task['pos'][1], 'go', markersize=12, markerfacecolor='none', markeredgewidth=3, label='Winning Task', zorder=8)
       
    for drone in drone_fleet:
        if drone.status == 'stationary': color, marker, size, label = 'red', 'o', 10, 'Stationary Drone'
        elif drone.status == 'exploring': color, marker, size, label = 'blue', 's', 8, 'Exploring Drone'
        else: color, marker, size, label = 'dimgray', 'x', 6, 'Available (Idle) Drone'
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
    plt.pause(0.01)

# ==============================================================================
# ANIMATION FUNCTIONS
# ==============================================================================
def animate_single_drone_path(
    drone: Drone,
    path_points: List[Tuple[float, float]],
    title_suffix: str,
    ax, polygon, drone_fleet, shared_map, frontiers, round_num,
    evaluated_tasks=None, winning_task=None
):
    drone.status = 'exploring'
    if not path_points: return
    total_path_len = get_path_total_length(path_points)
    if total_path_len == 0: return

    for step in range(TOTAL_ANIMATION_STEPS + 1):
        t = step / TOTAL_ANIMATION_STEPS
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
    path_key: str, 
    ax, polygon, drone_fleet, shared_map, frontiers, round_num,
    evaluated_tasks: Optional[List[Dict]] = None,
    winning_task: Optional[Dict] = None
):
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
        print("       - (Animation skipped: No paths to animate)")
        return

    for step in range(TOTAL_ANIMATION_STEPS + 1):
        t_global = step / TOTAL_ANIMATION_STEPS
        current_global_dist = t_global * max_path_length
        for data in animation_data_list:
            if not data['path_points']: continue
            drone = data['drone']
            target_dist = min(current_global_dist, data['total_path_len'])
            drone.pos = get_point_at_distance(data['path_points'], target_dist)
        
        title = f"Round {round_num}: {title_suffix}"
        update_plot(ax, polygon, drone_fleet, shared_map, 
                    title, frontiers=frontiers,
                    evaluated_tasks=evaluated_tasks,
                    winning_task=winning_task)

# ==============================================================================
# --- MAIN SIMULATION BLOCK (MODIFIED) ---
# ==============================================================================
if __name__ == "__main__":
    simulation_start_time = time.perf_counter()
    
    # 1. Define the Environment Polygon
    outer_boundary_complex = [(0, 0), (0, 40), (20, 40), (20, 25), (10, 25), (10, 15), (20, 15), (20, 0), (0, 0)]
    holes_complex = [
        [(2, 2), (2, 12), (8, 12), (8, 2), (2, 2)],
        [(12, 17), (12, 23), (18, 23), (18, 17), (12, 17)],
        [(2, 27), (2, 38), (8, 38), (8, 27), (2, 27)]
    ]
 
    
    holes_cathedral = [[[6.9375, 4.546875], [4.796875, 4.546875], [4.1875, 6.984375], [6.328125, 6.984375], [6.9375, 4.546875]], [[4.03125, 1.171875], [3.2695309999999997, 1.171875], [2.695313, 1.546875], [2.3125, 2.296875], [1.9257810000000002, 3.046875], [1.734375, 4.179687], [1.734375, 5.6875], [1.734375, 7.1875], [1.9257810000000002, 8.3125], [2.3125, 9.0625], [2.695313, 9.8125], [3.2695309999999997, 10.1875], [4.03125, 10.1875], [4.800781, 10.1875], [5.375, 9.8125], [5.75, 9.0625], [6.132813, 8.3125], [6.328125, 7.1875], [6.328125, 5.6875], [6.328125, 4.179687], [6.132813, 3.046875], [5.75, 2.296875], [5.375, 1.546875], [4.800781, 1.171875], [4.03125, 1.171875]], [[4.03125, 0.0], [5.257813, 0.0], [6.195313, 0.4921869999999995], [6.84375, 1.46875], [7.488281000000001, 2.4375], [7.8125, 3.84375], [7.8125, 5.6875], [7.8125, 7.5234369999999995], [7.488281000000001, 8.929687], [6.84375, 9.90625], [6.195313, 10.875], [5.257813, 11.359375], [4.03125, 11.359375], [2.8125, 11.359375], [1.875, 10.875], [1.21875, 9.90625], [0.5703130000000001, 8.929687], [0.25, 7.5234369999999995], [0.25, 5.6875], [0.25, 3.84375], [0.5703130000000001, 2.4375], [1.21875, 1.46875], [1.875, 0.4921869999999995], [2.8125, 0.0], [4.03125, 0.0]], [[0.0, 6.4375], [3.953125, 6.4375], [3.953125, 7.625], [0.0, 7.625], [0.0, 6.4375]], [[518.765625, 909.214844], [503.945313, 909.214844], [495.644531, 900.324219], [496.238281, 883.132813], [504.535156, 872.464844], [518.765625, 873.058594], [529.433594, 883.132813], [528.839844, 900.324219], [518.765625, 909.214844]], [[606.816406, 909.214844], [591.996094, 909.214844], [583.695313, 900.324219], [584.289063, 883.132813], [592.585937, 872.464844], [606.816406, 873.058594], [617.484375, 883.132813], [616.890625, 900.324219], [606.816406, 909.214844]], [[341.957031, 909.214844], [327.136719, 909.214844], [318.835937, 900.324219], [319.429687, 883.132813], [327.726563, 872.464844], [341.957031, 873.058594], [352.625, 883.132813], [352.03125, 900.324219], [341.957031, 909.214844]], [[430.007813, 909.214844], [415.1875, 909.214844], [406.886719, 900.324219], [407.480469, 883.132813], [415.777344, 872.464844], [430.007813, 873.058594], [440.675781, 883.132813], [440.082031, 900.324219], [430.007813, 909.214844]], [[872.382813, 909.214844], [857.5625, 909.214844], [849.265625, 900.324219], [849.859375, 883.132813], [858.152344, 872.464844], [872.382813, 873.058594], [883.050781, 883.132813], [882.457031, 900.324219], [872.382813, 909.214844]], [[960.433594, 909.214844], [945.613281, 909.214844], [937.316406, 900.324219], [937.910156, 883.132813], [946.203125, 872.464844], [960.433594, 873.058594], [971.101563, 883.132813], [970.507813, 900.324219], [960.433594, 909.214844]], [[695.574219, 909.214844], [680.753906, 909.214844], [672.453125, 900.324219], [673.046875, 883.132813], [681.34375, 872.464844], [695.574219, 873.058594], [706.242187, 883.132813], [705.648437, 900.324219], [695.574219, 909.214844]], [[783.625, 909.214844], [768.804687, 909.214844], [760.503906, 900.324219], [761.101563, 883.132813], [769.394531, 872.464844], [783.625, 873.058594], [794.292969, 883.132813], [793.699219, 900.324219], [783.625, 909.214844]], [[1130.523437, 909.214844], [1115.703125, 909.214844], [1107.40625, 900.324219], [1108.0, 883.132813], [1116.292969, 872.464844], [1130.523437, 873.058594], [1141.191406, 883.132813], [1140.597656, 900.324219], [1130.523437, 909.214844]], [[1212.917969, 909.214844], [1198.097656, 909.214844], [1189.796875, 900.324219], [1190.390625, 883.132813], [1198.6875, 872.464844], [1212.917969, 873.058594], [1223.585937, 883.132813], [1222.992187, 900.324219], [1212.917969, 909.214844]], [[1047.425781, 909.214844], [1032.605469, 909.214844], [1024.304687, 900.324219], [1024.898437, 883.132813], [1033.195313, 872.464844], [1047.425781, 873.058594], [1058.09375, 883.132813], [1057.5, 900.324219], [1047.425781, 909.214844]], [[521.652344, 464.183594], [531.730469, 473.511719], [532.320313, 491.550781], [521.652344, 502.125], [507.429687, 502.75], [499.128906, 491.550781], [498.535156, 473.511719], [506.835937, 464.183594], [521.652344, 464.183594]], [[609.695313, 464.183594], [619.769531, 473.511719], [620.363281, 491.550781], [609.695313, 502.125], [595.46875, 502.75], [587.171875, 491.550781], [586.578125, 473.511719], [594.878906, 464.183594], [609.695313, 464.183594]], [[344.859375, 464.183594], [354.9375, 473.511719], [355.527344, 491.550781], [344.859375, 502.125], [330.632813, 502.75], [322.335937, 491.550781], [321.742187, 473.511719], [330.042969, 464.183594], [344.859375, 464.183594]], [[432.902344, 464.183594], [442.976563, 473.511719], [443.570313, 491.550781], [432.902344, 502.125], [418.675781, 502.75], [410.378906, 491.550781], [409.789063, 473.511719], [418.082031, 464.183594], [432.902344, 464.183594]], [[875.238281, 464.183594], [885.3125, 473.511719], [885.90625, 491.550781], [875.238281, 502.125], [861.011719, 502.75], [852.714844, 491.550781], [852.121094, 473.511719], [860.421875, 464.183594], [875.238281, 464.183594]], [[963.28125, 464.183594], [973.355469, 473.511719], [973.949219, 491.550781], [963.28125, 502.125], [949.054687, 502.75], [940.753906, 491.550781], [940.164063, 473.511719], [948.460937, 464.183594], [963.28125, 464.183594]], [[698.445313, 464.183594], [708.519531, 473.511719], [709.113281, 491.550781], [698.445313, 502.125], [684.21875, 502.75], [675.921875, 491.550781], [675.328125, 473.511719], [683.625, 464.183594], [698.445313, 464.183594]], [[786.484375, 464.183594], [796.5625, 473.511719], [797.15625, 491.550781], [786.484375, 502.125], [772.261719, 502.75], [763.964844, 491.550781], [763.371094, 473.511719], [771.667969, 464.183594], [786.484375, 464.183594]], [[1133.351563, 464.183594], [1143.429687, 473.511719], [1144.023437, 491.550781], [1133.351563, 502.125], [1119.128906, 502.75], [1110.832031, 491.550781], [1110.238281, 473.511719], [1118.535156, 464.183594], [1133.351563, 464.183594]], [[1215.738281, 464.183594], [1225.8125, 473.511719], [1226.40625, 491.550781], [1215.738281, 502.125], [1201.511719, 502.75], [1193.21875, 491.550781], [1192.625, 473.511719], [1200.917969, 464.183594], [1215.738281, 464.183594]], [[1050.261719, 464.183594], [1060.339844, 473.511719], [1060.929687, 491.550781], [1050.261719, 502.125], [1036.039063, 502.75], [1027.738281, 491.550781], [1027.144531, 473.511719], [1035.445313, 464.183594], [1050.261719, 464.183594]], [[518.765625, 808.789063], [503.945313, 808.789063], [495.644531, 799.894531], [496.238281, 782.703125], [504.535156, 772.035156], [518.765625, 772.625], [529.433594, 782.703125], [528.839844, 799.894531], [518.765625, 808.789063]], [[606.816406, 808.789063], [591.996094, 808.789063], [583.695313, 799.894531], [584.289063, 782.703125], [592.585937, 772.035156], [606.816406, 772.625], [617.484375, 782.703125], [616.890625, 799.894531], [606.816406, 808.789063]], [[341.957031, 808.789063], [327.136719, 808.789063], [318.835937, 799.894531], [319.429687, 782.703125], [327.726563, 772.035156], [341.957031, 772.625], [352.625, 782.703125], [352.03125, 799.894531], [341.957031, 808.789063]], [[430.007813, 808.789063], [415.1875, 808.789063], [406.886719, 799.894531], [407.480469, 782.703125], [415.777344, 772.035156], [430.007813, 772.625], [440.675781, 782.703125], [440.082031, 799.894531], [430.007813, 808.789063]], [[872.382813, 808.789063], [857.5625, 808.789063], [849.265625, 799.894531], [849.859375, 782.703125], [858.152344, 772.035156], [872.382813, 772.625], [883.050781, 782.703125], [882.457031, 799.894531], [872.382813, 808.789063]], [[960.433594, 808.789063], [945.613281, 808.789063], [937.316406, 799.894531], [937.910156, 782.703125], [946.203125, 772.035156], [960.433594, 772.625], [971.101563, 782.703125], [970.507813, 799.894531], [960.433594, 808.789063]], [[695.574219, 808.789063], [680.753906, 808.789063], [672.453125, 799.894531], [673.046875, 782.703125], [681.34375, 772.035156], [695.574219, 772.625], [706.242187, 782.703125], [705.648437, 799.894531], [695.574219, 808.789063]], [[783.625, 808.789063], [768.804687, 808.789063], [760.503906, 799.894531], [761.101563, 782.703125], [769.394531, 772.035156], [783.625, 772.625], [794.292969, 782.703125], [793.699219, 799.894531], [783.625, 808.789063]], [[1130.523437, 808.789063], [1115.703125, 808.789063], [1107.40625, 799.894531], [1108.0, 782.703125], [1116.292969, 772.035156], [1130.523437, 772.625], [1141.191406, 782.703125], [1140.597656, 799.894531], [1130.523437, 808.789063]], [[1212.917969, 808.789063], [1198.097656, 808.789063], [1189.796875, 799.894531], [1190.390625, 782.703125], [1198.6875, 772.035156], [1212.917969, 772.625], [1223.585937, 782.703125], [1222.992187, 799.894531], [1212.917969, 808.789063]], [[1047.425781, 808.789063], [1032.605469, 808.789063], [1024.304687, 799.894531], [1024.898437, 782.703125], [1033.195313, 772.035156], [1047.425781, 772.625], [1058.09375, 782.703125], [1057.5, 799.894531], [1047.425781, 808.789063]], [[521.652344, 564.257813], [531.730469, 573.585937], [532.320313, 591.625], [521.652344, 602.199219], [507.429687, 602.824219], [499.128906, 591.625], [498.535156, 573.585937], [506.835937, 564.257813], [521.652344, 564.257813]], [[609.695313, 564.257813], [619.769531, 573.585937], [620.363281, 591.625], [609.695313, 602.199219], [595.46875, 602.824219], [587.171875, 591.625], [586.578125, 573.585937], [594.878906, 564.257813], [609.695313, 564.257813]], [[344.859375, 564.257813], [354.9375, 573.585937], [355.527344, 591.625], [344.859375, 602.199219], [330.632813, 602.824219], [322.335937, 591.625], [321.742187, 573.585937], [330.042969, 564.257813], [344.859375, 564.257813]], [[432.902344, 564.257813], [442.976563, 573.585937], [443.570313, 591.625], [432.902344, 602.199219], [418.675781, 602.824219], [410.378906, 591.625], [409.789063, 573.585937], [418.082031, 564.257813], [432.902344, 564.257813]], [[875.238281, 564.257813], [885.3125, 573.585937], [885.90625, 591.625], [875.238281, 602.199219], [861.011719, 602.824219], [852.714844, 591.625], [852.121094, 573.585937], [860.421875, 564.257813], [875.238281, 564.257813]], [[963.28125, 564.257813], [973.355469, 573.585937], [973.949219, 591.625], [963.28125, 602.199219], [949.054687, 602.824219], [940.753906, 591.625], [940.164063, 573.585937], [948.460937, 564.257813], [963.28125, 564.257813]], [[698.445313, 564.257813], [708.519531, 573.585937], [709.113281, 591.625], [698.445313, 602.199219], [684.21875, 602.824219], [675.921875, 591.625], [675.328125, 573.585937], [683.625, 564.257813], [698.445313, 564.257813]], [[786.484375, 564.257813], [796.5625, 573.585937], [797.15625, 591.625], [786.484375, 602.199219], [772.261719, 602.824219], [763.964844, 591.625], [763.371094, 573.585937], [771.667969, 564.257813], [786.484375, 564.257813]], [[1133.351563, 564.257813], [1143.429687, 573.585937], [1144.023437, 591.625], [1133.351563, 602.199219], [1119.128906, 602.824219], [1110.832031, 591.625], [1110.238281, 573.585937], [1118.535156, 564.257813], [1133.351563, 564.257813]], [[1215.738281, 564.257813], [1225.8125, 573.585937], [1226.40625, 591.625], [1215.738281, 602.199219], [1201.511719, 602.824219], [1193.21875, 591.625], [1192.625, 573.585937], [1200.917969, 564.257813], [1215.738281, 564.257813]], [[1050.261719, 564.257813], [1060.339844, 573.585937], [1060.929687, 591.625], [1050.261719, 602.199219], [1036.039063, 602.824219], [1027.738281, 591.625], [1027.144531, 573.585937], [1035.445313, 564.257813], [1050.261719, 564.257813]], [[1494.890625, 818.917969], [1466.589844, 818.917969], [1445.363281, 792.972656], [1445.363281, 761.136719], [1466.589844, 735.785156], [1494.300781, 736.375], [1519.652344, 761.136719], [1519.652344, 792.972656], [1494.890625, 818.917969]], [[1325.476563, 818.917969], [1297.175781, 818.917969], [1272.414063, 792.972656], [1272.414063, 761.136719], [1297.765625, 736.375], [1325.476563, 735.785156], [1346.703125, 761.136719], [1346.703125, 792.972656], [1325.476563, 818.917969]], [[1494.894531, 547.503906], [1519.660156, 573.445313], [1519.660156, 605.28125], [1494.304687, 630.042969], [1466.597656, 630.636719], [1445.371094, 605.28125], [1445.371094, 573.445313], [1466.597656, 547.503906], [1494.894531, 547.503906]], [[1272.414063, 573.445313], [1297.175781, 547.503906], [1325.476563, 547.503906], [1346.703125, 573.445313], [1346.703125, 605.28125], [1325.476563, 630.636719], [1297.765625, 630.042969], [1272.414063, 605.28125], [1272.414063, 573.445313]], [[1397.355469, 1178.621094], [1384.5625, 1178.621094], [1377.402344, 1170.851563], [1377.914063, 1155.824219], [1385.078125, 1146.5], [1397.355469, 1147.015625], [1406.5625, 1155.824219], [1406.050781, 1170.851563], [1397.355469, 1178.621094]], [[1486.484375, 1180.425781], [1471.664063, 1180.425781], [1463.363281, 1171.535156], [1463.957031, 1154.347656], [1472.257813, 1143.675781], [1486.484375, 1144.269531], [1497.152344, 1154.347656], [1496.5625, 1171.535156], [1486.484375, 1180.425781]], [[1313.917969, 1180.425781], [1299.097656, 1180.425781], [1290.796875, 1171.535156], [1291.390625, 1154.347656], [1299.691406, 1143.675781], [1313.917969, 1144.269531], [1324.585937, 1154.347656], [1323.996094, 1171.535156], [1313.917969, 1180.425781]], [[1312.492187, 1084.933594], [1299.703125, 1084.933594], [1292.542969, 1077.167969], [1293.050781, 1062.136719], [1300.214844, 1052.8125], [1312.492187, 1053.328125], [1321.699219, 1062.136719], [1321.191406, 1077.167969], [1312.492187, 1084.933594]], [[1312.492187, 988.046875], [1299.703125, 988.046875], [1292.542969, 980.273437], [1293.050781, 965.25], [1300.214844, 955.921875], [1312.492187, 956.441406], [1321.699219, 965.25], [1321.191406, 980.273437], [1312.492187, 988.046875]], [[1489.28125, 1084.9375], [1476.488281, 1084.9375], [1467.792969, 1077.167969], [1467.28125, 1062.136719], [1476.488281, 1053.328125], [1488.769531, 1052.8125], [1495.933594, 1062.136719], [1496.445313, 1077.167969], [1489.28125, 1084.9375]], [[1489.28125, 988.046875], [1476.488281, 988.046875], [1467.792969, 980.273437], [1467.28125, 965.246094], [1476.488281, 956.4375], [1488.769531, 955.921875], [1495.933594, 965.246094], [1496.445313, 980.273437], [1489.28125, 988.046875]], [[1314.753906, 898.183594], [1296.484375, 897.351563], [1290.257813, 892.367187], [1290.257813, 873.6875], [1296.902344, 852.929687], [1315.996094, 852.929687], [1325.546875, 882.402344], [1314.753906, 898.183594]], [[1493.582031, 897.351563], [1475.3125, 898.183594], [1464.515625, 882.402344], [1474.0625, 852.929687], [1493.164063, 852.929687], [1499.808594, 873.6875], [1499.808594, 892.367187], [1493.582031, 897.351563]], [[1398.773437, 186.34375], [1407.46875, 194.113281], [1407.980469, 209.140625], [1398.773437, 217.949219], [1386.496094, 218.46875], [1379.332031, 209.140625], [1378.820313, 194.113281], [1385.980469, 186.34375], [1398.773437, 186.34375]], [[1487.902344, 184.539063], [1497.980469, 193.429687], [1498.570313, 210.621094], [1487.902344, 220.695313], [1473.675781, 221.289063], [1465.375, 210.621094], [1464.78125, 193.429687], [1473.082031, 184.539063], [1487.902344, 184.539063]], [[1315.335937, 184.539063], [1325.414063, 193.429687], [1326.003906, 210.621094], [1315.335937, 220.695313], [1301.109375, 221.289063], [1292.808594, 210.621094], [1292.214844, 193.429687], [1300.515625, 184.539063], [1315.335937, 184.539063]], [[1313.910156, 280.03125], [1322.609375, 287.800781], [1323.117187, 302.828125], [1313.910156, 311.636719], [1301.632813, 312.152344], [1294.46875, 302.828125], [1293.960937, 287.800781], [1301.121094, 280.03125], [1313.910156, 280.03125]], [[1313.910156, 376.917969], [1322.609375, 384.691406], [1323.117187, 399.71875], [1313.910156, 408.527344], [1301.632813, 409.042969], [1294.46875, 399.71875], [1293.960937, 384.691406], [1301.121094, 376.917969], [1313.910156, 376.917969]], [[1469.210937, 287.796875], [1477.910156, 280.027344], [1490.699219, 280.027344], [1497.863281, 287.796875], [1497.351563, 302.828125], [1490.1875, 312.152344], [1477.910156, 311.636719], [1468.699219, 302.828125], [1469.210937, 287.796875]], [[1469.210937, 384.691406], [1477.910156, 376.921875], [1490.699219, 376.921875], [1497.863281, 384.691406], [1497.351563, 399.71875], [1490.1875, 409.042969], [1477.910156, 408.527344], [1468.699219, 399.71875], [1469.210937, 384.691406]], [[1316.171875, 466.78125], [1326.964844, 482.5625], [1317.414063, 512.0390629999999], [1298.320313, 512.0390629999999], [1291.675781, 491.277344], [1291.675781, 472.597656], [1297.902344, 467.613281], [1316.171875, 466.78125]], [[1465.933594, 482.5625], [1476.730469, 466.78125], [1495.0, 467.613281], [1501.226563, 472.597656], [1501.226563, 491.277344], [1494.582031, 512.0351559999999], [1475.480469, 512.0351559999999], [1465.933594, 482.5625]], [[1585.738281, 802.679687], [1572.945313, 802.679687], [1564.25, 794.910156], [1563.738281, 779.882813], [1572.945313, 771.074219], [1585.226563, 770.554687], [1592.386719, 779.882813], [1592.902344, 794.910156], [1585.738281, 802.679687]], [[1665.59375, 798.414063], [1654.710937, 798.414063], [1646.730469, 789.703125], [1646.730469, 774.460937], [1656.886719, 767.207031], [1665.59375, 767.207031], [1674.304687, 776.640625], [1674.304687, 787.523437], [1665.59375, 798.414063]], [[1712.042969, 775.1875], [1724.375, 775.914063], [1723.652344, 790.429687], [1711.3125, 791.152344], [1712.042969, 775.1875]], [[1744.699219, 773.011719], [1756.308594, 767.207031], [1766.46875, 779.542969], [1748.328125, 790.429687], [1744.699219, 773.011719]], [[1780.253906, 746.160156], [1786.785156, 741.082031], [1802.027344, 749.066406], [1789.691406, 761.398437], [1780.253906, 746.160156]], [[1800.578125, 708.425781], [1815.089844, 708.425781], [1815.089844, 720.035156], [1799.125, 719.308594], [1800.578125, 708.425781]], [[1567.765625, 580.792969], [1576.464844, 573.019531], [1589.253906, 573.019531], [1596.417969, 580.792969], [1595.90625, 595.816406], [1588.746094, 605.144531], [1576.464844, 604.625], [1567.253906, 595.816406], [1567.765625, 580.792969]], [[1669.128906, 577.289063], [1677.835937, 588.171875], [1677.835937, 599.058594], [1669.128906, 608.492187], [1660.417969, 608.492187], [1650.253906, 601.234375], [1650.253906, 585.996094], [1658.238281, 577.289063], [1669.128906, 577.289063]], [[1712.675781, 601.960937], [1711.949219, 585.996094], [1724.292969, 586.722656], [1725.015625, 601.234375], [1712.675781, 601.960937]], [[1751.144531, 607.765625], [1754.773437, 590.351563], [1772.917969, 601.234375], [1762.757813, 613.574219], [1751.144531, 607.765625]], [[1785.984375, 636.796875], [1795.417969, 621.554687], [1807.757813, 633.890625], [1792.515625, 641.871094], [1785.984375, 636.796875]], [[1799.773437, 675.984375], [1798.320313, 665.09375], [1814.289063, 664.371094], [1814.289063, 675.984375], [1799.773437, 675.984375]]]
    outer_boundary3 = [[925.929687, 357.847656], [903.554687, 357.847656], [903.554687, 382.414063], [877.109375, 402.4375], [859.742187, 402.4375], [837.839844, 377.089844], [837.839844, 357.847656], [811.234375, 357.6875], [811.078125, 382.570313], [793.085937, 402.28125], [768.207031, 402.28125], [749.117187, 377.40625], [749.117187, 358.003906], [722.675781, 358.003906], [722.359375, 388.203125], [703.898437, 402.910156], [683.558594, 402.753906], [678.082031, 397.121094], [678.550781, 372.082031], [674.011719, 372.082031], [673.855469, 357.53125], [629.105469, 357.6875], [629.105469, 367.230469], [620.65625, 367.230469], [620.1875, 397.121094], [615.492187, 402.910156], [593.433594, 402.4375], [576.84375, 392.738281], [577.488281, 357.859375], [550.363281, 357.859375], [550.957031, 381.441406], [536.808594, 397.949219], [500.25, 397.359375], [486.6875, 380.261719], [487.277344, 356.675781], [460.160156, 356.675781], [460.160156, 382.621094], [445.417969, 399.71875], [414.171875, 398.539063], [397.070313, 383.207031], [397.660156, 357.859375], [369.363281, 357.859375], [369.363281, 382.621094], [355.210937, 399.71875], [320.425781, 399.71875], [305.09375, 375.546875], [305.6875, 363.753906], [305.6875, 358.449219], [281.511719, 358.449219], [281.511719, 383.800781], [270.898437, 395.59375], [249.085937, 395.59375], [249.085937, 403.84375], [202.507813, 403.84375], [202.507813, 396.769531], [225.503906, 395.59375], [226.089844, 384.390625], [219.015625, 372.597656], [197.789063, 372.597656], [183.640625, 386.75], [184.226563, 410.332031], [193.664063, 420.351563], [242.488281, 421.804687], [243.808594, 485.394531], [223.941406, 485.394531], [223.941406, 429.089844], [175.585937, 430.417969], [175.585937, 392.660156], [170.289063, 392.0], [165.648437, 358.21875], [154.390625, 358.21875], [145.78125, 392.660156], [117.960937, 392.660156], [108.683594, 357.554687], [99.414063, 357.554687], [84.324219, 391.851563], [85.5, 391.335937], [84.839844, 432.402344], [59.007813, 432.402344], [28.542969, 444.988281], [28.542969, 454.921875], [55.035156, 465.519531], [56.359375, 497.980469], [29.203125, 506.589844], [29.203125, 517.1875], [57.683594, 531.09375], [86.167969, 531.09375], [86.167969, 558.253906], [94.777344, 558.253906], [95.4375, 575.472656], [173.597656, 575.472656], [172.9375, 559.578125], [178.898437, 558.917969], [178.234375, 525.796875], [223.941406, 526.460937], [225.265625, 512.5507809999999], [243.808594, 513.2109370000001], [243.808594, 570.175781], [254.40625, 576.800781], [254.40625, 594.019531], [245.796875, 611.242187], [220.625, 610.582031], [208.707031, 599.984375], [209.367187, 588.058594], [150.414063, 588.058594], [150.414063, 599.320313], [142.464844, 611.242187], [125.910156, 610.582031], [115.972656, 603.296875], [115.3125, 593.359375], [76.230469, 593.359375], [76.230469, 583.421875], [60.335937, 584.089844], [47.085937, 613.230469], [19.925781, 613.230469], [19.265625, 665.558594], [40.460937, 667.546875], [52.386719, 683.445313], [52.386719, 697.351563], [39.800781, 716.5625], [21.253906, 717.886719], [19.925781, 770.214844], [41.789063, 770.214844], [53.707031, 790.746094], [89.476563, 790.085937], [90.140625, 820.550781], [108.023437, 820.550781], [108.023437, 792.734375], [123.921875, 770.214844], [140.480469, 770.214844], [159.027344, 791.410156], [158.363281, 821.878906], [172.277344, 822.539063], [172.277344, 790.085937], [198.769531, 789.421875], [214.667969, 770.214844], [238.511719, 770.214844], [249.773437, 782.136719], [250.4375, 802.671875], [241.160156, 807.972656], [241.160156, 868.246094], [211.355469, 868.90625], [211.355469, 854.332031], [175.585937, 855.0], [176.910156, 831.816406], [81.53125, 831.152344], [81.53125, 854.332031], [57.683594, 854.332031], [22.578125, 867.582031], [22.578125, 880.167969], [53.046875, 890.765625], [53.046875, 920.574219], [20.589844, 931.832031], [21.253906, 942.429687], [52.386719, 957.003906], [76.890625, 958.328125], [77.554687, 987.472656], [86.828125, 987.472656], [97.425781, 1021.917969], [102.726563, 1021.253906], [112.0, 987.472656], [139.816406, 986.808594], [153.730469, 1021.253906], [171.613281, 986.148437], [171.613281, 951.707031], [210.027344, 952.367187], [210.691406, 941.769531], [224.601563, 942.429687], [225.265625, 896.0625], [237.851563, 896.0625], [236.523437, 960.316406], [200.757813, 959.652344], [179.5625, 972.238281], [178.898437, 991.445313], [193.472656, 1008.003906], [210.027344, 1008.003906], [221.953125, 1000.722656], [221.953125, 985.488281], [200.757813, 984.824219], [201.417969, 977.539063], [239.839844, 976.871094], [239.839844, 986.808594], [274.28125, 986.808594], [274.28125, 1017.28125], [302.101563, 1016.617187], [302.761719, 996.085937], [320.648437, 980.1875], [349.128906, 980.847656], [366.347656, 994.097656], [367.011719, 1016.617187], [390.859375, 1016.617187], [390.859375, 992.773437], [412.054687, 976.871094], [433.910156, 978.199219], [453.785156, 994.097656], [451.796875, 1016.617187], [480.277344, 1015.953125], [480.941406, 995.421875], [504.125, 977.539063], [527.308594, 977.539063], [541.882813, 990.785156], [541.214844, 1014.628906], [569.035156, 1013.96875], [570.363281, 991.445313], [584.933594, 976.210937], [614.078125, 976.210937], [630.640625, 990.785156], [630.640625, 1011.980469], [654.484375, 1012.640625], [655.804687, 990.785156], [675.679687, 974.890625], [700.851563, 974.890625], [720.71875, 990.785156], [721.386719, 1011.320313], [745.890625, 1012.640625], [745.890625, 990.125], [765.097656, 974.890625], [788.28125, 974.890625], [809.476563, 988.796875], [808.15625, 1011.320313], [835.308594, 1011.320313], [835.308594, 987.472656], [852.535156, 973.5625], [878.367187, 973.5625], [895.585937, 984.160156], [894.925781, 1011.980469], [923.40625, 1012.640625], [924.734375, 988.136719], [941.953125, 971.574219], [963.148437, 970.25], [975.074219, 981.511719], [975.074219, 1005.355469], [1020.777344, 1006.019531], [1022.761719, 980.847656], [1036.007813, 970.25], [1057.207031, 970.25], [1072.441406, 985.488281], [1073.105469, 1009.992187], [1094.960937, 1009.992187], [1095.625, 992.109375], [1109.535156, 970.25], [1185.707031, 970.914063], [1196.96875, 961.640625], [1216.179687, 961.640625], [1227.4375, 970.914063], [1227.4375, 985.488281], [1212.863281, 985.488281], [1212.203125, 1002.707031], [1169.148437, 1002.042969], [1146.628906, 1021.253906], [1147.289063, 1043.113281], [1157.226563, 1057.683594], [1183.058594, 1058.347656], [1195.644531, 1049.074219], [1195.644531, 1038.476563], [1175.109375, 1037.8125], [1175.109375, 1033.175781], [1211.542969, 1032.515625], [1212.203125, 1050.398437], [1222.140625, 1056.359375], [1220.8125, 1076.894531], [1194.980469, 1098.089844], [1182.398437, 1097.429687], [1183.058594, 1132.53125], [1208.890625, 1133.195313], [1222.140625, 1144.457031], [1222.800781, 1166.972656], [1200.277344, 1201.421875], [1184.382813, 1201.421875], [1184.382813, 1236.523437], [1210.214844, 1237.851563], [1240.023437, 1271.632813], [1239.359375, 1295.476563], [1273.140625, 1296.140625], [1273.804687, 1263.019531], [1303.609375, 1251.761719], [1328.121094, 1260.371094], [1328.121094, 1280.902344], [1365.210937, 1281.566406], [1365.875, 1262.359375], [1379.121094, 1252.421875], [1402.964844, 1252.421875], [1416.210937, 1263.683594], [1416.210937, 1281.566406], [1453.96875, 1282.230469], [1455.957031, 1268.316406], [1469.203125, 1253.75], [1489.738281, 1254.410156], [1516.234375, 1265.007813], [1515.570313, 1293.488281], [1550.675781, 1294.816406], [1551.335937, 1262.359375], [1571.871094, 1244.472656], [1606.980469, 1243.8125], [1614.925781, 1262.359375], [1630.824219, 1251.761719], [1615.589844, 1230.566406], [1622.210937, 1221.953125], [1646.058594, 1221.953125], [1646.058594, 1199.433594], [1622.210937, 1198.769531], [1621.550781, 1187.507813], [1632.148437, 1166.972656], [1616.910156, 1155.054687], [1601.679687, 1174.261719], [1563.921875, 1174.925781], [1563.261719, 1158.363281], [1577.171875, 1131.871094], [1595.054687, 1132.53125], [1595.054687, 1104.714844], [1581.144531, 1104.050781], [1567.234375, 1092.128906], [1566.574219, 1055.035156], [1603.664063, 1054.371094], [1615.589844, 1077.554687], [1635.460937, 1064.308594], [1622.875, 1044.433594], [1633.472656, 1033.175781], [1650.691406, 1033.175781], [1651.359375, 1009.332031], [1626.847656, 1008.667969], [1624.859375, 996.085937], [1634.796875, 976.210937], [1616.910156, 966.273437], [1603.664063, 990.125], [1567.898437, 989.457031], [1567.898437, 963.625], [1594.394531, 941.769531], [1595.054687, 924.546875], [1577.171875, 919.910156], [1567.234375, 894.742187], [1566.574219, 871.558594], [1590.417969, 871.558594], [1612.941406, 877.515625], [1612.277344, 903.351563], [1637.445313, 901.363281], [1650.03125, 866.917969], [1675.203125, 866.917969], [1674.542969, 906.664063], [1655.992187, 919.246094], [1665.929687, 937.792969], [1681.824219, 924.546875], [1691.761719, 933.160156], [1690.4375, 959.652344], [1709.644531, 959.652344], [1709.644531, 929.84375], [1717.59375, 919.910156], [1733.492187, 933.160156], [1740.777344, 915.273437], [1722.894531, 906.0], [1722.894531, 867.582031], [1750.714844, 867.582031], [1765.285156, 893.414063], [1781.183594, 886.792969], [1781.84375, 859.632813], [1794.429687, 844.402344], [1810.988281, 845.0625], [1828.210937, 876.855469], [1816.289063, 892.753906], [1826.222656, 902.027344], [1841.460937, 882.15625], [1855.367187, 883.480469], [1864.644531, 897.390625], [1880.539063, 884.144531], [1864.644531, 864.933594], [1868.617187, 857.648437], [1888.488281, 848.375], [1885.171875, 836.453125], [1862.65625, 840.425781], [1839.472656, 815.917969], [1840.132813, 802.671875], [1858.679687, 786.773437], [1881.203125, 792.734375], [1893.785156, 768.226563], [1883.851563, 763.589844], [1876.5625, 751.667969], [1877.226563, 721.863281], [1925.582031, 721.195313], [1925.582031, 743.058594], [1941.476563, 742.390625], [1949.425781, 722.523437], [1983.867187, 722.523437], [2001.753906, 702.652344], [2018.976563, 703.3125], [2019.265625, 682.171875], [2001.65625, 682.171875], [1983.441406, 656.0625], [1945.792969, 655.457031], [1938.507813, 637.84375], [1926.363281, 637.84375], [1926.011719, 656.242187], [1875.773437, 654.527344], [1876.34375, 634.546875], [1884.335937, 617.421875], [1895.183594, 610.566406], [1886.050781, 587.734375], [1851.796875, 588.875], [1844.945313, 570.605469], [1858.644531, 552.910156], [1881.480469, 552.910156], [1886.621094, 547.773437], [1886.050781, 534.636719], [1869.496094, 527.21875], [1864.925781, 516.3710940000001], [1882.625, 500.957031], [1876.34375, 489.539063], [1864.355469, 483.832031], [1863.785156, 492.964844], [1851.226563, 500.957031], [1840.378906, 491.253906], [1840.378906, 480.976563], [1831.816406, 481.546875], [1823.820313, 486.113281], [1822.679687, 519.796875], [1812.402344, 534.636719], [1800.414063, 534.636719], [1787.855469, 519.2226559999999], [1788.425781, 495.25], [1779.292969, 485.542969], [1761.59375, 485.542969], [1761.027344, 495.816406], [1751.886719, 510.664063], [1733.050781, 510.664063], [1724.484375, 500.386719], [1724.484375, 476.980469], [1744.46875, 467.273437], [1742.753906, 449.574219], [1737.617187, 443.296875], [1721.0625, 454.714844], [1712.5, 447.863281], [1713.640625, 425.027344], [1695.371094, 425.027344], [1694.800781, 449.007813], [1684.523437, 453.570313], [1669.683594, 439.871094], [1664.542969, 445.011719], [1661.117187, 459.851563], [1677.675781, 470.699219], [1677.675781, 496.390625], [1671.394531, 506.09375], [1653.695313, 505.523437], [1643.417969, 493.539063], [1643.417969, 473.554687], [1613.734375, 472.984375], [1613.164063, 494.105469], [1594.894531, 502.671875], [1575.480469, 502.097656], [1568.632813, 486.683594], [1578.339844, 460.425781], [1587.472656, 453.003906], [1601.746094, 454.144531], [1601.746094, 426.742187], [1586.332031, 427.3125], [1572.628906, 411.898437], [1573.203125, 389.632813], [1578.910156, 379.359375], [1604.597656, 379.925781], [1617.730469, 399.335937], [1638.855469, 387.347656], [1626.292969, 368.507813], [1635.425781, 357.09375], [1656.550781, 357.09375], [1657.121094, 333.683594], [1633.144531, 333.683594], [1628.003906, 321.128906], [1639.421875, 300.574219], [1626.292969, 289.726563], [1606.308594, 313.703125], [1581.191406, 314.847656], [1574.34375, 305.714844], [1574.34375, 287.441406], [1584.046875, 264.03515600000003], [1602.316406, 264.03515600000003], [1602.886719, 232.066406], [1584.046875, 232.636719], [1573.203125, 212.65625], [1572.628906, 192.675781], [1576.625, 185.824219], [1606.882813, 185.824219], [1621.726563, 209.230469], [1641.707031, 196.671875], [1629.148437, 172.125], [1639.996094, 157.277344], [1656.550781, 156.710937], [1656.550781, 139.582031], [1634.859375, 138.441406], [1629.148437, 132.160156], [1641.136719, 108.753906], [1626.863281, 100.191406], [1608.027344, 120.171875], [1582.078125, 120.226563], [1555.667969, 87.601563], [1555.667969, 61.972656], [1523.042969, 61.972656], [1522.65625, 80.609375], [1514.113281, 96.535156], [1490.035156, 104.691406], [1475.664063, 105.082031], [1467.507813, 97.699219], [1466.730469, 63.132813], [1413.519531, 63.523437], [1413.910156, 92.261719], [1406.140625, 106.632813], [1386.335937, 106.632813], [1375.460937, 98.089844], [1375.460937, 63.914063], [1326.136719, 63.914063], [1326.136719, 100.03125], [1319.144531, 106.632813], [1306.71875, 106.632813], [1284.96875, 100.03125], [1276.421875, 86.050781], [1276.421875, 66.632813], [1240.304687, 65.855469], [1240.695313, 92.261719], [1218.585937, 111.828125], [1218.585937, 126.875], [1205.3125, 130.855469], [1181.855469, 130.855469], [1181.855469, 167.148437], [1204.867187, 166.707031], [1223.011719, 187.503906], [1222.570313, 216.714844], [1204.867187, 232.644531], [1182.296875, 232.644531], [1181.855469, 266.72265600000003], [1205.3125, 266.28125], [1214.164063, 270.707031], [1225.667969, 288.851563], [1225.226563, 304.78125], [1217.703125, 314.078125], [1217.261719, 384.445313], [1225.226563, 384.445313], [1224.78125, 409.671875], [1206.636719, 409.671875], [1206.636719, 404.804687], [1194.246094, 404.804687], [1194.246094, 398.167969], [1180.527344, 398.609375], [1180.96875, 344.613281], [1188.933594, 344.613281], [1188.933594, 374.265625], [1206.195313, 361.875], [1206.636719, 338.859375], [1191.589844, 326.914063], [1180.96875, 326.914063], [1171.675781, 329.566406], [1161.050781, 344.613281], [1160.609375, 397.722656], [1148.21875, 397.722656], [1143.792969, 402.589844], [1111.816406, 402.910156], [1098.828125, 387.886719], [1098.984375, 357.847656], [1076.453125, 357.847656], [1076.296875, 387.730469], [1058.617187, 402.4375], [1028.730469, 402.4375], [1010.265625, 383.195313], [1010.265625, 358.003906], [987.894531, 357.847656], [987.734375, 382.878906], [970.050781, 402.28125], [944.550781, 402.4375], [925.929687, 382.414063], [925.929687, 357.847656]]
    
    
    polygon = Polygon(outer_boundary3, holes_cathedral)
    start_pos = (125, 500)
        
    polygon = polygon.buffer(0)
    total_free_area = polygon.area
    
    if polygon.is_empty:
        raise ValueError("Polygon is empty after cleaning, check input geometry.")
        
    print(f"--- Using start_pos: {start_pos} ---")
    fleet_size = 10
    num_stationary_goal = 10
    # --- (DELETED: frontier_cache) ---
    
    print("--- PRE-COMPUTATION: Triangulating Environment (for Visibility) ---")
    
    # ... (Triangulation setup for visibility) ...
    vertices = []
    segments = []
    holes_list = [] 
    
    if polygon.geom_type == 'Polygon': polygons_to_process = [polygon]
    elif polygon.geom_type == 'MultiPolygon': polygons_to_process = list(polygon.geoms)
    else: polygons_to_process = []
    
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
            hole_poly = Polygon(interior).buffer(0)
            if not hole_poly.is_empty:
                holes_list.append(list(hole_poly.representative_point().coords)[0])
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
    print("--- Simulation Start (Method: OC-Sensing + A* Hybrid-Movement) ---")
    
    # 6. Initialize Drone Fleet
    drone_fleet = [Drone(i, start_pos) for i in range(fleet_size)]
    first_drone = drone_fleet[0]
    first_drone.status = 'stationary'
    first_drone.vis_poly = get_visibility_from_triangulation(
        first_drone.pos, delaunator_like_data, obstructs
    )
    
    stationary_drones = [first_drone]
    
    # --- DEFERRED UPDATE LOGIC ---
    shared_map = first_drone.vis_poly
    pending_map_update_view = None # Start with no pending updates
    # ---
    
    root_node = first_drone
    gathering_spot_node = root_node 
    
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
        print(f"\n{'='*15} ROUND {round_num} (OC-Sensing + A* Hybrid) {'='*15}")
        print(f"       - Gathering Spot: D{gathering_spot_node.id} at {np.round(gathering_spot_node.pos, 1)}")
        
        # --- PROCESS DEFERRED UPDATES (for smooth animation) ---
        if pending_map_update_view is not None:
            print("       - Updating shared_map with last round's view...")
            shared_map = shared_map.union(pending_map_update_view).buffer(0).simplify(TOLERANCE, preserve_topology=True)
            
            # --- (DELETED: Cache Invalidation) ---
            
            pending_map_update_view = None # Clear the flag
            
            update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Map Updated, Calculating...", frontiers=None)
            print("       - Map updated. Generating nav graph...")
        # ---
        
        # --- 1. SENSE ---
        # --- 1a. Build Navigation Graph (This is a slow step) ---
        try:
            shared_map_simple = shared_map.simplify(TOLERANCE, preserve_topology=True)
            shared_map_clean = shared_map_simple.buffer(0)
        except Exception as e:
            print(f"       ! Warning: buffer(0) failed: {e}. Using simplified map directly.")
            shared_map_clean = shared_map
        if not shared_map_clean.is_valid:
            print("       ! Warning: buffer(0) result is invalid. Applying aggressive clean...")
            try:
                shared_map_clean = shared_map_clean.buffer(TOLERANCE).buffer(-TOLERANCE)
            except Exception as e:
                print(f"       ! CRITICAL: Aggressive clean failed: {e}. Cannot build graph.")
                round_num += 1
                continue 
        if shared_map_clean.is_empty:
             print("       ! Warning: Shared map became empty after cleaning. Cannot build graph.")
             round_num += 1
             continue 
        
        d_safe, safe_graph, safe_centroids, centroid_tree, nav_poly = generate_a_star_graph_from_polygon(shared_map_clean) 
        
        if not safe_graph or not centroid_tree:
            print("       ! CRITICAL: Failed to build safe graph or tree. Ending simulation.")
            break
        
        # ==================================================================
        # --- 1b. Find Frontiers (NEW OC-SENSING LOGIC) ---
        # ==================================================================
        last_drone = stationary_drones[-1]
        print(f"       - Getting frontiers from last-placed drone: D{last_drone.id}")
        new_frontiers_raw = get_windows(polygon, last_drone.vis_poly) # Frontiers from last drone

        # Filter out frontiers already covered by *previous* drones
        frontiers_geom = new_frontiers_raw
        previous_drones = stationary_drones[:-1]
        
        if previous_drones:
            print(f"       - Filtering against map of {len(previous_drones)} previous drones...")
            previous_polys = [d.vis_poly for d in previous_drones if d.vis_poly and not d.vis_poly.is_empty]
            if previous_polys:
                previous_shared_map = unary_union(previous_polys).buffer(0)
                # Filter out frontiers already covered by *previous* drones
                frontiers_geom = new_frontiers_raw.difference(previous_shared_map.buffer(TOLERANCE * 2))
        
        if frontiers_geom.is_empty:
            print("--- 1. SENSE: No new, useful frontiers found by last drone. Mission complete.")
            break
            
        tasks_geom = list(frontiers_geom.geoms) if frontiers_geom.geom_type == 'MultiLineString' else [frontiers_geom]
        task_list_raw = [f for f in tasks_geom if f.length > MIN_FRONTIER_LENGTH]
        
        if not task_list_raw:
            print("--- 1. SENSE: Frontiers found, but all are too small. Mission complete.")
            break

        # Assign Parent (Simple: it's always the last_drone)
        print(f"       - Assigning parent (D{last_drone.id}) to {len(task_list_raw)} new frontiers...")
        tasks_for_auction = [] # This is the list for the auction
        for frontier in task_list_raw:
            frontier_midpoint = frontier.interpolate(0.5, normalized=True).coords[0]
            tasks_for_auction.append({
                'frontier': frontier,
                'parent_drone': last_drone, # <-- The key change
                'target_pos': frontier_midpoint
            })
            
        if not tasks_for_auction:
            print("       - No frontiers could be linked (this shouldn't happen).")
            round_num += 1
            continue 
        # ==================================================================
        # --- (END OF NEW SENSE BLOCK) ---
        # ==================================================================

        print(f"--- 1. SENSE: Found {len(tasks_for_auction)} total frontiers.")
        update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Found {len(tasks_for_auction)} Frontiers", frontiers=frontiers_geom)
        time.sleep(0.1)

        # --- 2. EVALUATE (Simplified: No Caching, No Staleness) ---
        print(f"--- 2. AUCTION: Evaluating {len(tasks_for_auction)} new frontiers. ---")
        
        broadcast_channel = [] # Will hold all results
        all_drones_in_field_this_round = []
        
        batch_num = 1
        while tasks_for_auction: 
            mobile_drones = [d for d in drone_fleet if d.status == 'available']
            if not mobile_drones:
                print("       - No mobile drones available to evaluate.")
                break
            
            for d in mobile_drones:
                if d.id not in [f.id for f in all_drones_in_field_this_round]:
                    all_drones_in_field_this_round.append(d)
            
            num_drones = len(mobile_drones)
            num_tasks_in_batch = min(num_drones, len(tasks_for_auction))
            tasks_for_this_batch = tasks_for_auction[:num_tasks_in_batch]
            drones_for_this_batch = mobile_drones[:num_tasks_in_batch]
            
            print(f"       - Processing Auction Batch {batch_num}: {len(drones_for_this_batch)} drones vs {len(tasks_for_this_batch)} tasks.")

            cost_matrix = np.zeros((len(drones_for_this_batch), len(tasks_for_this_batch)))
            animation_paths = {} 
            
            for i in range(len(drones_for_this_batch)):
                drone = drones_for_this_batch[i]
                # --- HYBRID LOGIC: Path from *current position* ---
                start_pos = drone.pos 
                for j in range(len(tasks_for_this_batch)):
                    task = tasks_for_this_batch[j]
                    end_pos = task['target_pos']
                    waypoints, cost = get_a_star_waypoint_path(
                        start_pos, end_pos, d_safe, safe_graph, safe_centroids, 
                        nav_poly, centroid_tree
                    )
                    cost_matrix[i, j] = cost
                    animation_paths[(i, j)] = waypoints

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            animation_data_list_out = []
            assigned_task_indices_in_batch = set()

            for i in range(len(row_ind)):
                drone_index = row_ind[i]
                task_index = col_ind[i]
                drone = drones_for_this_batch[drone_index]
                task_data = tasks_for_this_batch[task_index]
                travel_cost = cost_matrix[drone_index, task_index]
                path_waypoints = animation_paths[(drone_index, task_index)]
                
                # Get global index to remove from master list
                global_task_index = -1
                for k, main_task in enumerate(tasks_for_auction):
                    if main_task['target_pos'] == task_data['target_pos']:
                         global_task_index = k
                         break
                if global_task_index != -1:
                    assigned_task_indices_in_batch.add(global_task_index)
                
                drone.total_distance_traveled += travel_cost
                
                animation_data_list_out.append({
                    'drone': drone,
                    'out_path_points': path_waypoints,
                    'task_data': task_data 
                })
                drone.status = 'exploring' 
                drone.current_task_target_pos = task_data['target_pos']
                print_cost = f"Cost: {travel_cost:.2f} units (A*)"
                print(f"                     - D{drone.id} deploying to {np.round(task_data['target_pos'],1)}. {print_cost}")
            
            if animation_data_list_out:
                print("       - Animating drone deployment...")
                animate_drones_parallel(
                    animation_data_list_out, f"Drones evaluating (Batch {batch_num})", 
                    path_key='out_path_points',
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                    shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                    evaluated_tasks=broadcast_channel, # Show previous results
                    winning_task=None 
                )
            
            print("       - Drones arrived. Calculating gains...")
            for data in animation_data_list_out:
                drone = data['drone']
                task_data = data['task_data']
                best_point = task_data['target_pos']
                view = get_visibility_from_triangulation(
                    best_point, delaunator_like_data, obstructs
                )
                try: view_clean = view.buffer(0)
                except Exception as e: view_clean = view 
                gain = view_clean.difference(shared_map).area # Use real map
                print(f"                     - D{drone.id} reports from {np.round(best_point,1)}. Gain: {gain:.2f}")

                task_result = {
                    'pos': best_point, 
                    'gain': gain, 
                    'parent_drone': task_data['parent_drone'],
                }
                broadcast_channel.append(task_result) 
                
                # --- (DELETED: Caching logic) ---
                
                # --- HYBRID LOGIC: Stay in field ---
                drone.pos = data['out_path_points'][-1] 
                drone.status = 'available' 
            
            # Remove assigned tasks from the main auction list
            for index in sorted(list(assigned_task_indices_in_batch), reverse=True):
                del tasks_for_auction[index]
            
            batch_num += 1
            current_best_task = None
            if broadcast_channel:
                current_best_task = max(broadcast_channel, key=lambda x: x['gain'])
            update_plot(ax, polygon, drone_fleet, shared_map, 
                        f"Round {round_num}: Batch {batch_num} complete, {len(tasks_for_auction)} tasks left", 
                        frontiers=frontiers_geom, 
                        evaluated_tasks=broadcast_channel, 
                        winning_task=current_best_task) 
            time.sleep(0.1) 
        # --- END OF AUCTION ---
        
        # --- (DELETED: Staleness Check and Sequential Re-evaluation) ---

        # --- RECALL ALL DRONES (This is correct) ---
        animation_data_list_in = []
        if all_drones_in_field_this_round:
            print(f"       - Recalling {len(all_drones_in_field_this_round)} drones to gathering spot (D{gathering_spot_node.id})...")
            
            current_best_task = None
            if broadcast_channel:
                current_best_task = max(broadcast_channel, key=lambda x: x.get('gain', -1.0))

            for drone in all_drones_in_field_this_round:
                start_pos = drone.pos
                end_pos = gathering_spot_node.pos
                waypoints, cost = get_a_star_waypoint_path(
                    start_pos, end_pos, d_safe, safe_graph, safe_centroids,
                    nav_poly, centroid_tree
                )
                animation_data_list_in.append({
                    'drone': drone, 'in_path_points': waypoints
                })
                print(f"                     - D{drone.id} returning. Cost: {cost:.2f} units (A*).")
                drone.total_distance_traveled += cost
            
            if animation_data_list_in:
                animate_drones_parallel(
                    animation_data_list_in, f"Drones returning to D{gathering_spot_node.id}", 
                    path_key='in_path_points',
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                    shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                    evaluated_tasks=broadcast_channel, 
                    winning_task=current_best_task
                )
            
            for drone in all_drones_in_field_this_round:
                drone.status = 'available'
                drone.pos = gathering_spot_node.pos
                drone.current_task_target_pos = None
        # --- *** END OF EVALUATE/RECALL SECTION *** ---

        # --- 3. CONSENSUS PHASE ---
        print("--- 3. CONSENSUS: All drones independently decide the winner.")
        
        if not broadcast_channel: 
            print("       - No valid tasks found this round. Ending.")
            break
        else:
            best_task = max(broadcast_channel, key=lambda x: x.get('gain', -1.0))
            print(f"       - Consensus reached: Best frontier is at {np.round(best_task['pos'], 2)} with a gain of {best_task['gain']:.2f}")

        # --- 4. UPDATE PHASE ---
        print("--- 4. UPDATE: Re-tasking drone from gathering spot.")
        
        mobile_drones = [d for d in drone_fleet if d.status == 'available']
        if not mobile_drones:
             print("       - No mobile drones left to place. Mission complete.")
             break

        winner_drone = mobile_drones[0]
        target_pos = best_task['pos']
        target_parent = best_task['parent_drone'] 
        
        start_pos = gathering_spot_node.pos
        
        waypoints, deployment_cost = get_a_star_waypoint_path(
            start_pos, target_pos,
            d_safe, safe_graph, safe_centroids,
            nav_poly, centroid_tree
        )
        
        print(f"       - D{winner_drone.id} deploying (Cost: {deployment_cost:.2f})...")

        animate_single_drone_path(
            winner_drone, waypoints, f"D{winner_drone.id} deploying to new site",
            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, 
            frontiers=frontiers_geom, round_num=round_num,
            evaluated_tasks=broadcast_channel, winning_task=best_task
        )
        
        winner_drone.pos = target_pos
        winner_drone.status = 'stationary'
        winner_drone.parent = target_parent
        winner_drone.current_task_target_pos = None
        stationary_drones.append(winner_drone)
        winner_drone.total_distance_traveled += deployment_cost
        
        # --- 4d. MIGRATE BASE ---
        old_gathering_spot_node = gathering_spot_node
        gathering_spot_node = winner_drone 
        
        print(f"       - Migrating base. All available drones moving to D{gathering_spot_node.id}.")
        
        animation_data_list_migrate = []
        migrating_drones = [d for d in mobile_drones if d.id != winner_drone.id]
        
        if migrating_drones:
            start_pos_migrate = old_gathering_spot_node.pos
            end_pos_migrate = gathering_spot_node.pos
            
            waypoints_migrate, migration_cost = get_a_star_waypoint_path(
                start_pos_migrate, end_pos_migrate,
                d_safe, safe_graph, safe_centroids,
                nav_poly, centroid_tree
            )

            for drone in migrating_drones:
                animation_data_list_migrate.append({
                    'drone': drone,
                    'migrate_path_points': waypoints_migrate # Use shared path
                })
                print(f"                     - D{drone.id} migrating to new base. Cost: {migration_cost:.2f} units (A*).")
                drone.total_distance_traveled += migration_cost

            if animation_data_list_migrate:
                animate_drones_parallel(
                    animation_data_list_migrate, f"Swarm migrating to new base (D{gathering_spot_node.id})", 
                    path_key='migrate_path_points',
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                    shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                    evaluated_tasks=broadcast_channel, # Show tasks during migration
                    winning_task=best_task
                )
            
            for drone in migrating_drones:
                drone.pos = gathering_spot_node.pos
                drone.status = 'available' 
        
        # --- 4e. DEFER SLOW UPDATE ---
        winner_view = get_visibility_from_triangulation(
            winner_drone.pos, delaunator_like_data, obstructs
        )
        
        winner_drone.vis_poly = winner_view # <-- This is the fix for the NoneType error
        pending_map_update_view = winner_view # Defer the union/invalidation
        
        update_plot(ax, polygon, drone_fleet, shared_map, 
                    f"Round {round_num}: Base Established. (Waiting for Round {round_num+1})", 
                    frontiers=frontiers_geom, 
                    evaluated_tasks=broadcast_channel, 
                    winning_task=best_task)
        
        round_num += 1
        time.sleep(0.5)
        
    # --- End of the main while loop ---
    simulation_end_time = time.perf_counter() 
    total_runtime = simulation_end_time - simulation_start_time 
        
    print(f"\n{'='*15} MISSION COMPLETE {'='*15}")
    
    # Final map area calculation
    visible_area = shared_map.area
    if pending_map_update_view is not None: # Process any final update
        visible_area = shared_map.union(pending_map_update_view).area
        
    coverage_percentage = 0.0
    if total_free_area > 0:
        coverage_percentage = (visible_area / total_free_area) * 100
    
    print(f"Algorithm:           OC-Sensing + A* Hybrid-Movement (No Cache)")
    print(f"Total Free Space:    {total_free_area:,.2f} sq. units")
    print(f"Total Visible Area:  {visible_area:,.2f} sq. units")
    print(f"Final Coverage:      {coverage_percentage:.2f}%")
    print(f"Total Runtime:       {total_runtime:.2f} seconds")

    print(f"\n{'='*15} FINAL TRAVEL LOG {'='*15}")
    total_fleet_travel = 0.0
    for drone in drone_fleet:
        print(f" - Drone {drone.id} (Status: {drone.status}) Traveled: {drone.total_distance_traveled:.2f} units")
        total_fleet_travel += drone.total_distance_traveled
    print(f"--- Total Fleet Travel: {total_fleet_travel:.2f} units ---")
    
    final_title = f"Coverage Complete! ({len(stationary_drones)} Placed) - {coverage_percentage:.2f}% Coverage"
    # Final plot with the fully updated map
    if pending_map_update_view is not None:
        shared_map = shared_map.union(pending_map_update_view).buffer(0)
    update_plot(ax, polygon, drone_fleet, shared_map, final_title)
    plt.ioff()
    plt.show()