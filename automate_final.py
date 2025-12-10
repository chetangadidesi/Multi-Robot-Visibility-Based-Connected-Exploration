import math
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union, nearest_points
import triangle as tr
from typing import List, Tuple, Callable, Dict, Optional
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree 
import heapq 
import csv 
import os 

from maps import MAP_CONFIGS



# ==============================================================================
# CONSTANTS
# ==============================================================================
TOLERANCE = 0.001
NAV_BUFFER = 0.1 
N_FRONTIER_SAMPLES = 0 
MIN_FRONTIER_LENGTH = 1.0 

DRONE_SPEED_UNITS_PER_SECOND = 3000 # Units per second (e.g., meters/sec)
ANIMATION_UPDATE_INTERVAL_SECONDS = 0.05 
MAX_SIM_ATTEMPTS = 5

# === GEOMETRY STABILITY CONSTANT ===
GEOM_SIMPLIFY_TOLERANCE = 1
# ===================================
NAV_MESH_SIMPLIFY_TOLERANCE = 1.0
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
# Type Aliases (unchanged)
# ==============================================================================
PointT = Tuple[float, float]
Segment = Tuple[float, float, float, float]
DelaunatorLike = Dict[str, np.ndarray]
AStarGraph = Dict[int, List[Tuple[int, float]]]

# ==============================================================================
# TRIANGULATION HELPERS (unchanged)
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
# TRIANGULAR EXPANSION (VISIBILITY) (unchanged)
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
# GEOMETRY/A* HELPERS (unchanged)
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
    # Frontier is the boundary of the visible area NOT touching the environment boundary
    window_segments = all_exterior_lines.difference(buffered_environment)
    return window_segments

class Drone:
    def __init__(self, drone_id, initial_pos):
        self.id = drone_id
        self.pos = initial_pos
        self.status = 'available'
        self.parent = None
        self.vis_poly = None
        self.total_distance_traveled = 0.0
        self.total_travel_time = 0.0
        self.current_task_target_pos = None 

    def __repr__(self):
        return f"Drone {self.id} @ ({self.pos[0]:.2f}, {self.pos[1]:.2f})"

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

def get_frontier_key(frontier_geom: LineString, precision: int = 3) -> Tuple:
    try:
        coords = list(frontier_geom.coords)
        rounded_coords = [(round(x, precision), round(y, precision)) for x, y in coords]
        return tuple(sorted(rounded_coords))
    except: return tuple()

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
    coords = d['coords'].reshape(-1, 2)
    q = (qx, qy)
    num_triangles = len(d['triangles']) // 3

    for t_idx in range(num_triangles):
        p_indices = points_of_tri(d, t_idx)
        p1, p2, p3 = coords[p_indices]
        if orient2d(p1, p2, q) >= 0 and orient2d(p2, p3, q) >= 0 and orient2d(p3, p1, q) >= 0:
            return t_idx
    
    if centroid_tree is not None:
        _, closest_t_idx = centroid_tree.query((qx, qy))
        return int(closest_t_idx)

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
    
    return ([], 0.0)

def get_a_star_waypoint_path(
    start_pos: PointT, 
    end_pos: PointT,
    d: DelaunatorLike,
    graph: AStarGraph,
    centroids: List[PointT],
    nav_poly: Polygon, 
    centroid_tree: cKDTree 
) -> Tuple[List[PointT], float]:
    
    start_pos_safe = snap_point_to_polygon(start_pos, nav_poly)
    end_pos_safe = snap_point_to_polygon(end_pos, nav_poly)

    start_t = find_triangle_for_point(d, start_pos_safe[0], start_pos_safe[1], centroid_tree)
    end_t = find_triangle_for_point(d, end_pos_safe[0], end_pos_safe[1], centroid_tree)
    
    if start_t == -1 or end_t == -1:
        return ([start_pos, end_pos], path_distance(start_pos, end_pos))

    triangle_path, _ = a_star_search(start_t, end_t, graph, centroids)
    
    if not triangle_path:
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
    
    # NOTE: Suppressing print messages here
    
    try:
        if not polygon_to_graph.is_valid:
            polygon_to_graph = polygon_to_graph.buffer(0)
        if polygon_to_graph.is_empty:
            return ({}, {}, [], None, Polygon())
    except Exception as e:
        return ({}, {}, [], None, Polygon())

    try:
        # Use the new, large tolerance (NAV_MESH_SIMPLIFY_TOLERANCE) instead of TOLERANCE (0.001)
        poly_for_nav = polygon_to_graph.buffer(NAV_BUFFER).simplify(NAV_MESH_SIMPLIFY_TOLERANCE, preserve_topology=True)
    except Exception as e:
        poly_for_nav = polygon_to_graph

    if poly_for_nav.is_empty:
        return ({}, {}, [], None, Polygon())

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
        return ({}, {}, [], None, poly_for_nav)

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
        final_vertices_list = [None] * len(master_vertices_map)
        for coord, index in master_vertices_map.items():
            final_vertices_list[index] = coord
        vertices_np = np.array(final_vertices_list)
        segments_np = np.array(list(master_segments_set))
        holes_np = np.array(master_holes_list) if master_holes_list else None
    except Exception as e:
        return ({}, {}, [], None, poly_for_nav)

    if len(final_vertices_list) < 3:
        return ({}, {}, [], None, poly_for_nav)

    # 3. Triangulate
    scene = { 'vertices': vertices_np, 'segments': segments_np }
    if holes_np is not None and holes_np.size > 0: scene['holes'] = holes_np
    
    try:
        B = tr.triangulate(scene, 'p')
    except Exception as e:
        return ({}, {}, [], None, poly_for_nav)
        
    if 'triangles' not in B or len(B['triangles']) == 0:
        return ({}, {}, [], None, poly_for_nav)
        
    halfedges = build_halfedges(B['triangles'])
    d_safe = {
        'coords': B['vertices'].flatten(),
        'triangles': B['triangles'].flatten(),
        'halfedges': halfedges,
    }
    
    obstructs_safe = lambda e: False
    safe_graph, safe_centroids = build_a_star_graph(d_safe, obstructs_safe)
    
    centroid_tree = cKDTree(safe_centroids) if safe_centroids else None
    
    return (d_safe, safe_graph, safe_centroids, centroid_tree, poly_for_nav)

# ==============================================================================
# PLOTTING AND ANIMATION FUNCTIONS 
# ==============================================================================
def update_plot(ax, polygon, drone_fleet, shared_map, title_text, frontiers=None, evaluated_tasks=None, winning_task=None, is_batch_mode=False):
    if is_batch_mode:
        return

    # --- 1. PERSISTENCE LOGIC ---
    # Initialize static memory if it doesn't exist
    if not hasattr(update_plot, "last_winning_task"):
        update_plot.last_winning_task = None

    # RESET memory if this is the start of a new map (Round 0)
    if "Round 0" in title_text:
        update_plot.last_winning_task = None

    # Update memory ONLY if a new valid winning task is provided
    if winning_task is not None and 'pos' in winning_task:
        update_plot.last_winning_task = winning_task

    # Use the persisted task for drawing
    task_to_draw = update_plot.last_winning_task
    # -----------------------------

    ax.clear()
    
    # --- DRAWING GEOMETRY ---
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
        ax.plot(hx, hy, 'k-', linewidth=1, zorder=4)
        
    ax.plot(ox, oy, 'k-', linewidth=1, zorder=4)

    # --- DRAWING NETWORK LINKS ---
    for i, drone in enumerate(drone_fleet):
        if drone.status == 'stationary' and drone.parent:
            p_pos, c_pos = drone.parent.pos, drone.pos
            ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], 'k-', linewidth=1.5, zorder=5, label='Network Link' if i == 1 else '_nolegend_')

    # --- DRAWING FRONTIERS ---
    if frontiers:
        if frontiers.geom_type == 'MultiLineString':
            for i, frontier in enumerate(frontiers.geoms):
                x, y = frontier.xy
                ax.plot(x, y, 'c--', linewidth=2, label='Frontiers' if i == 0 else '_nolegend_', zorder=6)
        elif frontiers.geom_type == 'LineString':
             x, y = frontiers.xy
             ax.plot(x, y, 'c--', linewidth=2, label='Frontiers', zorder=6)

    # --- DRAWING EVALUATED TASKS (TEXT) ---
    if evaluated_tasks:
        for task in evaluated_tasks:
            if 'pos' in task:
                ax.text(task['pos'][0], task['pos'][1] + 0.3, f"{task['gain']:.1f}", color='blue', fontsize=9, ha='center', fontweight='bold', zorder=7)

    # --- DRAWING WINNING TASK (PERSISTENT) ---
    if task_to_draw and 'pos' in task_to_draw:
       # I added 'markerfacecolor' with alpha to make it pop more without blocking the view
       ax.plot(task_to_draw['pos'][0], task_to_draw['pos'][1], 
               'go',                   # Green Circle
               markersize=10,          # Slightly larger
               markeredgewidth=2,      # Thick edge
               label='Winning Task', 
               zorder=8)

    # --- DRAWING DRONES ---
    for drone in drone_fleet:
        if drone.status == 'stationary': color, marker, size, label = 'red', 'o', 10, 'Stationary Robot'
        elif drone.status == 'exploring': color, marker, size, label = 'blue', 's', 8, 'Exploring Robot'
        else: color, marker, size, label = 'dimgray', 'x', 6, 'Available (Idle) Robot'
        
        drone_pos = drone.pos
        ax.plot(drone_pos[0], drone_pos[1], marker, color=color, markersize=size, label=label, zorder=9)
        ax.text(drone_pos[0] + 0.2, drone_pos[1] + 0.2, f'D{drone.id}', color=color, fontsize=10, fontweight='bold', zorder=10)

    # --- AXIS SETTINGS ---
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title_text, fontsize=20)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize = 10)
    
    plt.draw()
    plt.pause(0.01)

def animate_single_drone_path(
    drone: Drone,
    path_points: List[Tuple[float, float]],
    title_suffix: str,
    ax, polygon, drone_fleet, shared_map, frontiers, round_num,
    evaluated_tasks=None, winning_task=None, is_batch_mode=False
):
    if is_batch_mode:
        drone.pos = path_points[-1] if path_points else drone.pos
        return
        
    drone.status = 'exploring'
    total_path_len = get_path_total_length(path_points)
    if total_path_len == 0: return

    total_time = total_path_len / DRONE_SPEED_UNITS_PER_SECOND
    drone.total_travel_time += total_time
    time_step = ANIMATION_UPDATE_INTERVAL_SECONDS
    
    current_time = 0.0
    while current_time < total_time + TOLERANCE:
        distance_traveled = current_time * DRONE_SPEED_UNITS_PER_SECOND
        distance_traveled = min(distance_traveled, total_path_len)
        drone.pos = get_point_at_distance(path_points, distance_traveled)
        
        update_plot(ax, polygon, drone_fleet, shared_map, 
                    f"Round {round_num}: {title_suffix}",
                    frontiers=frontiers,
                    evaluated_tasks=evaluated_tasks,
                    winning_task=winning_task, is_batch_mode=is_batch_mode) 

        current_time += time_step
    
def animate_drones_parallel(
    animation_data_list: List[Dict],
    title_suffix: str,
    path_key: str, 
    ax, polygon, drone_fleet, shared_map, frontiers, round_num,
    evaluated_tasks: Optional[List[Dict]] = None,
    winning_task: Optional[Dict] = None,
    is_batch_mode=False
):
    
    if is_batch_mode:
        for data in animation_data_list:
            if data.get(path_key): data['drone'].pos = data[path_key][-1]
        return
        
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
    
    if max_path_length <= 0: return

    total_time = max_path_length / DRONE_SPEED_UNITS_PER_SECOND
    for data in animation_data_list:
        drone_path_len = data['total_path_len']
        if drone_path_len > 0:
            drone_move_time = drone_path_len / DRONE_SPEED_UNITS_PER_SECOND
            data['drone'].total_travel_time += drone_move_time
    time_step = ANIMATION_UPDATE_INTERVAL_SECONDS
    
    current_time = 0.0
    while current_time < total_time + TOLERANCE:
        distance_traveled_global = current_time * DRONE_SPEED_UNITS_PER_SECOND
        
        for data in animation_data_list:
            if not data['path_points']: continue
            drone = data['drone']
            path_len = data['total_path_len']
            target_dist = min(distance_traveled_global, path_len)
            drone.pos = get_point_at_distance(data['path_points'], target_dist)
            
        title = f"Round {round_num}: {title_suffix}"
        update_plot(ax, polygon, drone_fleet, shared_map, 
                    title, frontiers=frontiers,
                    evaluated_tasks=evaluated_tasks,
                    winning_task=winning_task, is_batch_mode=is_batch_mode) 

        current_time += time_step
        
    for data in animation_data_list:
        if data['path_points']:
            data['drone'].pos = data['path_points'][-1]


# ==============================================================================
# --- SUMMARY PRINT HELPER ---
# ==============================================================================
def print_summary_tables(map_name: str, round_summary_stats: List[Dict], total_free_area: float, visible_area: float, total_runtime: float, drone_fleet: List, algorithm_name: str):
    """Prints the final two required tables to the console, including algorithm name."""
    
    coverage_percentage = (visible_area / total_free_area) * 100 if total_free_area > 0 else 0.0
    
    print(f"\n\n--- MAP FINISHED: {map_name} ---")
    
    # --- ROUND SUMMARY STATS ---
    print(f"\n{'='*15} ROUND SUMMARY STATS {'='*15}")
    
    # Safely determine column widths
    max_round = max(stats['Round'] for stats in round_summary_stats) if round_summary_stats else 0
    max_drones = max(stats['Drones_Placed'] for stats in round_summary_stats) if round_summary_stats else 0
    round_width = max(len(str(max_round)), 3)
    drone_width = max(len(str(max_drones)), 6)
    
    # MODIFIED: Added Travel Time (s) column
    header_format = f"{{:<{round_width}}} | {{:<{drone_width}}} | {{:>10}} | {{:>10}} | {{:>15}} | {{:>10}}"
    row_format = f"{{:<{round_width}}} | {{:<{drone_width}}} | {{:>10.2f}} | {{:>10.2f}} | {{:>15.2f}} | {{:>10.2f}}"
    separator = "-" * (round_width + drone_width + 50)

    print(header_format.format("Rnd", "Drones", "Wall Time (s)", "Travel Time (s)", "Dist. Travelled", "Coverage (%)"))
    print(separator)
    for stats in round_summary_stats:
        print(row_format.format(
            stats['Round'],
            stats['Drones_Placed'],
            stats['Wall_Clock_Time'],
            stats.get('Total_Travel_Time', 0.0), # Added
            stats['Total_Distance_Traveled'],
            stats['Coverage_Percentage']
        ))
    print(separator)
    
    # --- OVERALL ALGORITHM METRICS ---
    total_fleet_travel = sum(d.total_distance_traveled for d in drone_fleet)
    total_fleet_travel_time = sum(d.total_travel_time for d in drone_fleet) # Added
    
    print(f"\n{'='*15} OVERALL ALGORITHM METRICS {'='*15}")
    print(f"Algorithm: 	{algorithm_name}")
    print(f"Total Free Space: {total_free_area:,.2f} sq. units")
    print(f"Total Visible Area: {visible_area:,.2f} sq. units")
    print(f"Final Coverage: {coverage_percentage:.2f}%")
    print(f"Total Fleet Travel: {total_fleet_travel:,.2f} units")
    print(f"Total Travel Time: {total_fleet_travel_time:.2f} seconds") # Added
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print("=" * 49)

# ==============================================================================
# --- CORE ALGORITHM 1: GREEDY AUCTION + A* PATHS (Refactored Original) ---
# ==============================================================================
def run_greedy_auction_a_star(map_config: Dict, is_batch_mode: bool) -> List[Dict]:
    """
    Runs the multi-drone Greedy Auction Algorithm with A* path planning,
    geometry robustness, and frontier caching/re-evaluation.
    """
    ALGORITHM_NAME = "Chained-Auction + A* Paths (Robust Nav + Caching)"
    map_name = map_config.get('name', 'Unknown Map')
    
    # --- RETRY LOOP START ---
    attempt = 0
    while attempt < MAX_SIM_ATTEMPTS:
        attempt += 1
        print(f"\n--- ATTEMPT {attempt}/{MAX_SIM_ATTEMPTS} FOR MAP: {map_name} (Algorithm: {ALGORITHM_NAME}) ---")
        
        fig, ax = None, None 

        try:
            # --- GET MAP PARAMETERS ---
            outer_boundary = map_config.get('outer_boundary', [])
            holes = map_config.get('holes', [])
            start_pos = map_config.get('start_pos', (125, 500)) 
            fleet_size = map_config.get('fleet_size', 3) 
            num_stationary_goal = map_config.get('stationary_goal', 3) 
            
            # --- INITIAL SETUP ---
            simulation_start_time = time.perf_counter()
            round_summary_stats = []
            frontier_cache = {} 
            

            try:
                polygon = Polygon(outer_boundary, holes) 
            except Exception as e:
                print(f"!!! CRITICAL ERROR: Initial Polygon failed: {e.__class__.__name__}. Trying simple box.")
                polygon = Polygon([(0, 0), (0, 0), (0, 20), (20, 20), (20, 0), (0, 0)]) 
                
            polygon = polygon.buffer(0)
            if polygon.is_empty: raise ValueError("Polygon is empty after initial cleaning.")
            
            polygon = polygon.simplify(GEOM_SIMPLIFY_TOLERANCE).buffer(0) 
            
            if polygon.is_empty or polygon.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError("Map became invalid or empty after aggressive simplification.")

            total_free_area = polygon.area
            
            # --- PRE-COMPUTATION: Triangulation (Visibility) ---
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
            
            # 6. Initialize Drone Fleet
            drone_fleet = [Drone(i, start_pos) for i in range(fleet_size)]
            first_drone = drone_fleet[0]
            first_drone.status = 'stationary'
            first_drone.vis_poly = get_visibility_from_triangulation(
                first_drone.pos, delaunator_like_data, obstructs
            )
            
            stationary_drones = [first_drone]
            shared_map = first_drone.vis_poly
            pending_map_update_view = None 
            
            root_node = first_drone
            gathering_spot_node = root_node 
            
            for drone in drone_fleet:
                if drone.id != root_node.id:
                    drone.parent = root_node
                    
            fig, ax = plt.subplots(figsize=(12, 12)) 
            plt.ion()
            update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round 0: Initial State", is_batch_mode=is_batch_mode) 
            plt.pause(5)
            # Record initial state (Round 0)
            round_summary_stats.append({
                'Round': 0,
                'Drones_Placed': 1,
                'Wall_Clock_Time': time.perf_counter() - simulation_start_time,
                'Total_Distance_Traveled': 0.0, 
                'Visible_Area': shared_map.area,
                'Coverage_Percentage': (shared_map.area / total_free_area) * 100
            })

            # 7. Run Main Simulation Loop 
            round_num = 1
            while len(stationary_drones) < num_stationary_goal:
                
                # --- PROCESS DEFERRED UPDATES ---
                if pending_map_update_view is not None:
                    shared_map = shared_map.union(pending_map_update_view).buffer(0).simplify(TOLERANCE, preserve_topology=True)
                    pending_map_update_view = None 
                    update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round {round_num}: Map Updated, Calculating...", frontiers=None, is_batch_mode=is_batch_mode) 
                
                # --- 1. SENSE & Nav Graph Build ---
                try:
                    shared_map_simple = shared_map.simplify(TOLERANCE, preserve_topology=True)
                    shared_map_clean = shared_map_simple.buffer(0)
                except Exception as e:
                    shared_map_clean = shared_map
                if not shared_map_clean.is_valid:
                    try:
                        shared_map_clean = shared_map_clean.buffer(TOLERANCE).buffer(-TOLERANCE)
                    except Exception as e:
                        round_num += 1
                        continue 
                if shared_map_clean.is_empty: break
                
                d_safe, safe_graph, safe_centroids, centroid_tree, nav_poly = generate_a_star_graph_from_polygon(shared_map_clean) 
                
                if not safe_graph or not centroid_tree:
                    raise RuntimeError("Graph generation failure.") 
                
                # --- 1b. Find Frontiers ---
                frontiers_geom = get_windows(polygon, shared_map)
                if frontiers_geom.is_empty: break
                    
                tasks_geom = list(frontiers_geom.geoms) if frontiers_geom.geom_type == 'MultiLineString' else [frontiers_geom]
                task_list_raw = [f for f in tasks_geom if f.length > MIN_FRONTIER_LENGTH]
                
                if not task_list_raw: break

                # --- 1c. Assign Parents to Frontiers ---
                all_current_frontiers = []
                for frontier in task_list_raw:
                    frontier_midpoint_obj = frontier.interpolate(0.5, normalized=True)
                    frontier_midpoint = frontier_midpoint_obj.coords[0]
                    best_parent = None
                    min_parent_dist = float('inf')
                    for s_drone in stationary_drones:
                        if s_drone.vis_poly is None: continue
                        if s_drone.vis_poly.buffer(TOLERANCE).contains(frontier_midpoint_obj):
                            dist = Point(s_drone.pos).distance(frontier_midpoint_obj)
                            if dist < min_parent_dist:
                                min_parent_dist = dist
                                best_parent = s_drone
                    if best_parent:
                        all_current_frontiers.append({
                            'frontier': frontier,
                            'parent_drone': best_parent,
                            'target_pos': frontier_midpoint
                        })
                
                if not all_current_frontiers:
                    round_num += 1
                    continue 

                update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round {round_num}: Found {len(all_current_frontiers)} Frontiers", frontiers=frontiers_geom, is_batch_mode=is_batch_mode) 
                time.sleep(0.1 if not is_batch_mode else 0)

                # --- 2. EVALUATE (Auction/Cache Logic) ---
                tasks_for_auction_new = []
                tasks_for_auction_stale = []
                cached_tasks_fresh = []
                cached_tasks_stale_data = []
                current_map_area = shared_map.area
                
                for task in all_current_frontiers:
                    key = get_frontier_key(task['frontier'])
                    if not key: continue 
                    
                    if key in frontier_cache:
                        cached_data = frontier_cache[key]
                        cached_data['parent_drone'] = task['parent_drone'] 
                        
                        if abs(cached_data.get('map_area_when_cached', 0) - current_map_area) < 1.0:
                            cached_tasks_fresh.append(cached_data)
                        else:
                            tasks_for_auction_stale.append(task)
                            cached_tasks_stale_data.append(cached_data)
                    else:
                        tasks_for_auction_new.append(task)
                
                newly_evaluated_tasks = []
                re_evaluated_tasks = []
                all_drones_in_field_this_round = []
                
                tasks_for_auction = tasks_for_auction_new
                
                batch_num = 1
                while tasks_for_auction: 
                    mobile_drones = [d for d in drone_fleet if d.status == 'available']
                    if not mobile_drones: break
                    
                    for d in mobile_drones:
                        if d.id not in [f.id for f in all_drones_in_field_this_round]:
                            all_drones_in_field_this_round.append(d)
                    
                    num_drones = len(mobile_drones)
                    num_tasks_in_batch = min(num_drones, len(tasks_for_auction))
                    tasks_for_this_batch = tasks_for_auction[:num_tasks_in_batch]
                    drones_for_this_batch = mobile_drones[:num_tasks_in_batch]
                    
                    cost_matrix = np.zeros((len(drones_for_this_batch), len(tasks_for_this_batch)))
                    animation_paths = {} 
                    
                    for i in range(len(drones_for_this_batch)):
                        drone = drones_for_this_batch[i]
                        start_pos_drone = drone.pos 
                        for j in range(len(tasks_for_this_batch)):
                            task = tasks_for_this_batch[j]
                            end_pos = task['target_pos']
                            waypoints, cost = get_a_star_waypoint_path(
                                start_pos_drone, end_pos, d_safe, safe_graph, safe_centroids, 
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
                        global_task_index = tasks_for_auction.index(task_data)
                        assigned_task_indices_in_batch.add(global_task_index)
                        drone.total_distance_traveled += travel_cost
                        
                        animation_data_list_out.append({
                            'drone': drone,
                            'out_path_points': path_waypoints,
                            'task_data': task_data 
                        })
                        drone.status = 'exploring' 
                        drone.current_task_target_pos = task_data['target_pos']
                    
                    if animation_data_list_out:
                        animate_drones_parallel(
                            animation_data_list_out, f"Drones evaluating", 
                            path_key='out_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                            shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                            evaluated_tasks=newly_evaluated_tasks, 
                            winning_task=None, is_batch_mode=is_batch_mode
                        )
                    
                    for data in animation_data_list_out:
                        drone = data['drone']
                        task_data = data['task_data']
                        best_point = task_data['target_pos']
                        view = get_visibility_from_triangulation(
                            best_point, delaunator_like_data, obstructs
                        )
                        try: view_clean = view.buffer(0)
                        except Exception as e: view_clean = view 
                        gain = view_clean.difference(shared_map).area 

                        task_result = {
                            'pos': best_point, 
                            'gain': gain, 
                            'parent_drone': task_data['parent_drone'],
                            'map_area_when_cached': current_map_area 
                        }
                        newly_evaluated_tasks.append(task_result) 
                        
                        key = get_frontier_key(task_data['frontier'])
                        if key:
                            if gain > 0 or N_FRONTIER_SAMPLES == 0:
                                frontier_cache[key] = task_result
                        
                        drone.pos = data['out_path_points'][-1] 
                        drone.status = 'available' 
                    
                    for index in sorted(list(assigned_task_indices_in_batch), reverse=True):
                        del tasks_for_auction[index]
                    
                    batch_num += 1
                    current_best_task = None
                    all_tasks_so_far = newly_evaluated_tasks
                    if all_tasks_so_far:
                        current_best_task = max(all_tasks_so_far, key=lambda x: x['gain'])
                    update_plot(ax, polygon, drone_fleet, shared_map, 
                                f"{map_name} - Round {round_num}, {len(tasks_for_auction)} tasks left", 
                                frontiers=frontiers_geom, 
                                evaluated_tasks=all_tasks_so_far, 
                                winning_task=current_best_task, is_batch_mode=is_batch_mode) 
                    time.sleep(0.1 if not is_batch_mode else 0)

                # --- 2b/2c. STALENESS CHECK & SEQUENTIAL RE-EVALUATION ---
                max_new_gain = max(task['gain'] for task in newly_evaluated_tasks) if newly_evaluated_tasks else -1.0
                max_cached_gain = max(task.get('gain', -1.0) for task in (cached_tasks_fresh + cached_tasks_stale_data)) if (cached_tasks_fresh + cached_tasks_stale_data) else -1.0
                
                broadcast_channel = []
                
                if max_cached_gain > max_new_gain:
                    stale_candidates_for_re_eval = []
                    trusted_stale_data = [] 
                    for task, data in zip(tasks_for_auction_stale, cached_tasks_stale_data):
                        if data.get('gain', -1.0) > max_new_gain:
                            stale_candidates_for_re_eval.append((task, data))
                        else:
                            trusted_stale_data.append(data)
                    stale_candidates_for_re_eval.sort(key=lambda x: x[1].get('gain', -1.0), reverse=True)
                    
                    best_task_so_far = max(newly_evaluated_tasks, key=lambda x: x.get('gain', -1.0)) if newly_evaluated_tasks else None
                    current_champion_gain = max_new_gain
                    
                    for i, (task_to_check, stale_data) in enumerate(stale_candidates_for_re_eval):
                        cached_gain = stale_data.get('gain', -1.0)
                        if cached_gain <= current_champion_gain:
                            break 
                            
                        mobile_drones = [d for d in drone_fleet if d.status == 'available']
                        if not mobile_drones: break
                            
                        drone = mobile_drones[0]
                        if drone.id not in [f.id for f in all_drones_in_field_this_round]:
                            all_drones_in_field_this_round.append(drone)

                        start_pos_drone = drone.pos
                        end_pos = task_to_check['target_pos']
                        waypoints, cost = get_a_star_waypoint_path(start_pos_drone, end_pos, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)
                        drone.total_distance_traveled += cost
                        
                        animation_data_list_out = [{'drone': drone, 'out_path_points': waypoints, 'task_data': task_to_check}]
                        animate_drones_parallel(animation_data_list_out, f"D{drone.id} re-evaluating cached frontiers", path_key='out_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                            evaluated_tasks=(newly_evaluated_tasks + re_evaluated_tasks), winning_task=best_task_so_far, is_batch_mode=is_batch_mode) 

                        best_point = task_to_check['target_pos']
                        view = get_visibility_from_triangulation(best_point, delaunator_like_data, obstructs)
                        try: view_clean = view.buffer(0)
                        except Exception as e: view_clean = view
                        new_updated_gain = view_clean.difference(shared_map).area
                        
                        task_result = {'pos': best_point, 'gain': new_updated_gain, 'parent_drone': task_to_check['parent_drone'], 'map_area_when_cached': current_map_area }
                        re_evaluated_tasks.append(task_result)
                        
                        key = get_frontier_key(task_to_check['frontier']) 
                        if key: frontier_cache[key] = task_result 

                        if new_updated_gain > current_champion_gain:
                            current_champion_gain = new_updated_gain
                            best_task_so_far = task_result
                        
                        if (i + 1) < len(stale_candidates_for_re_eval):
                            next_task_cached_gain = stale_candidates_for_re_eval[i+1][1].get('gain', -1.0)
                            if current_champion_gain >= next_task_cached_gain:
                                drone.pos = waypoints[-1]; drone.status = 'available'; drone.current_task_target_pos = task_to_check['target_pos']
                                break 
                            
                        drone.pos = waypoints[-1]; drone.status = 'available'; drone.current_task_target_pos = task_to_check['target_pos']
                    
                    num_checked = len(re_evaluated_tasks)
                    untested_stale_data = [data for task, data in stale_candidates_for_re_eval[num_checked:]]
                    
                    broadcast_channel = (newly_evaluated_tasks + re_evaluated_tasks + cached_tasks_fresh + trusted_stale_data + untested_stale_data)
                
                else:
                    broadcast_channel = newly_evaluated_tasks + cached_tasks_fresh + cached_tasks_stale_data
                
                # --- RECALL ALL DRONES ---
                animation_data_list_in = []
                if all_drones_in_field_this_round:
                    
                    current_best_task = max(broadcast_channel, key=lambda x: x.get('gain', -1.0)) if broadcast_channel else None

                    for drone in all_drones_in_field_this_round:
                        start_pos_drone = drone.pos
                        end_pos = gathering_spot_node.pos
                        waypoints, cost = get_a_star_waypoint_path(start_pos_drone, end_pos, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)
                        animation_data_list_in.append({'drone': drone, 'in_path_points': waypoints})
                        drone.total_distance_traveled += cost
                    
                    if animation_data_list_in:
                        animate_drones_parallel(animation_data_list_in, f"Drones returning to D{gathering_spot_node.id}", path_key='in_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                            evaluated_tasks=broadcast_channel, winning_task=current_best_task, is_batch_mode=is_batch_mode) 
                    
                    for drone in all_drones_in_field_this_round:
                        drone.status = 'available'
                        drone.pos = gathering_spot_node.pos
                        drone.current_task_target_pos = None

                # --- 3. CONSENSUS PHASE ---
                if not broadcast_channel: break
                else:
                    best_task = max(broadcast_channel, key=lambda x: x.get('gain', -1.0))

                # --- 4. UPDATE PHASE (Deployment & Migration) ---
                mobile_drones = [d for d in drone_fleet if d.status == 'available']
                if not mobile_drones: break

                winner_drone = mobile_drones[0]
                target_pos = best_task['pos']
                target_parent = best_task['parent_drone'] 
                
                start_pos_deploy = gathering_spot_node.pos
                
                waypoints, deployment_cost = get_a_star_waypoint_path(start_pos_deploy, target_pos, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)
                
                animate_single_drone_path(winner_drone, waypoints, f"D{winner_drone.id} deploying to new site",
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, 
                    frontiers=frontiers_geom, round_num=round_num,
                    evaluated_tasks=broadcast_channel, winning_task=best_task, is_batch_mode=is_batch_mode) 
                
                winner_drone.pos = target_pos
                winner_drone.status = 'stationary'
                winner_drone.parent = target_parent
                winner_drone.current_task_target_pos = None
                stationary_drones.append(winner_drone)
                winner_drone.total_distance_traveled += deployment_cost
                
                old_gathering_spot_node = gathering_spot_node
                gathering_spot_node = winner_drone 
                
                animation_data_list_migrate = []
                migrating_drones = [d for d in mobile_drones if d.id != winner_drone.id]
                
                if migrating_drones:
                    start_pos_migrate = old_gathering_spot_node.pos
                    end_pos_migrate = gathering_spot_node.pos
                    
                    waypoints_migrate, migration_cost = get_a_star_waypoint_path(start_pos_migrate, end_pos_migrate, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)

                    for drone in migrating_drones:
                        animation_data_list_migrate.append({'drone': drone, 'migrate_path_points': waypoints_migrate })
                        drone.total_distance_traveled += migration_cost

                    if animation_data_list_migrate:
                        animate_drones_parallel(animation_data_list_migrate, f"Swarm migrating to new base (D{gathering_spot_node.id})", path_key='migrate_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                            evaluated_tasks=broadcast_channel, winning_task=best_task, is_batch_mode=is_batch_mode) 
                    
                    for drone in migrating_drones:
                        drone.pos = gathering_spot_node.pos
                        drone.status = 'available' 
                
                # --- 4e. DEFER SLOW UPDATE ---
                winner_view = get_visibility_from_triangulation(winner_drone.pos, delaunator_like_data, obstructs)
                winner_drone.vis_poly = winner_view 
                pending_map_update_view = winner_view 
                
                update_plot(ax, polygon, drone_fleet, shared_map, 
                            f"{map_name} - Round {round_num}: Base Established. (Waiting for Round {round_num+1})", 
                            frontiers=frontiers_geom, 
                            evaluated_tasks=broadcast_channel, 
                            winning_task=best_task, is_batch_mode=is_batch_mode) 
                
                # VITAL: Record end-of-round statistics
                current_total_distance = sum(d.total_distance_traveled for d in drone_fleet)
                current_total_travel_time = sum(d.total_travel_time for d in drone_fleet) # Added
                current_visible_area = shared_map.union(pending_map_update_view).area 
    
                round_summary_stats.append({
                    'Round': round_num,
                    'Drones_Placed': len(stationary_drones),
                    'Wall_Clock_Time': time.perf_counter() - simulation_start_time,
                    'Total_Distance_Traveled': current_total_distance,
                    'Total_Travel_Time': current_total_travel_time, # Added
                    'Visible_Area': current_visible_area,
                    'Coverage_Percentage': (current_visible_area / total_free_area) * 100
                })
                
                round_num += 1
                time.sleep(0.5 if not is_batch_mode else 0)
                
            # --- MISSION COMPLETE ---
            simulation_end_time = time.perf_counter() 
            total_runtime = simulation_end_time - simulation_start_time 
                
            print(f"\n{'='*15} MISSION COMPLETE {'='*15}")
            
            # Final calculation 
            visible_area = shared_map.area
            if pending_map_update_view is not None: 
                visible_area = shared_map.union(pending_map_update_view).area
                
            print_summary_tables(map_name, round_summary_stats, total_free_area, visible_area, total_runtime, drone_fleet, ALGORITHM_NAME)
            
            # Final plot update
            if pending_map_update_view is not None:
                shared_map = shared_map.union(pending_map_update_view).buffer(0)
            coverage_percentage = (visible_area / total_free_area) * 100 if total_free_area > 0 else 0.0
            final_title = f"{len(stationary_drones)} Robots Placed - {coverage_percentage:.2f}% Coverage"
            update_plot(ax, polygon, drone_fleet, shared_map, final_title, is_batch_mode=is_batch_mode) 
            
            plt.ioff()

            print(f"--- ATTEMPT {attempt} SUCCEEDED. ---")
            return round_summary_stats

        except Exception as e:
            # Catch any unexpected crash during the run
            print(f"!!! CRITICAL SIMULATION CRASH IN ATTEMPT {attempt} (Error Type: {e.__class__.__name__})")
            print(f"    Error Details: {e}")
            
            if fig is not None:
                plt.close(fig) # Ensure the figure is closed
            
            if attempt >= MAX_SIM_ATTEMPTS:
                print(f"!!! MAX ATTEMPTS REACHED for {map_name}. ABORTING MAP.")
                return [] 
            
            time.sleep(1) 
    
    return [] 

# ==============================================================================
# --- CORE ALGORITHM 2: ONLY CHILD SENSING ---
# ==============================================================================
def run_only_child_sensing(map_config: Dict, is_batch_mode: bool) -> List[Dict]:
    """
    Runs the 'Only Child Sensing' algorithm implemented within the Greedy Auction framework.
    Frontiers are only generated from the latest stationary drone's field of view, 
    but all available drones bid on these frontiers concurrently.
    - No caching or staleness checks are performed.
    """
    ALGORITHM_NAME = "Only Child Sensing (Auction + A*)"
    map_name = map_config.get('name', 'Unknown Map')
    
    # --- RETRY LOOP START ---
    attempt = 0
    while attempt < MAX_SIM_ATTEMPTS:
        attempt += 1
        print(f"\n--- ATTEMPT {attempt}/{MAX_SIM_ATTEMPTS} FOR MAP: {map_name} (Algorithm: {ALGORITHM_NAME}) ---")

        fig, ax = None, None
        
        try:
            # --- GET MAP PARAMETERS & INITIAL SETUP ---
            outer_boundary = map_config.get('outer_boundary', [])
            holes = map_config.get('holes', [])
            start_pos = map_config.get('start_pos', (125, 500)) 
            fleet_size = map_config.get('fleet_size', 3) 
            num_stationary_goal = map_config.get('stationary_goal', 3) 
            
            simulation_start_time = time.perf_counter()
            round_summary_stats = []
            
            try: polygon = Polygon(outer_boundary, holes) 
            except: polygon = Polygon([(0, 0), (0, 0), (0, 20), (20, 20), (20, 0), (0, 0)]) 
                
            polygon = polygon.buffer(0)
            if polygon.is_empty: raise ValueError("Polygon is empty.")
            polygon = polygon.simplify(GEOM_SIMPLIFY_TOLERANCE).buffer(0)
            if polygon.is_empty or polygon.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError("Map became invalid or empty after simplification.")

            total_free_area = polygon.area
            
            # --- PRE-COMPUTATION: Triangulation (Visibility & Nav) ---
            vertices = []
            segments = []
            holes_list = [] 
            if polygon.geom_type == 'Polygon': polygons_to_process = [polygon]
            elif polygon.geom_type == 'MultiPolygon': polygons_to_process = list(polygon.geoms)
            else: polygons_to_process = []
            current_vertex_index = 0
            for poly in polygons_to_process:
                if poly.is_empty: continue
                exterior_coords = list(poly.exterior.coords[:-1]); vertices.extend(exterior_coords)
                ext_len = len(exterior_coords); segments.extend([[current_vertex_index + i, current_vertex_index + (i + 1) % ext_len] for i in range(ext_len)])
                current_vertex_index += ext_len
                for interior in poly.interiors:
                    hole_coords = list(interior.coords[:-1]); vertices.extend(hole_coords)
                    hole_len = len(hole_coords); segments.extend([[current_vertex_index + i, current_vertex_index + (i + 1) % hole_len] for i in range(hole_len)])
                    hole_poly = Polygon(interior).buffer(0);
                    if not hole_poly.is_empty: holes_list.append(list(hole_poly.representative_point().coords)[0])
                    current_vertex_index += hole_len

            scene = { 'vertices': np.array(vertices), 'segments': np.array(segments) }
            if holes_list: scene['holes'] = np.array(holes_list)
            B = tr.triangulate(scene, 'pA') 
            halfedges = build_halfedges(B['triangles'])
            delaunator_like_data = { 'coords': B['vertices'].flatten(), 'triangles': B['triangles'].flatten(), 'halfedges': halfedges }
            obstructing_segments = { tuple(sorted(B['segments'][i])) for i, m in enumerate(B['segment_markers']) if m == 1 }
            def obstructs(edge_idx: int) -> bool:
                p1 = delaunator_like_data['triangles'][edge_idx]
                p2 = delaunator_like_data['triangles'][next_edge(edge_idx)]
                return tuple(sorted((p1, p2))) in obstructing_segments

            # 6. Initialize Drone Fleet
            drone_fleet = [Drone(i, start_pos) for i in range(fleet_size)]
            first_drone = drone_fleet[0]; first_drone.status = 'stationary'
            first_drone.vis_poly = get_visibility_from_triangulation(first_drone.pos, delaunator_like_data, obstructs)
            
            stationary_drones = [first_drone]
            shared_map = first_drone.vis_poly
            pending_map_update_view = None 
            gathering_spot_node = first_drone 
            for drone in drone_fleet:
                if drone.id != gathering_spot_node.id: drone.parent = gathering_spot_node
                    
            fig, ax = plt.subplots(figsize=(12, 12)) 
            plt.ion()
            update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round 0: Initial State", is_batch_mode=is_batch_mode) 

            # Record initial state (Round 0)
            round_summary_stats.append({
                'Round': 0, 'Drones_Placed': 1, 'Wall_Clock_Time': time.perf_counter() - simulation_start_time,
                'Total_Distance_Traveled': 0.0, 'Visible_Area': shared_map.area,
                'Coverage_Percentage': (shared_map.area / total_free_area) * 100
            })
            
            # 7. Run Main Simulation Loop 
            round_num = 1
            while len(stationary_drones) < num_stationary_goal:
                
                # --- PROCESS DEFERRED UPDATES ---
                if pending_map_update_view is not None:
                    shared_map = shared_map.union(pending_map_update_view).buffer(0).simplify(TOLERANCE, preserve_topology=True)
                    pending_map_update_view = None 
                    update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round {round_num}: Map Updated, Calculating...", frontiers=None, is_batch_mode=is_batch_mode) 
                
                # --- 1. SENSE & Nav Graph Build ---
                try: shared_map_clean = shared_map.simplify(TOLERANCE, preserve_topology=True).buffer(0)
                except: shared_map_clean = shared_map.buffer(0)
                if shared_map_clean.is_empty: break
                
                d_safe, safe_graph, safe_centroids, centroid_tree, nav_poly = generate_a_star_graph_from_polygon(shared_map_clean) 
                if not safe_graph or not centroid_tree: break
                
                # ==================================================================
                # --- 1b. Find Frontiers (OCS Logic: Only from Last Drone) ---
                # ==================================================================
                last_drone = stationary_drones[-1]
                
                # 1. Get frontiers based ONLY on the last drone's view
                new_frontiers_raw = get_windows(polygon, last_drone.vis_poly)
                
                # 2. Filter out frontiers already covered by *all* stationary drones
                frontiers_geom = new_frontiers_raw
                if len(stationary_drones) > 1: # Only filter if there are older drones
                    previous_polys = [d.vis_poly for d in stationary_drones[:-1] if d.vis_poly and not d.vis_poly.is_empty]
                    if previous_polys:
                        previous_shared_map = unary_union(previous_polys).buffer(0)
                        frontiers_geom = new_frontiers_raw.difference(previous_shared_map.buffer(TOLERANCE * 2))
                
                if frontiers_geom.is_empty: break
                    
                tasks_geom = list(frontiers_geom.geoms) if frontiers_geom.geom_type == 'MultiLineString' else [frontiers_geom]
                task_list_raw = [f for f in tasks_geom if f.length > MIN_FRONTIER_LENGTH]
                
                if not task_list_raw: break

                # 3. Assign Parent (Simple: it's always the last_drone in OCS)
                all_current_frontiers = []
                for frontier in task_list_raw:
                    frontier_midpoint = frontier.interpolate(0.5, normalized=True).coords[0]
                    all_current_frontiers.append({
                        'frontier': frontier,
                        'parent_drone': last_drone, 
                        'target_pos': frontier_midpoint
                    })
                
                if not all_current_frontiers:
                    round_num += 1
                    continue 
                
                frontiers_plot = frontiers_geom # Keep the filtered geometry for plotting
                # ==================================================================
                # --- (END OF OCS SENSE BLOCK) ---
                # ==================================================================

                update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round {round_num}: Found {len(all_current_frontiers)} OCS Frontiers", frontiers=frontiers_plot, is_batch_mode=is_batch_mode) 
                time.sleep(0.1 if not is_batch_mode else 0)

                # --- 2. EVALUATE (Full Concurrent Greedy Auction Logic) ---
                tasks_for_auction = all_current_frontiers # OCS uses all available OCS frontiers
                newly_evaluated_tasks = []
                all_drones_in_field_this_round = []
                
                batch_num = 1
                while tasks_for_auction: 
                    mobile_drones = [d for d in drone_fleet if d.status == 'available']
                    if not mobile_drones: break
                    
                    for d in mobile_drones:
                        if d.id not in [f.id for f in all_drones_in_field_this_round]:
                            all_drones_in_field_this_round.append(d)
                    
                    num_drones = len(mobile_drones)
                    num_tasks_in_batch = min(num_drones, len(tasks_for_auction))
                    tasks_for_this_batch = tasks_for_auction[:num_tasks_in_batch]
                    drones_for_this_batch = mobile_drones[:num_tasks_in_batch]
                    
                    cost_matrix = np.zeros((len(drones_for_this_batch), len(tasks_for_this_batch)))
                    animation_paths = {} 
                    
                    for i in range(len(drones_for_this_batch)):
                        drone = drones_for_this_batch[i]
                        start_pos_drone = drone.pos 
                        for j in range(len(tasks_for_this_batch)):
                            task = tasks_for_this_batch[j]
                            end_pos = task['target_pos']
                            waypoints, cost = get_a_star_waypoint_path(
                                start_pos_drone, end_pos, d_safe, safe_graph, safe_centroids, 
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
                        global_task_index = tasks_for_auction.index(task_data)
                        assigned_task_indices_in_batch.add(global_task_index)
                        drone.total_distance_traveled += travel_cost
                        
                        animation_data_list_out.append({
                            'drone': drone,
                            'out_path_points': path_waypoints,
                            'task_data': task_data 
                        })
                        drone.status = 'exploring' 
                        drone.current_task_target_pos = task_data['target_pos']
                    
                    if animation_data_list_out:
                        animate_drones_parallel(
                            animation_data_list_out, f"Drones evaluating", 
                            path_key='out_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, 
                            shared_map=shared_map, frontiers=frontiers_plot, round_num=round_num,
                            evaluated_tasks=newly_evaluated_tasks, 
                            winning_task=None, is_batch_mode=is_batch_mode
                        )
                    
                    for data in animation_data_list_out:
                        drone = data['drone']
                        task_data = data['task_data']
                        best_point = task_data['target_pos']
                        view = get_visibility_from_triangulation(best_point, delaunator_like_data, obstructs)
                        try: view_clean = view.buffer(0)
                        except Exception as e: view_clean = view 
                        gain = view_clean.difference(shared_map).area 

                        task_result = {
                            'pos': best_point, 
                            'gain': gain, 
                            'parent_drone': task_data['parent_drone']
                        }
                        newly_evaluated_tasks.append(task_result) 
                        drone.pos = data['out_path_points'][-1] 
                        drone.status = 'available' 
                    
                    for index in sorted(list(assigned_task_indices_in_batch), reverse=True):
                        del tasks_for_auction[index]
                    
                    batch_num += 1
                    current_best_task = None
                    if newly_evaluated_tasks: current_best_task = max(newly_evaluated_tasks, key=lambda x: x['gain'])
                    
                    update_plot(ax, polygon, drone_fleet, shared_map, 
                                f"{map_name} - Round {round_num} complete", 
                                frontiers=frontiers_plot, 
                                evaluated_tasks=newly_evaluated_tasks, 
                                winning_task=current_best_task, is_batch_mode=is_batch_mode) 
                    time.sleep(0.1 if not is_batch_mode else 0)

                broadcast_channel = newly_evaluated_tasks
                
                # --- RECALL ALL DRONES ---
                animation_data_list_in = []
                if all_drones_in_field_this_round:
                    current_best_task = max(broadcast_channel, key=lambda x: x.get('gain', -1.0)) if broadcast_channel else None

                    for drone in all_drones_in_field_this_round:
                        start_pos_drone = drone.pos
                        end_pos = gathering_spot_node.pos
                        waypoints, cost = get_a_star_waypoint_path(start_pos_drone, end_pos, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)
                        animation_data_list_in.append({'drone': drone, 'in_path_points': waypoints})
                        drone.total_distance_traveled += cost
                    
                    if animation_data_list_in:
                        animate_drones_parallel(animation_data_list_in, f"Drones returning to D{gathering_spot_node.id}", path_key='in_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, frontiers=frontiers_plot, round_num=round_num,
                            evaluated_tasks=broadcast_channel, winning_task=current_best_task, is_batch_mode=is_batch_mode) 
                    
                    for drone in all_drones_in_field_this_round:
                        drone.status = 'available'
                        drone.pos = gathering_spot_node.pos
                        drone.current_task_target_pos = None

                # --- 3. CONSENSUS PHASE ---
                if not broadcast_channel: break
                else:
                    best_task = max(broadcast_channel, key=lambda x: x.get('gain', -1.0))

                # --- 4. UPDATE PHASE (Deployment & Migration) ---
                mobile_drones = [d for d in drone_fleet if d.status == 'available']
                if not mobile_drones: break

                winner_drone = mobile_drones[0]
                target_pos = best_task['pos']
                target_parent = best_task['parent_drone'] 
                start_pos_deploy = gathering_spot_node.pos
                
                waypoints, deployment_cost = get_a_star_waypoint_path(start_pos_deploy, target_pos, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)
                
                animate_single_drone_path(winner_drone, waypoints, f"D{winner_drone.id} deploying to new site",
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, 
                    frontiers=frontiers_plot, round_num=round_num,
                    evaluated_tasks=broadcast_channel, winning_task=best_task, is_batch_mode=is_batch_mode) 
                
                winner_drone.pos = target_pos
                winner_drone.status = 'stationary'
                winner_drone.parent = target_parent
                winner_drone.current_task_target_pos = None
                stationary_drones.append(winner_drone)
                winner_drone.total_distance_traveled += deployment_cost
                
                old_gathering_spot_node = gathering_spot_node
                gathering_spot_node = winner_drone 
                
                animation_data_list_migrate = []
                migrating_drones = [d for d in mobile_drones if d.id != winner_drone.id]
                
                if migrating_drones:
                    start_pos_migrate = old_gathering_spot_node.pos
                    end_pos_migrate = gathering_spot_node.pos
                    
                    waypoints_migrate, migration_cost = get_a_star_waypoint_path(start_pos_migrate, end_pos_migrate, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)

                    for drone in migrating_drones:
                        animation_data_list_migrate.append({'drone': drone, 'migrate_path_points': waypoints_migrate })
                        drone.total_distance_traveled += migration_cost

                    if animation_data_list_migrate:
                        animate_drones_parallel(animation_data_list_migrate, f"Swarm migrating to new base (D{gathering_spot_node.id})", path_key='migrate_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, frontiers=frontiers_plot, round_num=round_num,
                            evaluated_tasks=broadcast_channel, winning_task=best_task, is_batch_mode=is_batch_mode) 
                    
                    for drone in migrating_drones:
                        drone.pos = gathering_spot_node.pos
                        drone.status = 'available' 
                
                # --- 4e. DEFER SLOW UPDATE ---
                winner_view = get_visibility_from_triangulation(winner_drone.pos, delaunator_like_data, obstructs)
                winner_drone.vis_poly = winner_view 
                pending_map_update_view = winner_view 
                
                update_plot(ax, polygon, drone_fleet, shared_map, 
                            f"{map_name} - Round {round_num}: Base Established. (Waiting for Round {round_num+1})", 
                            frontiers=frontiers_plot, 
                            evaluated_tasks=broadcast_channel, 
                            winning_task=best_task, is_batch_mode=is_batch_mode) 
                
                # VITAL: Record end-of-round statistics
                current_total_distance = sum(d.total_distance_traveled for d in drone_fleet)
                current_total_travel_time = sum(d.total_travel_time for d in drone_fleet) # Added
                current_visible_area = shared_map.union(pending_map_update_view).area 
    
                round_summary_stats.append({
                    'Round': round_num,
                    'Drones_Placed': len(stationary_drones),
                    'Wall_Clock_Time': time.perf_counter() - simulation_start_time,
                    'Total_Distance_Traveled': current_total_distance,
                    'Total_Travel_Time': current_total_travel_time, # Added
                    'Visible_Area': current_visible_area,
                    'Coverage_Percentage': (current_visible_area / total_free_area) * 100
                })
                
                round_num += 1
                time.sleep(0.5 if not is_batch_mode else 0)
                
            # --- MISSION COMPLETE ---
            simulation_end_time = time.perf_counter() 
            total_runtime = simulation_end_time - simulation_start_time 
                
            print(f"\n{'='*15} MISSION COMPLETE {'='*15}")
            
            # Final calculation 
            visible_area = shared_map.area
            if pending_map_update_view is not None: 
                visible_area = shared_map.union(pending_map_update_view).area
                
            print_summary_tables(map_name, round_summary_stats, total_free_area, visible_area, total_runtime, drone_fleet, ALGORITHM_NAME)
            
            # Final plot update
            if pending_map_update_view is not None:
                shared_map = shared_map.union(pending_map_update_view).buffer(0)
            coverage_percentage = (visible_area / total_free_area) * 100 if total_free_area > 0 else 0.0
            final_title = f"{len(stationary_drones)} Robots Placed - {coverage_percentage:.2f}% Coverage"
            update_plot(ax, polygon, drone_fleet, shared_map, final_title, is_batch_mode=is_batch_mode) 
            
            plt.ioff()

            print(f"--- ATTEMPT {attempt} SUCCEEDED. ---")
            return round_summary_stats

        except Exception as e:
            print(f"!!! CRASH IN ATTEMPT {attempt} (Error Type: {e.__class__.__name__})")
            print(f"    Error Details: {e}")
            
            if fig is not None:
                plt.close(fig) 
            
            if attempt >= MAX_SIM_ATTEMPTS:
                print(f"!!! MAX ATTEMPTS REACHED for {map_name}. ABORTING MAP.")
                return [] 
            
            time.sleep(1) 
    
    return []

# ==============================================================================
# --- CORE ALGORITHM 3: WIDEST WINDOW GREEDY ---
# ==============================================================================
def run_widest_window_greedy(map_config: Dict, is_batch_mode: bool) -> List[Dict]:
    """
    Runs the 'Widest Window Greedy' algorithm: Selects the single longest 
    frontier and assigns the closest available drone for deployment using A* path planning.
    """
    ALGORITHM_NAME = "Widest Window Greedy + A* Paths"
    map_name = map_config.get('name', 'Unknown Map')
    
    # --- RETRY LOOP START ---
    attempt = 0
    while attempt < MAX_SIM_ATTEMPTS:
        attempt += 1
        print(f"\n--- ATTEMPT {attempt}/{MAX_SIM_ATTEMPTS} FOR MAP: {map_name} (Algorithm: {ALGORITHM_NAME}) ---")

        fig, ax = None, None
        
        try:
            # --- INITIAL SETUP ---
            outer_boundary = map_config.get('outer_boundary', [])
            holes = map_config.get('holes', [])
            start_pos = map_config.get('start_pos', (125, 500)) 
            fleet_size = map_config.get('fleet_size', 3) 
            num_stationary_goal = map_config.get('stationary_goal', 3) 
            
            simulation_start_time = time.perf_counter()
            round_summary_stats = []
            
            polygon = Polygon(outer_boundary, holes)
            polygon = polygon.buffer(0)
            if polygon.is_empty: raise ValueError("Polygon is empty.")
            polygon = polygon.simplify(GEOM_SIMPLIFY_TOLERANCE).buffer(0)
            if polygon.is_empty or polygon.geom_type not in ('Polygon', 'MultiPolygon'):
                raise ValueError("Map became invalid or empty after simplification.")

            total_free_area = polygon.area
            
            # --- PRE-COMPUTATION: Triangulation (Visibility & Nav) ---
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

            # 6. Initialize Drone Fleet
            drone_fleet = [Drone(i, start_pos) for i in range(fleet_size)]
            first_drone = drone_fleet[0]
            first_drone.status = 'stationary'
            first_drone.vis_poly = get_visibility_from_triangulation(
                first_drone.pos, delaunator_like_data, obstructs
            )
            
            stationary_drones = [first_drone]
            shared_map = first_drone.vis_poly
            pending_map_update_view = None 
            gathering_spot_node = first_drone 
            
            fig, ax = plt.subplots(figsize=(12, 12)) 
            plt.ion()
            update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round 0: Initial State", is_batch_mode=is_batch_mode) 

            # Record initial state (Round 0)
            round_summary_stats.append({
                'Round': 0,
                'Drones_Placed': 1,
                'Wall_Clock_Time': time.perf_counter() - simulation_start_time,
                'Total_Distance_Traveled': 0.0, 
                'Visible_Area': shared_map.area,
                'Coverage_Percentage': (shared_map.area / total_free_area) * 100
            })
            
            # 7. Run Main Simulation Loop 
            round_num = 1
            while len(stationary_drones) < num_stationary_goal:
                
                # --- PROCESS DEFERRED UPDATES ---
                if pending_map_update_view is not None:
                    shared_map = shared_map.union(pending_map_update_view).buffer(0).simplify(TOLERANCE, preserve_topology=True)
                    pending_map_update_view = None 
                    update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round {round_num}: Map Updated, Calculating...", frontiers=None, is_batch_mode=is_batch_mode) 
                
                # --- 1. SENSE & Nav Graph Build ---
                try:
                    shared_map_simple = shared_map.simplify(TOLERANCE, preserve_topology=True)
                    shared_map_clean = shared_map_simple.buffer(0)
                except Exception as e:
                    shared_map_clean = shared_map
                if not shared_map_clean.is_valid:
                    try:
                        shared_map_clean = shared_map_clean.buffer(TOLERANCE).buffer(-TOLERANCE)
                    except Exception as e:
                        round_num += 1
                        continue 
                if shared_map_clean.is_empty: break
                
                # Build A* Graph
                d_safe, safe_graph, safe_centroids, centroid_tree, nav_poly = generate_a_star_graph_from_polygon(shared_map_clean) 
                if not safe_graph or not centroid_tree: raise RuntimeError("Graph generation failure.") 
                
                # --- 1b. Find Frontiers ---
                frontiers_geom = get_windows(polygon, shared_map)
                if frontiers_geom.is_empty: break
                    
                tasks_geom = list(frontiers_geom.geoms) if frontiers_geom.geom_type == 'MultiLineString' else [frontiers_geom]
                task_list_raw = [f for f in tasks_geom if f.length > MIN_FRONTIER_LENGTH]
                
                if not task_list_raw: break

                update_plot(ax, polygon, drone_fleet, shared_map, f"{map_name} - Round {round_num}: Found {len(task_list_raw)} Frontiers", frontiers=frontiers_geom, is_batch_mode=is_batch_mode) 
                time.sleep(0.1 if not is_batch_mode else 0)
                
                # --- 2. EVALUATE (Widest Window) ---
                best_frontier = max(task_list_raw, key=lambda f: f.length)
                target_pos = best_frontier.interpolate(0.5, normalized=True).coords[0]
                
                # The "Parent" is the stationary drone closest to the target.
                target_pos_point = Point(target_pos)
                min_dist_to_parent = float('inf')
                best_parent = None
                for s_drone in stationary_drones:
                    if s_drone.vis_poly is None: continue
                    if s_drone.vis_poly.buffer(TOLERANCE).contains(target_pos_point):
                        dist = Point(s_drone.pos).distance(target_pos_point)
                        if dist < min_dist_to_parent:
                            min_dist_to_parent = dist
                            best_parent = s_drone
                            
                # Get the deployment drone (closest available drone)
                mobile_drones = [d for d in drone_fleet if d.status == 'available']
                if not mobile_drones: break

                min_travel_cost = float('inf')
                winner_drone = None
                best_waypoints = []
                
                for drone in mobile_drones:
                    waypoints, cost = get_a_star_waypoint_path(drone.pos, target_pos, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)
                    if cost < min_travel_cost:
                        min_travel_cost = cost
                        winner_drone = drone
                        best_waypoints = waypoints
                
                if not winner_drone: break 
                
                # Estimate gain (only for comparison/plotting)
                view = get_visibility_from_triangulation(target_pos, delaunator_like_data, obstructs)
                try: view_clean = view.buffer(0)
                except Exception as e: view_clean = view
                gain = view_clean.difference(shared_map).area

                best_task = {'pos': target_pos, 'gain': gain, 'parent_drone': best_parent}
                
                # --- 3. DEPLOYMENT (A*) ---
                winner_drone.total_distance_traveled += min_travel_cost

                animate_single_drone_path(winner_drone, best_waypoints, f"D{winner_drone.id} deploying to widest window",
                    ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, 
                    frontiers=frontiers_geom, round_num=round_num,
                    evaluated_tasks=None, winning_task=best_task, is_batch_mode=is_batch_mode)
                
                # Update drone state
                winner_drone.pos = target_pos
                winner_drone.status = 'stationary'
                winner_drone.parent = best_parent if best_parent else gathering_spot_node
                winner_drone.current_task_target_pos = None
                stationary_drones.append(winner_drone)
                
                old_gathering_spot_node = gathering_spot_node
                gathering_spot_node = winner_drone 
                
                # --- Migration (Closest available drones migrate) ---
                animation_data_list_migrate = []
                migrating_drones = [d for d in mobile_drones if d.id != winner_drone.id]
                
                if migrating_drones:
                    start_pos_migrate = old_gathering_spot_node.pos
                    end_pos_migrate = gathering_spot_node.pos
                    
                    waypoints_migrate, migration_cost = get_a_star_waypoint_path(start_pos_migrate, end_pos_migrate, d_safe, safe_graph, safe_centroids, nav_poly, centroid_tree)

                    for drone in migrating_drones:
                        animation_data_list_migrate.append({'drone': drone, 'migrate_path_points': waypoints_migrate })
                        drone.total_distance_traveled += migration_cost

                    if animation_data_list_migrate:
                        animate_drones_parallel(animation_data_list_migrate, f"Swarm migrating to new base (D{gathering_spot_node.id})", path_key='migrate_path_points',
                            ax=ax, polygon=polygon, drone_fleet=drone_fleet, shared_map=shared_map, frontiers=frontiers_geom, round_num=round_num,
                            evaluated_tasks=None, winning_task=best_task, is_batch_mode=is_batch_mode) 
                    
                    for drone in migrating_drones:
                        drone.pos = gathering_spot_node.pos
                        drone.status = 'available' 

                # --- 4. DEFER SLOW UPDATE & RECORD STATS ---
                winner_view = get_visibility_from_triangulation(winner_drone.pos, delaunator_like_data, obstructs)
                winner_drone.vis_poly = winner_view 
                pending_map_update_view = winner_view 
                
                update_plot(ax, polygon, drone_fleet, shared_map, 
                            f"{map_name} - Round {round_num}: Base Established. (Waiting for Round {round_num+1})", 
                            frontiers=frontiers_geom, 
                            evaluated_tasks=None, 
                            winning_task=best_task, is_batch_mode=is_batch_mode) 
                
                # Record end-of-round statistics
                current_total_distance = sum(d.total_distance_traveled for d in drone_fleet)
                current_total_travel_time = sum(d.total_travel_time for d in drone_fleet) # Added
                current_visible_area = shared_map.union(pending_map_update_view).area 
    
                round_summary_stats.append({
                    'Round': round_num,
                    'Drones_Placed': len(stationary_drones),
                    'Wall_Clock_Time': time.perf_counter() - simulation_start_time,
                    'Total_Distance_Traveled': current_total_distance,
                    'Total_Travel_Time': current_total_travel_time, # Added
                    'Visible_Area': current_visible_area,
                    'Coverage_Percentage': (current_visible_area / total_free_area) * 100
                })
                
                round_num += 1
                time.sleep(0.5 if not is_batch_mode else 0)
                
            # --- MISSION COMPLETE ---
            simulation_end_time = time.perf_counter() 
            total_runtime = simulation_end_time - simulation_start_time 
            
            visible_area = shared_map.area
            if pending_map_update_view is not None: 
                visible_area = shared_map.union(pending_map_update_view).area
                
            print_summary_tables(map_name, round_summary_stats, total_free_area, visible_area, total_runtime, drone_fleet, ALGORITHM_NAME)
            
            # Final plot update
            if pending_map_update_view is not None:
                shared_map = shared_map.union(pending_map_update_view).buffer(0)
            coverage_percentage = (visible_area / total_free_area) * 100 if total_free_area > 0 else 0.0
            final_title = f"{len(stationary_drones)} Robots Placed - {coverage_percentage:.2f}% Coverage"
            update_plot(ax, polygon, drone_fleet, shared_map, final_title, is_batch_mode=is_batch_mode) 
            
            plt.ioff()
            print(f"--- ATTEMPT {attempt} SUCCEEDED. ---")
            return round_summary_stats

        except Exception as e:
            print(f"!!! CRITICAL SIMULATION CRASH IN ATTEMPT {attempt} (Error Type: {e.__class__.__name__})")
            print(f"    Error Details: {e}")
            if fig is not None: plt.close(fig)
            if attempt >= MAX_SIM_ATTEMPTS: return []
            time.sleep(1) 
    return []

# ==============================================================================
# --- MAIN EXECUTION BLOCK (Refactored to select Algorithm) ---
# ==============================================================================
# ==============================================================================
# --- MAIN EXECUTION BLOCK (Updated with Range/Batch Support) ---
# ==============================================================================
def main_refactored():
    map_options = {i + 1: config for i, config in enumerate(MAP_CONFIGS)}
    
    algorithms = {
        1: {'name': 'Greedy Auction + A*', 'func': run_greedy_auction_a_star, 'abbr': 'GREEDY_AUCTION'},
        2: {'name': 'Only Child Sensing', 'func': run_only_child_sensing, 'abbr': 'CHILD_SENSE'},
        3: {'name': 'Widest Window Greedy + A*', 'func': run_widest_window_greedy, 'abbr': 'WIDEST_WINDOW'}
    }

    print("\n--- Multi-Algorithm Exploration Simulator ---")
    
    print("\nAvailable Algorithms:")
    for index, algo in algorithms.items():
        print(f"  [{index}]: {algo['name']}")
    
    while True:
        try:
            algo_input = input("\nEnter the algorithm number to run: ").strip()
            algo_index = int(algo_input)
            
            if algo_index not in algorithms:
                print("Invalid algorithm selection.")
                continue
            
            selected_algo = algorithms[algo_index]
            print(f"\n--- Selected Algorithm: {selected_algo['name']} ---")
            
            print("\nAvailable Maps:")
            for index, config in map_options.items():
                print(f"  [{index}]: {config['name']}")
            print(f"  [ALL]: Run ALL {len(MAP_CONFIGS)} maps (Batch Mode - No Plotting)")
            print(f"  [X-Y]: Run range (e.g. '1-10') (Interactive Mode - With Plotting)")
            print(f"  [0]: Exit")

            map_input = input("\nEnter selection (Number, Range, or ALL): ").strip().upper()
            
            if map_input == '0':
                print("Exiting simulator.")
                break
            
            # ==========================================
            # 1. SEQUENCE EXECUTION (ALL or RANGE)
            # ==========================================
            if map_input == 'ALL' or '-' in map_input:
                maps_to_process = []
                batch_label = ""
                
                # --- DETERMINE MODE ---
                # ALL = Batch Mode (No Plotting, Fast)
                # Range = Interactive Mode (With Plotting, Sequential)
                is_batch = (map_input == 'ALL')

                # Case A: Run ALL
                if map_input == 'ALL':
                    maps_to_process = MAP_CONFIGS
                    batch_label = "ALL_MAPS"
                
                # Case B: Run RANGE (e.g., "1-10")
                else:
                    try:
                        parts = map_input.split('-')
                        if len(parts) != 2: raise ValueError
                        start_idx = int(parts[0].strip())
                        end_idx = int(parts[1].strip())
                        
                        if start_idx < 1 or end_idx > len(MAP_CONFIGS):
                            print(f"Error: Range must be between 1 and {len(MAP_CONFIGS)}.")
                            continue
                        if start_idx > end_idx:
                            print("Error: Start number cannot be higher than end number.")
                            continue
                            
                        maps_to_process = MAP_CONFIGS[start_idx-1 : end_idx]
                        batch_label = f"MAPS_{start_idx}-{end_idx}"
                        
                    except ValueError:
                        print("Invalid format. Please use 'Start-End' (e.g., 1-10).")
                        continue

                # --- Execute Sequence ---
                if is_batch:
                    print(f"\n--- Initiating Batch Run: {batch_label} ({len(maps_to_process)} maps) ---")
                    print("⚡ Plotting is DISABLED for performance.")
                else:
                    print(f"\n--- Initiating Sequential Run: {batch_label} ({len(maps_to_process)} maps) ---")
                    print("📈 Plotting is ENABLED. Maps will run sequentially.")
                
                all_results_to_export = []
                fieldnames = ['Map_Name', 'Algorithm', 'Round', 'Drones_Placed', 'Wall_Clock_Time', 
                              'Total_Travel_Time', 'Total_Distance_Traveled', 'Visible_Area', 'Coverage_Percentage']
                
                for i, config in enumerate(maps_to_process):
                    if not is_batch:
                        print(f"\n>> Running Map {i+1}/{len(maps_to_process)}: {config.get('name', 'Unknown')}...")

                    # Run algorithm (Plotting enabled if is_batch is False)
                    map_stats = selected_algo['func'](config, is_batch_mode=is_batch)
                    
                    # Close the plot window after the map finishes so the next one can start cleanly
                    plt.close('all') 

                    if not map_stats:
                        print(f"!!! Map {config.get('name', 'Unknown')} failed/skipped. !!!")
                        continue 

                    map_name_str = config.get('name', 'Unknown Map')
                    for row in map_stats:
                        row['Map_Name'] = map_name_str
                        row['Algorithm'] = selected_algo['abbr']
                        all_results_to_export.append(row)

                # Export CSV
                output_filename = f"{selected_algo['abbr'].lower()}_sequence_{batch_label.lower()}.csv"
                with open(output_filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results_to_export)
                
                print(f"\n✅ Sequence Complete. Statistics saved to: **{output_filename}**")
                break 

            # ==========================================
            # 2. SINGLE MAP MODE (Integer input)
            # ==========================================
            else:
                map_index = int(map_input)
                if 1 <= map_index <= len(MAP_CONFIGS):
                    selected_config = map_options[map_index]
                    # Run single map WITH plotting (is_batch_mode=False)
                    map_stats = selected_algo['func'](selected_config, is_batch_mode=False) 
                    
                    if map_stats:
                        plt.show() 
                    else:
                        print("\nSimulation failed to complete successfully after all retries.")
                    break
                else:
                    print(f"Invalid map selection. Please enter a number between 1 and {len(MAP_CONFIGS)}.")
        
        except ValueError:
            print("Invalid input. Please enter a valid number or range (e.g., 1-10).")
        except IndexError:
            print("Invalid map number selection.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

# ==============================================================================
# --- MAIN EXECUTION BLOCK (New Entry Point) ---
# ==============================================================================
if __name__ == "__main__":
    main_refactored()