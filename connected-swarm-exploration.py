import math
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, MultiLineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

# A small tolerance value used for geometric buffering operations
TOLERANCE = 1e-3

def cast_rays_with_holes(polygon, viewpoint):
    """Calculates the visibility polygon from a viewpoint."""

    ray_points = []
    angles = []
    all_vertices = list(polygon.exterior.coords)
    edges = list(zip(all_vertices[:-1], all_vertices[1:]))
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        all_vertices.extend(interior_coords)
        edges.extend(zip(interior_coords[:-1], interior_coords[1:]))
    for vx, vy in all_vertices:
        angle = math.atan2(vy - viewpoint[1], vx - viewpoint[0])
        angles.extend([angle - 0.00001, angle, angle + 0.00001])
    angles.sort()
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        near_point = (viewpoint[0] + dx * 0.01, viewpoint[1] + dy * 0.01)
        if not polygon.contains(Point(near_point)):
            ray_points.append(viewpoint)
            continue
        far_point = (viewpoint[0] + dx * 1000, viewpoint[1] + dy * 1000)
        ray = LineString([near_point, far_point])
        min_dist = float('inf')
        closest_pt = None
        for seg_start, seg_end in edges:
            edge = LineString([seg_start, seg_end])
            if ray.intersects(edge):
                intersection = ray.intersection(edge)
                if isinstance(intersection, Point):
                    dist = Point(viewpoint).distance(intersection)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pt = intersection
        if closest_pt:
            ray_points.append((round(closest_pt.x, 3), round(closest_pt.y, 3)))
    return ray_points


def get_windows(polygon, vis_polygon):
    """
    Finds visibility windows (frontiers) using a robust geometric difference.
    This works correctly for both Polygon and MultiPolygon inputs.
    """
    if vis_polygon is None or vis_polygon.is_empty:
        return LineString()
    vis_boundary = vis_polygon.boundary
    env_boundary = polygon.boundary
    buffered_env = env_boundary.buffer(TOLERANCE * 2)
    windows = vis_boundary.difference(buffered_env)
    return windows

# --- HELPER FUNCTIONS FOR FRONTIER ANALYSIS ---
def generate_points_on_lines(lines, distance_step, margin=TOLERANCE * 10): # Added margin parameter
    """
    Generates evenly spaced points along a LineString or MultiLineString,
    avoiding the endpoints by a small margin.
    """
    points = []
    if lines is None or lines.is_empty: return points
    lines_list = list(lines.geoms) if isinstance(lines, MultiLineString) else [lines]

    for line in lines_list:
        if line.length <= margin * 2:
            if line.length > 0:
                 points.append(line.centroid)
            continue

        start_dist = margin
        end_dist = line.length - margin

        # Generate points within the adjusted range
        for dist in np.arange(start_dist, end_dist, distance_step):
             points.append(line.interpolate(dist))

        last_dist = np.arange(start_dist, end_dist, distance_step)[-1]
        if end_dist - last_dist > distance_step / 2: # Add if there's a significant gap
             points.append(line.interpolate(end_dist))


    unique_points = []
    seen_coords = set()
    for p in points:
        # Check if point object is valid before accessing coordinates
        if p is None: continue
        coords = (round(p.x, 5), round(p.y, 5))
        if coords not in seen_coords:
            unique_points.append(p)
            seen_coords.add(coords)

    # Handle the case where margins removed all points from a valid line
    # If unique_points is empty but the line wasn't initially skipped, add centroid
    if not unique_points and any(l.length > margin * 2 for l in lines_list):
         # Find centroid of the original potentially multi-part line
         if lines and not lines.is_empty:
             points.append(lines.centroid)
             # Manually add to unique points if needed, check for None first
             if points[-1]:
                 coords = (round(points[-1].x, 5), round(points[-1].y, 5))
                 if coords not in seen_coords:
                     unique_points.append(points[-1])
                     seen_coords.add(coords)


    return unique_points

def find_best_point_on_frontier(frontier_line, environment_poly, current_shared_map, distance_step=1):
    """
    Analyzes points along a frontier line and returns the point with the
    maximum ADDED visibility gain and the gain itself.
    """
    best_point_coords = None
    max_added_gain = -1.0
    sample_points_geom = generate_points_on_lines(frontier_line, distance_step)
    internal_point = None
    if current_shared_map and current_shared_map.is_valid and not current_shared_map.is_empty:
        internal_point = current_shared_map.representative_point()
    if internal_point is None:
        print("Warning: Could not determine internal point for frontier analysis.")
        internal_point = Point(0, 0)
    if not sample_points_geom:
        candidate_pos = frontier_line.centroid.coords[0]
        try:
            view = Polygon(cast_rays_with_holes(environment_poly, candidate_pos)).simplify(TOLERANCE)
            if view.is_valid:
                 gain = view.difference(current_shared_map).area
                 return candidate_pos, gain if gain > 0 else 0.0
        except Exception: pass
        return candidate_pos, 0.0
    for point_geom in sample_points_geom:
        direction_inward = LineString([point_geom, internal_point])
        if direction_inward.length < TOLERANCE:
             safe_point_geom = point_geom
        else:
             safe_point_geom = direction_inward.interpolate(TOLERANCE * 2)
        safe_point = (safe_point_geom.x, safe_point_geom.y)
        vp = cast_rays_with_holes(environment_poly, safe_point)
        if not vp or len(vp) < 3: continue
        try:
            poly = Polygon(vp).simplify(TOLERANCE)
            if not poly.is_valid or poly.is_empty: continue
            added_region = poly.difference(current_shared_map)
            added_gain = added_region.area
        except Exception: continue
        if added_gain > max_added_gain:
            max_added_gain = added_gain
            best_point_coords = safe_point
    if best_point_coords is None:
         candidate_pos = frontier_line.centroid.coords[0]
         try:
            view = Polygon(cast_rays_with_holes(environment_poly, candidate_pos)).simplify(TOLERANCE)
            if view.is_valid:
                 gain = view.difference(current_shared_map).area
                 return candidate_pos, gain if gain > 0 else 0.0
         except Exception: pass
         return candidate_pos, 0.0
    return best_point_coords, max_added_gain if max_added_gain > 0 else 0.0


class Drone:
    """A class to represent a single mobile drone with its own state."""
    def __init__(self, drone_id, initial_pos):
        self.id = drone_id
        self.pos = initial_pos
        self.status = 'available'
        self.parent = None
        self.vis_poly = None
    def __repr__(self):
        pos_str = f"({self.pos[0]:.2f}, {self.pos[1]:.2f})"
        return f"Drone {self.id} @ {pos_str} ({self.status})"

def update_plot(ax, polygon, drone_fleet, shared_map, title_text, frontiers=None, evaluated_tasks=None, winning_task=None):
    """Clears and redraws the Matplotlib plot with enhanced information."""
    ax.clear()
    ox, oy = polygon.exterior.xy
    ax.plot(ox, oy, 'k-', label='_nolegend_')
    ax.fill(ox, oy, color='gray', alpha=0.1, zorder=1)
    for interior in polygon.interiors:
        hx, hy = interior.xy
        ax.fill(hx, hy, color='white', edgecolor='black', linewidth=1, zorder=1.5) # zorder higher than main fill but lower than coverage

    if shared_map and hasattr(shared_map, 'geom_type'):
        if shared_map.geom_type == 'Polygon':
            # Check for validity before accessing exterior
            if shared_map.is_valid and hasattr(shared_map, 'exterior') and shared_map.exterior:
                 vx, vy = shared_map.exterior.xy
                 ax.fill(vx, vy, color='gold', alpha=0.5, label='Total Coverage', zorder=2)
        elif shared_map.geom_type == 'MultiPolygon':
            for i, poly in enumerate(shared_map.geoms):
                 if poly and poly.is_valid and hasattr(poly, 'exterior') and poly.exterior:
                    vx, vy = poly.exterior.xy
                    ax.fill(vx, vy, color='gold', alpha=0.5, label='Total Coverage' if i == 0 else '_nolegend_', zorder=2)
    for i, drone in enumerate(drone_fleet):
        if drone.status == 'stationary' and drone.parent:
            p_pos, c_pos = drone.parent.pos, drone.pos
            ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], 'k-', linewidth=1.5, zorder=3, label='Network Link' if i == 1 else '_nolegend_')

    # --- Plot temporary links using distance check ---
    stationary_nodes = [d for d in drone_fleet if d.status == 'stationary']
    if stationary_nodes:
        exploring_drones = [d for d in drone_fleet if d.status == 'exploring']
        for i, drone in enumerate(exploring_drones):
            exploring_pos_point = Point(drone.pos)
            # 1. Find all stationary drones whose vis_poly is VERY close to the explorer
            visible_stationary_parents = [s for s in stationary_nodes if s.vis_poly and s.vis_poly.is_valid and (s.vis_poly.distance(exploring_pos_point) < TOLERANCE * 2)]
            # 2. If any are close enough, find the closest one
            if visible_stationary_parents:
                closest_visible_parent = min(visible_stationary_parents, key=lambda s: Point(s.pos).distance(exploring_pos_point))
                p_pos, c_pos = closest_visible_parent.pos, drone.pos
                ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], 'gray', linestyle=':', linewidth=1, zorder=3, label='Exploring Link' if i == 0 else '_nolegend_')

    if frontiers:
        frontiers_list = list(frontiers.geoms) if isinstance(frontiers, MultiLineString) else [frontiers]
        for i, frontier in enumerate(frontiers_list):
            if frontier and not frontier.is_empty:
                x, y = frontier.xy
                ax.plot(x, y, 'c--', linewidth=2, label='Frontiers' if i == 0 else '_nolegend_', zorder=3)
    if evaluated_tasks:
        for task in evaluated_tasks:
             if 'pos' in task and task['pos']:
                ax.text(task['pos'][0], task['pos'][1] + 0.3, f"{task['gain']:.1f}", color='blue', fontsize=9, ha='center', fontweight='bold')
    if winning_task and frontiers:
        frontiers_list = list(frontiers.geoms) if isinstance(frontiers, MultiLineString) else [frontiers]
        winning_pos_point = Point(winning_task['pos'])
        for frontier in frontiers_list:
             if frontier and not frontier.is_empty and frontier.distance(winning_pos_point) < TOLERANCE * 5:
                x, y = frontier.xy
                ax.plot(x, y, color='gold', linestyle='-', linewidth=4, zorder=4, label='Winning Frontier')
                break
    for drone in drone_fleet:
        if drone.status == 'stationary': color, marker, size, label = 'red', 'o', 10, 'Stationary Drone'
        elif drone.status == 'exploring': color, marker, size, label = 'blue', 's', 8, 'Exploring Drone'
        else: color, marker, size, label = 'dimgray', 'x', 6, 'Available Drone'
        ax.plot(drone.pos[0], drone.pos[1], marker, color=color, markersize=size, label=label, zorder=5)
        ax.text(drone.pos[0] + 0.2, drone.pos[1] + 0.2, f'D{drone.id}', color=color, fontsize=10, fontweight='bold', zorder=6)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title_text, fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.draw()
    plt.pause(0.1)

# --- Main Execution ---
if __name__ == "__main__":
    outer_boundary = [ [0.29, 4.82], [3.03, 2.50], [4.67, 4.55], [8.12, 2.97], [9.5, 7.35], [6.96, 8.04], [6.28, 7.11], [8.33, 6.64], [6.72, 4.82], [3.08, 7.11], [4.88, 8.04], [2.11, 8.04], [1.90, 6.64], [3.95, 5.27], [2.82, 4.13], [1.68, 4.82], [2.58, 5.50] ]
    outer_boundary2 = [(0, 3),(0,11),(5,11),(5,12),(0,12),(0,21),(5,21),(5,16),(12,16),(12,17),(6,17),
                (6,21),(21,21),(21,17),(15,17),(15,16),(21,16),(21,9),(15,9),(15,6),(16,6),
                (16,7),(21,7),(21,0),(16,0),(16,1),(15,1),(15,0),(9,0),(9,8),(10,8),(10,9),
                (9,9),(9,12),(8,12),(8,9),(7,9),(7,8),(8,8),(8,0),(3,0),(3,3)]
    polygon = Polygon(outer_boundary2)  # CHANGE HERE TO TRY DIFFERENT ENVIRONMENTS
    start_pos1 = (5, 5)
    start_pos = (3,3)
    fleet_size = 8
    num_stationary_goal = 8

    drone_fleet = [Drone(i, start_pos1) for i in range(fleet_size)]  # DONT FORGET TO CHANGE START POSITION HERE
    first_drone = drone_fleet[0]
    first_drone.status = 'stationary'

    initial_points = cast_rays_with_holes(polygon, first_drone.pos)
    shared_map = None
    if len(initial_points) >= 3:
        try:
            initial_poly = Polygon(initial_points)
            if initial_poly.is_valid and initial_poly.area > TOLERANCE:
                shared_map = initial_poly.simplify(TOLERANCE)
                first_drone.vis_poly = shared_map
        except Exception as e: print(f"Init Error: {e}")
    if shared_map is None: exit("FATAL ERROR: Initial visibility polygon invalid.")

    stationary_drones = [first_drone]
    
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.ion()
    update_plot(ax, polygon, drone_fleet, shared_map, "Round 0: Initial State")
    print("--- Simulation Start ---")
    time.sleep(0.1)

    round_num = 1
    while len(stationary_drones) < num_stationary_goal:
        print(f"\n{'='*15} ROUND {round_num} {'='*15}")
        
        frontiers_geom = get_windows(polygon, shared_map)
        if frontiers_geom is None or frontiers_geom.is_empty: break
        
        tasks_geom = list(frontiers_geom.geoms) if isinstance(frontiers_geom, MultiLineString) else [frontiers_geom]
        
        # Filter out invalid or empty geometries first
        valid_tasks_geom_pre_filter = [geom for geom in tasks_geom if geom and not geom.is_empty]
        if not valid_tasks_geom_pre_filter:
             print("No valid frontiers remaining after initial filtering. Mission complete.")
             break

        min_frontier_length = TOLERANCE * 10 # Adjust this threshold if needed
        valid_tasks_geom = [geom for geom in valid_tasks_geom_pre_filter if geom.length > min_frontier_length]
        
        if not valid_tasks_geom:
             print(f"No frontiers longer than {min_frontier_length:.4f} found. Assuming area is covered. Mission complete.")
             break

        print(f"--- 1. SENSE: Found {len(valid_tasks_geom)} valid frontiers (after length filtering).")
        # Plot the original frontiers found
        update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Found {len(tasks_geom)} Frontiers", frontiers=frontiers_geom)
        time.sleep(0.1)

        print("--- 2. EVALUATE: Drones finding optimal points on frontiers.")
        broadcast_channel = []
        task_list_geom = list(valid_tasks_geom)

        while task_list_geom:
            mobile_drones = [d for d in drone_fleet if d.status != 'stationary']
            if not mobile_drones: break

            assignments = []
            available_drones_sub = list(mobile_drones)
            tasks_sub_geom = list(task_list_geom)

            while available_drones_sub and tasks_sub_geom:
                best_pair = { 'drone': None, 'task_geom': None, 'dist': float('inf') }
                for drone in available_drones_sub:
                    drone_point = Point(drone.pos)
                    for task_geom in tasks_sub_geom:
                        dist = drone_point.distance(task_geom)
                        if dist < best_pair['dist']:
                            best_pair.update({'dist': dist, 'drone': drone, 'task_geom': task_geom})
                
                if best_pair['drone']:
                    assignments.append(best_pair)
                    available_drones_sub.remove(best_pair['drone'])
                    tasks_sub_geom.remove(best_pair['task_geom'])
                else: break
            
            print(f"  - Assigning {len(assignments)} drone(s) to evaluate closest frontiers...")
            for assignment in assignments:
                drone, frontier_line = assignment['drone'], assignment['task_geom']
                
                best_point_on_frontier, gain = find_best_point_on_frontier(frontier_line, polygon, shared_map)
                
                if best_point_on_frontier is not None and gain > TOLERANCE :
                    drone.pos = best_point_on_frontier
                    drone.status = 'exploring'
                    broadcast_channel.append({'pos': best_point_on_frontier, 'gain': gain})
                    print(f"    - D{drone.id} evaluated frontier, best gain {gain:.2f} at {np.round(best_point_on_frontier, 2)}.")
                else:
                     print(f"    - D{drone.id} could not find a valid evaluation point for frontier.")
                     drone.status = 'available'

            assigned_geoms = [a['task_geom'] for a in assignments]
            task_list_geom = [geom for geom in task_list_geom if geom not in assigned_geoms]
            
            update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Evaluating...", frontiers=frontiers_geom, evaluated_tasks=broadcast_channel)
            time.sleep(0.1)

        print("--- 3. CONSENSUS: All drones independently decide the winner.")
        if not broadcast_channel: break
        
        best_task = max(broadcast_channel, key=lambda x: x['gain'])
        print(f"  - Consensus reached: Best optimal point is at {np.round(best_task['pos'], 2)} with a gain of {best_task['gain']:.2f}")

        print("--- 4. UPDATE: Closest mobile drone is re-tasked to become stationary.")
        mobile_drones = [d for d in drone_fleet if d.status != 'stationary']
        if not mobile_drones: break
        winner_drone = min(mobile_drones, key=lambda d: Point(d.pos).distance(Point(best_task['pos'])))
        
        winner_drone.pos, winner_drone.status = best_task['pos'], 'stationary'
        
        # --- Parent Finding using distance check ---
        winner_pos_point = Point(winner_drone.pos)
        # 1. Find stationary drones whose vis_poly is VERY close to the winner
        visible_parents = [d for d in stationary_drones if d.vis_poly and d.vis_poly.is_valid and (d.vis_poly.distance(winner_pos_point) < TOLERANCE * 2)]
        
        # 2. From that list, find the one that is closest.
        if visible_parents:
            winner_drone.parent = min(visible_parents, key=lambda p: Point(p.pos).distance(winner_pos_point))
        else: # Explicitly set parent to None if none found
             winner_drone.parent = None

        stationary_drones.append(winner_drone)
        for drone in drone_fleet:
            if drone.status == 'exploring': drone.status = 'available'
        
        winner_view_points = cast_rays_with_holes(polygon, winner_drone.pos)
        winner_view = None
        if len(winner_view_points) >= 3:
             try:
                 temp_poly = Polygon(winner_view_points)
                 if temp_poly.is_valid: winner_view = temp_poly.simplify(TOLERANCE)
             except Exception: pass

        if winner_view and winner_view.is_valid:
             winner_drone.vis_poly = winner_view
             try:
                 if shared_map and shared_map.is_valid:
                     updated_map = shared_map.union(winner_view)
                 else:
                      updated_map = winner_view
                 if updated_map.is_valid:
                     shared_map = updated_map
                 else:
                      buffered_map = updated_map.buffer(0)
                      if buffered_map.is_valid: shared_map = buffered_map
                      else: print("Error: buffer(0) failed. Skipping map update.")
             except Exception as e:
                 print(f"Error during map union: {e}. Skipping map update.")
        else:
             print(f"Warning: Winner drone {winner_drone.id} has invalid visibility polygon. Not updating map.")
        
        update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Drone {winner_drone.id} Placed", frontiers=frontiers_geom, evaluated_tasks=broadcast_channel, winning_task=best_task)
        
        round_num += 1
        time.sleep(0.1)
        
    print(f"\n{'='*15} MISSION COMPLETE {'='*15}")
    update_plot(ax, polygon, drone_fleet, shared_map, f"Mission Complete! ({len(stationary_drones)} Drones Placed)")
    plt.ioff()
    plt.show()