import math
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

# A small tolerance value used for geometric buffering operations
TOLERANCE = 0.001

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
    """Finds visibility windows (frontiers)."""

    if vis_polygon.is_empty:
        return LineString()
    multi_line_vis = vis_polygon.boundary
    environment_lines = polygon.boundary
    buffered_environment = environment_lines.buffer(TOLERANCE)
    window_segments = multi_line_vis.difference(buffered_environment)
    return window_segments

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
    if shared_map.geom_type == 'Polygon':
        vx, vy = shared_map.exterior.xy
        ax.fill(vx, vy, color='gold', alpha=0.5, label='Total Coverage', zorder=2)
    elif shared_map.geom_type == 'MultiPolygon':
        for i, poly in enumerate(shared_map.geoms):
            vx, vy = poly.exterior.xy
            ax.fill(vx, vy, color='gold', alpha=0.5, label='Total Coverage' if i == 0 else '_nolegend_', zorder=2)
    for i, drone in enumerate(drone_fleet):
        if drone.status == 'stationary' and drone.parent:
            p_pos, c_pos = drone.parent.pos, drone.pos
            ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], 'k-', linewidth=1.5, zorder=3, label='Network Link' if i == 1 else '_nolegend_')
    stationary_nodes = [d for d in drone_fleet if d.status == 'stationary']
    if stationary_nodes:
        exploring_drones = [d for d in drone_fleet if d.status == 'exploring']
        for i, drone in enumerate(exploring_drones):
            exploring_pos_point = Point(drone.pos)
            visible_stationary_parents = [s for s in stationary_nodes if s.vis_poly and s.vis_poly.buffer(TOLERANCE).contains(exploring_pos_point)]
            if visible_stationary_parents:
                closest_visible_parent = min(visible_stationary_parents, key=lambda s: Point(s.pos).distance(exploring_pos_point))
                p_pos, c_pos = closest_visible_parent.pos, drone.pos
                ax.plot([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]], 'gray', linestyle=':', linewidth=1, zorder=3, label='Exploring Link' if i == 0 else '_nolegend_')
    if frontiers:
        for i, frontier in enumerate(frontiers):
            x, y = frontier.xy
            ax.plot(x, y, 'c--', linewidth=2, label='Frontiers' if i == 0 else '_nolegend_', zorder=3)
    if evaluated_tasks:
        for task in evaluated_tasks:
            ax.text(task['pos'][0], task['pos'][1] + 0.3, f"{task['gain']:.1f}", color='blue', fontsize=9, ha='center', fontweight='bold')
    if winning_task and frontiers:
        for frontier in frontiers:
            if frontier.centroid.coords[0] == winning_task['pos']:
                x, y = frontier.xy
                ax.plot(x, y, color='gold', linestyle='-', linewidth=4, zorder=4, label='Winning Frontier')
                break
    for drone in drone_fleet:
        if drone.status == 'stationary': color, marker, size, label = 'red', 'o', 5, 'Stationary Drone'
        elif drone.status == 'exploring': color, marker, size, label = 'blue', 's', 5, 'Exploring Drone'
        else: color, marker, size, label = 'dimgray', 'x', 6, 'Available Drone'
        ax.plot(drone.pos[0], drone.pos[1], marker, color=color, markersize=size, label=label, zorder=5)
        ax.text(drone.pos[0] + 0.2, drone.pos[1] + 0.2, f'D{drone.id}', color=color, fontsize=10, fontweight='bold', zorder=6)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(title_text, fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.draw()
    plt.pause(0.1)

# --- Main Execution ---
if __name__ == "__main__":
    outer_boundary = [
        [0.29, 4.82], [3.03, 2.50], [4.67, 4.55], [8.12, 2.97],
        [9.5, 7.35], [6.96, 8.04], [6.28, 7.11], [8.33, 6.64],
        [6.72, 4.82], [3.08, 7.11], [4.88, 8.04], [2.11, 8.04],
        [1.90, 6.64], [3.95, 5.27], [2.82, 4.13], [1.68, 4.82],
        [2.58, 5.50]
    ]
    outer_boundary2 = [(0, 3),(0,11),(5,11),(5,12),(0,12),(0,21),(5,21),(5,16),(12,16),(12,17),(6,17),
                (6,21),(21,21),(21,17),(15,17),(15,16),(21,16),(21,9),(15,9),(15,6),(16,6),
                (16,7),(21,7),(21,0),(16,0),(16,1),(15,1),(15,0),(9,0),(9,8),(10,8),(10,9),
                (9,9),(9,12),(8,12),(8,9),(7,9),(7,8),(8,8),(8,0),(3,0),(3,3)]
    polygon = Polygon(outer_boundary2)
    start_pos = (10, 15)
    #start_pos = (125,500)
    fleet_size = 5 # The total number of drones in our fixed fleet
    num_stationary_goal = 5 # The mission ends when this many drones are stationary

    # --- Initialization with a Fixed Fleet ---
    # The entire fleet exists from the start.
    drone_fleet = [Drone(i, start_pos) for i in range(fleet_size)]
    
    # Bootstrap the process by placing the first drone.
    first_drone = drone_fleet[0]
    first_drone.status = 'stationary'
    first_drone.vis_poly = Polygon(cast_rays_with_holes(polygon, first_drone.pos)).simplify(0.01)
    
    stationary_drones = [first_drone]
    shared_map = first_drone.vis_poly
    
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.ion()
    update_plot(ax, polygon, drone_fleet, shared_map, "Round 0: Initial State")
    print("--- Simulation Start ---")
    time.sleep(0.5)

    round_num = 1
    while len(stationary_drones) < num_stationary_goal:
        print(f"\n{'='*15} ROUND {round_num} {'='*15}")
        
        frontiers_geom = get_windows(polygon, shared_map)
        if frontiers_geom.is_empty: break
        tasks_geom = list(frontiers_geom.geoms) if frontiers_geom.geom_type == 'MultiLineString' else [frontiers_geom]
        task_list = [frontier.interpolate(0.5, normalized=True).coords[0] for frontier in tasks_geom]
        print(f"--- 1. SENSE: Found {len(task_list)} frontiers.")
        update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Found {len(task_list)} Frontiers", frontiers=tasks_geom)
        time.sleep(0.5)

        print("--- 2. EVALUATE: Drones patrolling and broadcasting findings.")
        broadcast_channel = [] # This simulates the network message board
        while task_list:
            mobile_drones = [d for d in drone_fleet if d.status != 'stationary']
            assignments = []
            available_drones_sub = list(mobile_drones)
            tasks_sub = list(task_list)
            while available_drones_sub and tasks_sub:
                best_pair = { 'drone': None, 'task_pos': None, 'dist': float('inf') }
                for drone in available_drones_sub:
                    for task_pos in tasks_sub:
                        dist = Point(drone.pos).distance(Point(task_pos))
                        if dist < best_pair['dist']:
                            best_pair.update({'dist': dist, 'drone': drone, 'task_pos': task_pos})
                if best_pair['drone']:
                    assignments.append(best_pair)
                    available_drones_sub.remove(best_pair['drone'])
                    tasks_sub.remove(best_pair['task_pos'])
            
            print(f"  - Assigning {len(assignments)} drone(s) to their closest frontiers...")
            for assignment in assignments:
                drone, task_pos = assignment['drone'], assignment['task_pos']
                drone.pos, drone.status = task_pos, 'exploring'
                view = Polygon(cast_rays_with_holes(polygon, drone.pos))
                gain = view.difference(shared_map).area
                # The drone "broadcasts" its finding to the channel for all to see
                broadcast_channel.append({'pos': task_pos, 'gain': gain})
                print(f"    - D{drone.id} travels {assignment['dist']:.2f} units and broadcasts gain of {gain:.2f}.")

            task_list = [t for t in task_list if t not in [a['task_pos'] for a in assignments]]
            update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Evaluating...", frontiers=tasks_geom, evaluated_tasks=broadcast_channel)
            time.sleep(0.5)

        # ---  CONSENSUS PHASE ---
        print("--- 3. CONSENSUS: All drones independently decide the winner.")
        if not broadcast_channel: break
        
        # In a real system, EVERY drone would run this 'max' function on the messages it received.
        # Since they all have the same data and logic, they all arrive at the same conclusion.
        best_task = max(broadcast_channel, key=lambda x: x['gain'])
        print(f"  - Consensus reached: Best frontier is at {np.round(best_task['pos'], 2)} with a gain of {best_task['gain']:.2f}")

        # ---  UPDATE PHASE with Re-Tasking ---
        print("--- 4. UPDATE: Closest mobile drone is re-tasked to become stationary.")
        mobile_drones = [d for d in drone_fleet if d.status != 'stationary']
        # Find the best drone from the available pool to take on the role
        winner_drone = min(mobile_drones, key=lambda d: Point(d.pos).distance(Point(best_task['pos'])))
        
        # The winning drone moves and changes its own role
        winner_drone.pos, winner_drone.status = best_task['pos'], 'stationary'
        
        visible_parents = [d for d in stationary_drones if d.vis_poly and d.vis_poly.buffer(TOLERANCE).contains(Point(winner_drone.pos))]
        if visible_parents:
            winner_drone.parent = min(visible_parents, key=lambda p: Point(p.pos).distance(Point(winner_drone.pos)))
        
        stationary_drones.append(winner_drone)
        for drone in drone_fleet:
            if drone.status == 'exploring': drone.status = 'available'
        
        winner_view = Polygon(cast_rays_with_holes(polygon, winner_drone.pos)).simplify(0.01)
        winner_drone.vis_poly = winner_view
        shared_map = shared_map.union(winner_view)
        
        update_plot(ax, polygon, drone_fleet, shared_map, f"Round {round_num}: Drone {winner_drone.id} Placed", frontiers=tasks_geom, evaluated_tasks=broadcast_channel, winning_task=best_task)
        
        round_num += 1
        time.sleep(0.5)
        
    print(f"\n{'='*15} MISSION COMPLETE {'='*15}")
    update_plot(ax, polygon, drone_fleet, shared_map, f"Mission Complete! ({len(stationary_drones)} Drones Placed)")
    plt.ioff()
    plt.show()