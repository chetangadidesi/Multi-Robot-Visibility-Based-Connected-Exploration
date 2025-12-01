import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
from tqdm import tqdm

# A small tolerance value used for geometric operations
TOLERANCE = 1e-6
# How many points to sample inside the initial visible area
NUM_AREA_SAMPLES = 5000 
# How far apart to sample points along the frontier
FRONTIER_STEP_SIZE = 0.25

# --- 1. CORE VISIBILITY FUNCTIONS---

def cast_rays_with_holes(polygon, viewpoint):
    """
    Calculates the visibility polygon from a viewpoint by casting rays towards every vertex.
    """
    ray_points = []
    angles = []
    
    # Collect all vertices and edges from the exterior and all interiors
    all_vertices = list(polygon.exterior.coords)
    edges = list(zip(all_vertices[:-1], all_vertices[1:]))

    for interior in polygon.interiors:
        interior_coords = list(interior.coords)
        all_vertices.extend(interior_coords)
        edges.extend(zip(interior_coords[:-1], interior_coords[1:]))

    # For each vertex, cast three rays: one at the exact angle, and two
    # slightly perturbed to ensure we catch vertices and edge cases.
    for vx, vy in all_vertices:
        angle = math.atan2(vy - viewpoint[1], vx - viewpoint[0])
        angles.append(angle - TOLERANCE)
        angles.append(angle)
        angles.append(angle + TOLERANCE)
    
    angles.sort()

    # Cast each ray and find the closest intersection
    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        # Start the ray just slightly outside the viewpoint to avoid self-intersection
        near_point = (viewpoint[0] + dx * TOLERANCE, viewpoint[1] + dy * TOLERANCE)
        
        # Ensure the ray starts inside the polygon
        if not polygon.contains(Point(near_point)):
            ray_points.append(viewpoint)
            continue

        # Create a very long ray
        far_point = (viewpoint[0] + dx * 2000, viewpoint[1] + dy * 2000) # Increased length
        ray = LineString([near_point, far_point])
        
        min_dist = float('inf')
        closest_pt = None
        
        # Check intersection with all edges
        for seg_start, seg_end in edges:
            edge = LineString([seg_start, seg_end])
            if ray.intersects(edge):
                intersection = ray.intersection(edge)
                # Handle cases where intersection might be a line (collinear)
                if isinstance(intersection, Point):
                    dist = Point(viewpoint).distance(intersection)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pt = intersection
                elif isinstance(intersection, (LineString, MultiLineString)):
                    # If collinear, take the closest point of the intersection
                    for pt in intersection.coords:
                        dist = Point(viewpoint).distance(Point(pt))
                        if dist < min_dist:
                            min_dist = dist
                            closest_pt = Point(pt)

        if closest_pt:
            # Rounding to fixed precision can help merge very close points
            ray_points.append((round(closest_pt.x, 6), round(closest_pt.y, 6)))
        else:
            ray_points.append(far_point)
            
    return ray_points

def get_windows(polygon, vis_polygon, tolerance=1e-3):
    """
    A method to find visibility windows (frontiers) 
    """
    window_segments = []
    
    # Create a single geometry representing all "walls"
    env_lines = unary_union(
        [LineString(polygon.exterior.coords)] + 
        [LineString(interior.coords) for interior in polygon.interiors]
    )
    # Buffer the walls slightly to create a thickness
    buffered_env_lines = env_lines.buffer(tolerance)

    # Get all segments of the visibility polygon's boundary
    vis_boundary_coords = vis_polygon.exterior.coords
    vis_segments = [
        LineString([vis_boundary_coords[i], vis_boundary_coords[i+1]]) 
        for i in range(len(vis_boundary_coords) - 1)
    ]

    for seg in vis_segments:
        # A window is a segment whose midpoint is NOT touching a wall.
        if not seg.centroid.within(buffered_env_lines):
            window_segments.append(seg)
            
    if not window_segments:
        return LineString()

    return unary_union(window_segments)

def generate_random_points_in_polygon(polygon, num_points):
    """
    Generates a list of random Shapely Point objects inside a given polygon.
    """
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    while len(points) < num_points:
        random_point = Point(
            np.random.uniform(min_x, max_x), 
            np.random.uniform(min_y, max_y)
        )
        if polygon.contains(random_point):
            points.append(random_point)
    return points

def generate_points_on_lines(lines, distance_step):
    """
    Generates evenly spaced points along a LineString or MultiLineString.
    """
    points = []
    if lines.is_empty:
        return points

    lines_list = list(lines.geoms) if isinstance(lines, MultiLineString) else [lines]
    
    for line in lines_list:
        for dist in np.arange(0, line.length, distance_step):
            points.append(line.interpolate(dist))
        # Add the last point of each line
        if line.length > 0:
            points.append(line.boundary.geoms[1])
            
    return points

# --- 2. ANALYSIS FUNCTIONS  ---

def calculate_visibility(polygon, viewpoint):
    """
    Helper function to calculate visibility and handle polygon validity.
    Returns a valid Shapely Polygon or None.
    """
    vis_points = cast_rays_with_holes(polygon, viewpoint)
    if len(vis_points) < 3:
        return None
    
    try:
        # Create polygon and use buffer(0) to fix any invalid geometry
        poly = Polygon(vis_points).buffer(0)
        
        # Simplify to clean up tiny artifacts
        poly = poly.simplify(TOLERANCE, preserve_topology=True)
        
        # Ensure the result is a simple, valid polygon
        if poly.is_empty or not poly.is_valid or not isinstance(poly, Polygon):
            return None
            
        return poly
    except Exception as e:
        print(f"Warning: Could not create valid polygon from viewpoint {viewpoint}. Error: {e}")
        return None

def find_best_point_in_area(env_poly, initial_vis_poly, num_samples):
    """
    Samples points within the initial visibility polygon to find the one
    that provides the maximum added visibility.
    """
    print(f"Sampling {num_samples} random points in the visible area...")
    sample_points = generate_random_points_in_polygon(initial_vis_poly, num_samples)
    
    best_added_area = -1
    best_point = None
    best_vis_poly = None
    
    all_scores = []
    all_locations = []

    for point in tqdm(sample_points, desc="Analyzing area points"):
        vp = (point.x, point.y)
        current_vis_poly = calculate_visibility(env_poly, vp)
        
        if current_vis_poly is None:
            all_scores.append(0)
            all_locations.append(vp)
            continue
        
        added_region = current_vis_poly.difference(initial_vis_poly)
        added_area = added_region.area
        
        all_scores.append(added_area)
        all_locations.append(vp)
        
        if added_area > best_added_area:
            best_added_area = added_area
            best_point = vp
            best_vis_poly = current_vis_poly
            
    return best_point, best_vis_poly, all_locations, all_scores

def find_best_point_on_frontier(env_poly, initial_vis_poly, windows, step_size):
    """
    Samples points along the frontier (windows) to find the one
    that provides the maximum added visibility.
    """
    print(f"Sampling points every {step_size}m along the frontier...")
    window_points = generate_points_on_lines(windows, distance_step=step_size)
    
    if not window_points:
        print("No windows found to analyze.")
        return None, None

    best_added_area = -1
    best_point = None  # The point on the window
    best_vis_poly = None
    
    initial_vis_centroid = initial_vis_poly.centroid

    for point in tqdm(window_points, desc="Analyzing frontier points"):
        # Create a "safe point" just inside the window to see from
        direction_inward = LineString([point, initial_vis_centroid])
        safe_point_geom = direction_inward.interpolate(TOLERANCE * 10)
        safe_point = (safe_point_geom.x, safe_point_geom.y)

        current_vis_poly = calculate_visibility(env_poly, safe_point)
        
        if current_vis_poly is None:
            continue
            
        added_region = current_vis_poly.difference(initial_vis_poly)
        added_area = added_region.area
        
        if added_area > best_added_area:
            best_added_area = added_area
            best_point = (point.x, point.y) # Store the *actual* window point
            best_vis_poly = current_vis_poly
            
    return best_point, best_vis_poly

# --- 3. PLOTTING FUNCTIONS ---

def plot_environment(ax, env_poly):
    """Helper function to plot the base map (boundary and holes)."""
    ox, oy = env_poly.exterior.xy
    ax.plot(ox, oy, 'k-', label='Environment Boundary')
    ax.fill(ox, oy, color='gray', alpha=0.1)
    
    for hole in env_poly.interiors:
        hx, hy = hole.xy
        ax.fill(hx, hy, color='white', edgecolor='black', linewidth=1)

def plot_step_1_initial_visibility(env_poly, vis_poly, windows, viewpoint, percent_covered):
    """Generates Plot 1: Initial Visibility and Frontiers."""
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax, env_poly)
    
    # Plot visibility polygon
    vx, vy = vis_poly.exterior.xy
    ax.fill(vx, vy, color='lightgreen', alpha=0.6, label='Initial Visibility')
    
    # Plot windows
    if not windows.is_empty:
        if isinstance(windows, MultiLineString):
            for i, line in enumerate(windows.geoms):
                px, py = line.xy
                ax.plot(px, py, "m-", linewidth=3, label='Visibility Window' if i == 0 else "")
        elif isinstance(windows, LineString):
            px, py = windows.xy
            ax.plot(px, py, "m-", linewidth=3, label='Frontier (Window)')
            
    # Plot viewpoint
    ax.plot(*viewpoint, 'ro', markersize=8, label='Home Viewpoint')
    
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Visibility Windows from Home Location\nTotal Coverage: {percent_covered:.2f}%")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_step_2_area_sampling(env_poly, initial_vis_poly, best_point, 
                              sample_locations, added_scores, total_percent_covered):
    """Generates Plot 2: Heatmap of Added Visibility from Area Samples."""
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_environment(ax, env_poly)
    
    # Plot initial visibility as a dashed boundary
    ivx, ivy = initial_vis_poly.exterior.xy
    ax.plot(ivx, ivy, color='blue', linestyle='--', alpha=0.7, label='Initial Visibility Region')
    
    # Plot heatmap
    x_coords = [loc[0] for loc in sample_locations]
    y_coords = [loc[1] for loc in sample_locations]
    # Use added_scores (area) directly for the color
    scatter = ax.scatter(x_coords, y_coords, c=added_scores, cmap='viridis', s=40, zorder=5, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Added Visible Area (m²)', rotation=270, labelpad=20)
    
    # Plot best point
    if best_point:
        ax.plot(*best_point, 'o', markerfacecolor='cyan', markeredgecolor='black', 
                markersize=10, zorder=6, label=f'Best Sample Point')
    
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Plot 2: Best Point from Area Sampling Heatmap\n(Total Coverage from Best Point: {total_percent_covered:.2f}%)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- [ MODIFIED PLOTTING FUNCTION ] ---
def plot_step_2_detail_area(env_poly, initial_vis_poly, best_point, 
                              best_vis_poly, total_percent_covered, home_viewpoint): 
    """
    Generates Plot 3: Detail view of the best point from AREA sampling,
    showing initial and added visibility.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax, env_poly)
    
    # Plot initial region
    ivx, ivy = initial_vis_poly.exterior.xy
    ax.fill(ivx, ivy, color='lightgreen', alpha=0.4, label='Initial Visible Area')

    # Highlight the added area in red
    added_region = best_vis_poly.difference(initial_vis_poly)
    if added_region.geom_type == 'Polygon':
        arx, ary = added_region.exterior.xy
        ax.fill(arx, ary, color='red', alpha=0.6, label='Added Visible Area')
    elif added_region.geom_type == 'MultiPolygon':
        for i, poly in enumerate(added_region.geoms):
            arx, ary = poly.exterior.xy
            ax.fill(arx, ary, color='red', alpha=0.6, label='Added Visible Area' if i == 0 else "")

    # Plot the best point
    ax.plot(*best_point, 'o', markerfacecolor='cyan', markeredgecolor='black', 
            markersize=10, zorder=6, label=f'Best Area Sample Point')
    
    # Plot the home viewpoint as a cross
    ax.plot(*home_viewpoint, 'k+', markersize=10, markeredgewidth=2, label='Home Viewpoint')
    
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Visibility from Best Area Sample Point\nTotal Coverage: {total_percent_covered:.2f}%")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()



def plot_step_3_frontier_sampling(env_poly, initial_vis_poly, best_point, 
                                  best_vis_poly, total_percent_covered, home_viewpoint):
    """
    Generates Plot 4: Visibility from the Best Frontier Point,
    showing initial and added visibility.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_environment(ax, env_poly)
    
    # Plot initial region for context
    ivx, ivy = initial_vis_poly.exterior.xy
    ax.fill(ivx, ivy, color='lightgreen', alpha=0.4, label='Initial Visible Area')

    # Highlight the added area in red
    added_region = best_vis_poly.difference(initial_vis_poly)
    if added_region.geom_type == 'Polygon':
        arx, ary = added_region.exterior.xy
        ax.fill(arx, ary, color='red', alpha=0.6, label='Added Visible Area')
    elif added_region.geom_type == 'MultiPolygon':
        for i, poly in enumerate(added_region.geoms):
            arx, ary = poly.exterior.xy
            ax.fill(arx, ary, color='red', alpha=0.6, label='Added Visible Area' if i == 0 else "")

    # Plot the best point on the frontier
    ax.plot(*best_point, 'o', markerfacecolor='crimson', markeredgecolor='black', 
            markersize=10, zorder=6, label=f'Best Frontier Point')
    
    # Plot the home viewpoint as a cross
    ax.plot(*home_viewpoint, 'k+', markersize=10, markeredgewidth=2, label='Home Viewpoint') 
    
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Visibility from Best Frontier Sample Point\nTotal Coverage from Best Point: {total_percent_covered:.2f}%")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()



# --- 4. MAIN EXECUTION ---

if __name__ == "__main__":
    
    # --- Define Environment --- CHANGE THE BOUNDARY HERE

    outer_boundary = [
        [-5,2.5], [10,5],[12.5,20], [15,5],  [17.5,20], [20,5], [35,2.5],
        [20, 0], [17.5,-15], [15,0],[12.5,-15], [10,0],  [-5,2.5] 
    ]
    # 44.21 %
    outer_boundary2 = [
        [10,5], [12,4], [11.25,1], [12.875,1], [12.5, 4], [12.8415,4], [15.375, 0.5], [17,1], [12.375, 5],
        [17,9], [15.375,9.625] , [12.8415,6], [12.5,6], [12.75,9.25], [11.25, 9.125], [12,6], [10,5]
    ]
    holes = []
    
    environment_poly = Polygon(outer_boundary2, holes)  # CHANGE THE BOUNDARY HERE
    total_env_area = environment_poly.area
    
    home_viewpoint = (10.5, 5)

    print(f"Total Environment Area: {total_env_area:.2f} m²")

    # --- Step 1: Initial Visibility and Frontiers ---
    print("\n--- STEP 1: Calculating Initial Visibility ---")
    initial_vis_poly = calculate_visibility(environment_poly, home_viewpoint)
    
    if initial_vis_poly:
        windows = get_windows(environment_poly, initial_vis_poly)
        initial_percent = (initial_vis_poly.area / total_env_area) * 100
        
        print(f"Initial visibility covers {initial_vis_poly.area:.2f} m² ({initial_percent:.2f}%)")
        print(f"Found {len(windows.geoms) if isinstance(windows, MultiLineString) else 1} window segment(s).")

        plot_step_1_initial_visibility(
            environment_poly, initial_vis_poly, windows, home_viewpoint, initial_percent
        )
    else:
        print("Error: Could not calculate initial visibility. Exiting.")
        exit()

    # --- Step 2: Find Best Point in Visible Area ---
    print("\n--- STEP 2: Finding Best Point in Visible Area ---")
    best_area_pt, best_area_vis_poly, samples, scores = find_best_point_in_area(
        environment_poly, initial_vis_poly, NUM_AREA_SAMPLES
    )
    
    if best_area_pt:
        total_area_percent = (best_area_vis_poly.area / total_env_area) * 100
        added_area = best_area_vis_poly.difference(initial_vis_poly).area
        added_percent = (added_area / total_env_area) * 100

        print(f"Best area point found at {best_area_pt}.")
        print(f"  > Adds {added_area:.2f} m² ({added_percent:.2f}%)")
        print(f"  > New total coverage: {best_area_vis_poly.area:.2f} m² ({total_area_percent:.2f}%)")
        
        # Plot 2: The Heatmap
        plot_step_2_area_sampling(
            environment_poly, initial_vis_poly, best_area_pt, 
            samples, scores, total_area_percent
        )
        

        # Plot 3: The Detailed View for the Best Area Point
        plot_step_2_detail_area(
            environment_poly, initial_vis_poly, best_area_pt,
            best_area_vis_poly, total_area_percent, home_viewpoint 
        )

        
    else:
        print("Could not find a best sampling point in the area.")

    # --- Step 3: Find Best Point on Frontier ---
    print("\n--- STEP 3: Finding Best Point on Frontier ---")
    best_frontier_pt, best_frontier_vis_poly = find_best_point_on_frontier(
        environment_poly, initial_vis_poly, windows, FRONTIER_STEP_SIZE
    )
    
    if best_frontier_pt:
        total_frontier_percent = (best_frontier_vis_poly.area / total_env_area) * 100
        added_area = best_frontier_vis_poly.difference(initial_vis_poly).area
        added_percent = (added_area / total_env_area) * 100

        print(f"Best frontier point found at {best_frontier_pt}.")
        print(f"  > Adds {added_area:.2f} m² ({added_percent:.2f}%)")
        print(f"  > New total coverage: {best_frontier_vis_poly.area:.2f} m² ({total_frontier_percent:.2f}%)")


        plot_step_3_frontier_sampling(
            environment_poly, initial_vis_poly, best_frontier_pt, 
            best_frontier_vis_poly, total_frontier_percent, home_viewpoint 
        )

    else:
        print("Could not find a best sampling point on the frontier.")

    print("\nAnalysis complete.")