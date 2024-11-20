import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import random
from scipy.spatial import KDTree
import networkx as nx
import time

# Constants
k_a = 10.0  # Attractive potential constant
k_r = 100.0  # Repulsive potential constant
d_th = 2.0  # Threshold distance for repulsive potential
eta = 0.1  # Step size for gradient descent
robotLength = 1.0
robotWidth = 0.5

# Helper functions
def scene_from_file(filename):
    environment = {}
    polygons = []
    with open(filename, 'r') as file:
        lines = file.readlines()

        # Extract grid size
        grid_size_line = lines[0].strip()
        environment['gridSize'] = float(grid_size_line.split(": ")[1])

        # Parse the obstacles
        current_polygon = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("Obstacle"):
                if current_polygon:
                    # Append the current polygon to the list of polygons
                    polygons.append(np.array(current_polygon))
                current_polygon = []
            else:
                # Extract the coordinates and add them to the current polygon
                x, y = map(float, line.split(", "))
                current_polygon.append([x, y])

        # Append the last polygon
        if current_polygon:
            polygons.append(np.array(current_polygon))

    environment['obstacles'] = polygons
    return environment


def gradient_repulsive_potential(q, polygon):
    """
    Compute the gradient of the repulsive potential for a polygonal obstacle.
    """
    grad_U_r = np.zeros_like(q)
    min_distance = float('inf')
    closest_point = None

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        
        edge = p2 - p1
        t = np.dot(q - p1, edge) / np.dot(edge, edge)
        t = np.clip(t, 0, 1)
        point_on_edge = p1 + t * edge
        distance = np.linalg.norm(q - point_on_edge)

        if distance < min_distance:
            min_distance = distance
            closest_point = point_on_edge

    if min_distance <= d_th:
        grad_U_r = k_r * (1 / min_distance - 1 / d_th) * (-1 / min_distance**2) * (q - closest_point) / np.linalg.norm(q - closest_point)

    return grad_U_r

def gradient_potential(q, goal, obstacles):
    grad_U_a = k_a * (q - goal)  # Gradient of attractive potential
    grad_U_r = np.zeros_like(q)

    for polygon in obstacles:
        grad_U_r += gradient_repulsive_potential(q, polygon)

    return grad_U_a + grad_U_r


def collisionCheckFreeBody(newPolygon, existingPolygon):
    for obstacle in [newPolygon, existingPolygon]:
        for i in range(len(obstacle)):
            p1 = obstacle[i]
            p2 = obstacle[(i+1) % len(obstacle)]
            edge = p2 - p1
            ax = np.array([-edge[1], edge[0]])
            
            ax = ax / np.linalg.norm(ax)
            
            dot1 = np.dot(newPolygon, ax)
            dot2 = np.dot(existingPolygon, ax)
            poly1 = [np.min(dot1), np.max(dot1)]
            poly2 = [np.min(dot2), np.max(dot2)]
            
            if poly1[1] < poly2[0] or poly2[1] < poly1[0]:
                return False # no collision
    return True

def pathCollisionCheck(p1, p2, polygon):
    for i in range(len(polygon)):
        # Get current edge of the polygon
        edge_start = polygon[i]
        edge_end = polygon[(i + 1) % len(polygon)]
        
        # Check for intersection between the segment (p1, p2) and the edge
        if line_segment_intersection(p1, p2, edge_start, edge_end):
            return True
    return False

def line_segment_intersection(p1, p2, q1, q2):
    def orientation(a, b, c):
        # Calculate orientation of triplet (a, b, c)
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-9:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or counterclockwise
    
    def on_segment(a, b, c):
        # Check if point c lies on segment (a, b)
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and \
               min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    
    # Orientations
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, p2, q2): return True
    if o3 == 0 and on_segment(q1, q2, p1): return True
    if o4 == 0 and on_segment(q1, q2, p2): return True
    
    return False

def visualize_robot_path(q, robotCorners, obstacles, goal, path):
    fig, ax = plt.subplots()

    # Extract x, y positions for plotting the path
    xVals = [pose[0] for pose in path]
    yVals = [pose[1] for pose in path]

    # Setup 
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_title('Robot Path Animation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)

    ax.plot(xVals, yVals, 'k--', label='Path')
    
    #add obstacles 
    for obs in obstacles:
        poly = plt.Polygon(obs, color='gray')
        ax.add_patch(poly)

    #robot polygon
    robot_polygon = robotCorners + q
    robotPatch = plt.Polygon(robot_polygon, closed=True, fc='blue', ec='black')
    ax.add_patch(robotPatch)
    
    # Plot goal
    plt.plot(goal[0], goal[1], "gx", markersize=10, label="Goal")

    #updates the robot's positionper frame
    def update(frame):
        vX, vY= path[frame]
        robotPosition = robotCorners + np.array([vX, vY])
        robotPatch.set_xy(robotPosition)

        return robotPatch,

    ani = FuncAnimation(fig, update, frames=len(path), blit=True, interval=500, repeat=True)
    plt.legend()
    plt.show()

# Main
def main():
    global eta;
    parser = argparse.ArgumentParser(description='Potential Function Path Planning')
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--map', type=str, required=True)

    args = parser.parse_args()
    start = np.array(args.start)
    goal = np.array(args.goal)
    start_time = time.time()

    environment = scene_from_file(args.map)
    obstacles = environment['obstacles']
    q = start
    path = [q]
    robotCorners = np.array([
        [-(robotLength / 2), -(robotWidth / 2)],
        [(robotLength / 2), -(robotWidth / 2)],
        [(robotLength / 2), (robotWidth / 2)],
        [-(robotLength / 2), (robotWidth / 2)]
    ])

    i=0
    while np.linalg.norm(q - goal) > 0.01:
        grad = gradient_potential(q, goal, obstacles)
        q_new = q - eta * grad
        
        # Check path collisions
        collision = False
        for obs in obstacles:
            if pathCollisionCheck(q, q_new, obs):
                collision = True
                break
            
        if collision:
            eta *= 0.5
            if eta < 1e-4:
                print("Unable to find collision-free path.")
                return 
            continue 
        
        # Update robot corners
        q = q_new
        robot_polygon = robotCorners + q

        # Check for collisions
        for obs in obstacles:
            if collisionCheckFreeBody(robot_polygon, obs):
                print("Collision detected!")
                return

        path.append(q)
        i += 1
        if i > 5000:  # Prevent infinite loop
            print("Max iterations reached!")
            return

    visualize_robot_path(q, robotCorners, obstacles, goal, path)

if __name__ == "__main__":
    main()

