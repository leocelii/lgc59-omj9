import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import matplotlib.animation as animation

def wrap_angle(angle):
    """Wraps angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Trajectory Optimization in SE(2)')
    parser.add_argument('--start', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='Start pose (x y theta)')
    parser.add_argument('--goal', nargs=3, type=float, default=[5.0, 5.0, 1.57], help='Goal pose (x y theta)')
    parser.add_argument('--T', type=int, default=50, help='Number of time steps')
    parser.add_argument('--x0', nargs=3, type=float, default=[-1.0, 2.0, 0.78], help='Intermediate pose 1 (x y theta)')
    parser.add_argument('--x1', nargs=3, type=float, default=[3.0, 7.0, 3.14], help='Intermediate pose 2 (x y theta)')
    args = parser.parse_args()

    # Parameters
    dt = 0.1  # Time step
    T = args.T
    start = np.array(args.start)
    goal = np.array(args.goal)
    x0_in = np.array(args.x0)
    x1_in = np.array(args.x1)

    # Factor Graph Representation
    class FactorGraph:
        def __init__(self, T, start, goal, dt):
            self.T = T
            self.start = start
            self.goal = goal
            self.dt = dt
            self.num_states = T + 1  # States from t=0 to t=T
            self.num_controls = T    # Controls from t=0 to t=T-1
            self.factors = []

        def add_factor(self, factor):
            self.factors.append(factor)

        def compute_total_cost(self, variables):
            # Extract poses and controls from variables
            q = variables[:self.num_states * 3].reshape(self.num_states, 3)
            u = variables[self.num_states * 3:].reshape(self.num_controls, 3)
            total_cost = 0
            for factor in self.factors:
                total_cost += factor(q, u)
            return total_cost

    # Factors
    def dynamics_factor(dt, t):
        """Dynamics factor: Enforces q_{t+1} = q_t + u_t * dt."""
        def factor(q, u):
            q_t = q[t]
            q_t1 = q[t + 1]
            u_t = u[t]
            # Predicted next state
            q_pred = q_t + u_t * dt
            # Wrap the angle
            q_pred[2] = wrap_angle(q_pred[2])
            # Compute residual
            residual = q_t1 - q_pred
            # Wrap angle residual
            residual[2] = wrap_angle(residual[2])
            return np.sum(residual**2)
        return factor

    def start_factor(start):
        """Start factor: Enforces q_0 = start."""
        def factor(q, u):
            residual = q[0] - start
            residual[2] = wrap_angle(residual[2])
            return np.sum(residual**2) * 1e6  # Large weight to enforce equality
        return factor

    def goal_factor(goal):
        """Goal factor: Enforces q_T = goal."""
        def factor(q, u):
            residual = q[-1] - goal
            residual[2] = wrap_angle(residual[2])
            return np.sum(residual**2) * 1e6  # Large weight to enforce equality
        return factor

    def intermediate_factor(intermediate_pose, t):
        """Intermediate pose constraint: Enforces q_t = intermediate_pose."""
        def factor(q, u):
            residual = q[t] - intermediate_pose
            residual[2] = wrap_angle(residual[2])
            return np.sum(residual**2) * 1e6  # Large weight
        return factor

    def control_cost_factor(t):
        """Control cost: Penalizes large control inputs."""
        def factor(q, u):
            u_t = u[t]
            return np.sum(u_t**2)
        return factor

    def acceleration_cost_factor(dt, t):
        """Acceleration cost: Penalizes changes in control (accelerations)."""
        def factor(q, u):
            u_t = u[t]
            u_t1 = u[t + 1]
            acceleration = (u_t1 - u_t) / dt
            return np.sum(acceleration**2)
        return factor

    # Initialize Factor Graph
    factor_graph = FactorGraph(T, start, goal, dt)

    # Add factors
    factor_graph.add_factor(start_factor(start))  # Start constraint
    factor_graph.add_factor(goal_factor(goal))    # Goal constraint
    factor_graph.add_factor(intermediate_factor(x0_in, T // 3))  # First intermediate state
    factor_graph.add_factor(intermediate_factor(x1_in, 2 * T // 3))  # Second intermediate state
    for t in range(T):
        factor_graph.add_factor(control_cost_factor(t))  # Control cost
        factor_graph.add_factor(dynamics_factor(dt, t))  # Dynamics constraint
    for t in range(T - 1):
        factor_graph.add_factor(acceleration_cost_factor(dt, t))  # Acceleration cost

    # Initial guess: Straight line with intermediate points
    waypoints = [start, x0_in, x1_in, goal]
    waypoint_times = [0, T // 3, 2 * T // 3, T]
    initial_positions = np.zeros((factor_graph.num_states, 3))

    for i in range(len(waypoints) - 1):
        start_idx = waypoint_times[i]
        end_idx = waypoint_times[i + 1]
        num_points = end_idx - start_idx + 1
        for j in range(3):  # For x, y, theta
            initial_positions[start_idx:end_idx + 1, j] = np.linspace(waypoints[i][j], waypoints[i + 1][j], num_points)
    # Wrap angles
    initial_positions[:, 2] = wrap_angle(initial_positions[:, 2])

    # Initial guess for controls
    initial_controls = np.diff(initial_positions, axis=0) / dt
    # Wrap angle velocities
    initial_controls[:, 2] = wrap_angle(initial_controls[:, 2])

    # Flatten initial guess
    initial_variables = np.hstack((initial_positions.flatten(), initial_controls.flatten()))

    # Optimize trajectory
    result = minimize(
        factor_graph.compute_total_cost,
        initial_variables,
        method="SLSQP",
        options={"disp": True, "maxiter": 2000}
    )

    # Extract optimized positions and controls
    optimized_positions = result.x[:factor_graph.num_states * 3].reshape(factor_graph.num_states, 3)
    optimized_controls = result.x[factor_graph.num_states * 3:].reshape(factor_graph.num_controls, 3)

    # Wrap angles
    optimized_positions[:, 2] = wrap_angle(optimized_positions[:, 2])
    optimized_controls[:, 2] = wrap_angle(optimized_controls[:, 2])

    # Validation
    if result.success:
        print("Optimization succeeded.")
    else:
        print("Optimization failed:", result.message)

    print("Start matches:", np.allclose(optimized_positions[0], start, atol=1e-2))
    print("Goal matches:", np.allclose(optimized_positions[-1], goal, atol=1e-2))
    print("Intermediate 1 matches:", np.allclose(optimized_positions[T // 3], x0_in, atol=1e-2))
    print("Intermediate 2 matches:", np.allclose(optimized_positions[2 * T // 3], x1_in, atol=1e-2))

    # Plot trajectory
    plt.figure()
    plt.plot(initial_positions[:, 0], initial_positions[:, 1], label='Initial Trajectory', linestyle='--')
    plt.plot(optimized_positions[:, 0], optimized_positions[:, 1], label='Optimized Trajectory')
    plt.scatter([start[0], x0_in[0], x1_in[0], goal[0]],
                [start[1], x0_in[1], x1_in[1], goal[1]],
                c='red', label='Key Points', zorder=5)
    plt.title('Trajectory Optimization in SE(2)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

    # Plot orientation over time
    time_steps = np.arange(factor_graph.num_states) * dt
    plt.figure()
    plt.plot(time_steps, optimized_positions[:, 2], label='Theta (Optimized)')
    plt.title('Orientation over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (rad)')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot velocity profiles
    time_steps_u = np.arange(factor_graph.num_controls) * dt

    plt.figure()
    plt.plot(time_steps_u, optimized_controls[:, 0], label='Velocity X')
    plt.plot(time_steps_u, optimized_controls[:, 1], label='Velocity Y')
    plt.plot(time_steps_u, optimized_controls[:, 2], label='Angular Velocity')
    plt.title('Control Inputs Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity / Angular Velocity')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot acceleration profiles
    accelerations = np.diff(optimized_controls, axis=0) / dt
    time_steps_acc = np.arange(factor_graph.num_controls - 1) * dt

    plt.figure()
    plt.plot(time_steps_acc, accelerations[:, 0], label='Acceleration X')
    plt.plot(time_steps_acc, accelerations[:, 1], label='Acceleration Y')
    plt.plot(time_steps_acc, accelerations[:, 2], label='Angular Acceleration')
    plt.title('Accelerations Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration / Angular Acceleration')
    plt.legend()
    plt.grid()
    plt.show()
    # trajectory animation
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(optimized_positions[:, 0]) - 1, np.max(optimized_positions[:, 0]) + 1)
    ax.set_ylim(np.min(optimized_positions[:, 1]) - 1, np.max(optimized_positions[:, 1]) + 1)
    robot, = ax.plot([], [], 'bo', markersize=5)
    trajectory_line, = ax.plot([], [], 'b-', linewidth=1)

    def init():
        robot.set_data([], [])
        trajectory_line.set_data([], [])
        return robot, trajectory_line

    def animate(i):
        x = [optimized_positions[i, 0]]  # Wrap in a list
        y = [optimized_positions[i, 1]]  # Wrap in a list
        robot.set_data(x, y)
        trajectory_line.set_data(optimized_positions[:i+1, 0], optimized_positions[:i+1, 1])
        return robot, trajectory_line

    ani = animation.FuncAnimation(fig, animate, frames=factor_graph.num_states, init_func=init, blit=True)
    plt.title('Robot Trajectory Animation')
    plt.show()

    # Visualize the Factor Graph
    def visualize_factor_graph(T):
        G = nx.Graph()
        # Add nodes for states and controls
        for t in range(T + 1):
            G.add_node(f"q_{t}")
        for t in range(T):
            G.add_node(f"u_{t}")
        # Add edges for dynamics factors
        for t in range(T):
            G.add_edge(f"q_{t}", f"q_{t+1}", label="dynamics")
            G.add_edge(f"q_{t}", f"u_{t}", label="dynamics")
        # Add edges for control cost factors
        for t in range(T):
            G.add_edge(f"u_{t}", f"u_{t}", label="control cost")
        # Add edges for acceleration cost factors
        for t in range(T - 1):
            G.add_edge(f"u_{t}", f"u_{t+1}", label="acceleration cost")
        # Add start, goal, and intermediate constraints
        G.add_edge("start", "q_0", label="start")
        G.add_edge(f"q_{T}", "goal", label="goal")
        G.add_edge(f"q_{T//3}", "x0_in", label="intermediate_1")
        G.add_edge(f"q_{2*T//3}", "x1_in", label="intermediate_2")

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
        plt.title("Factor Graph Representation in SE(2)")
        plt.show()

    visualize_factor_graph(T=min(5, T))

if __name__ == "__main__":
    main()
