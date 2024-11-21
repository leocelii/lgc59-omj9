import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import matplotlib.animation as animation

def wrap_angles(angle):
    """Wrap angles to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def main():

    parser = argparse.ArgumentParser(description='Trajectory Optimization for a 2-Link Robot Arm')
    parser.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0], help='Start joint angles (theta0 theta1)')
    parser.add_argument('--goal', nargs=2, type=float, default=[np.pi, np.pi/2], help='Goal joint angles (theta0 theta1)')
    parser.add_argument('--T', type=int, default=50, help='Number of time steps')
    args = parser.parse_args()

    
    dt = 0.1  # Time step
    T = args.T
    start = np.array(args.start)
    goal = np.array(args.goal)

    
    class FactorGraph:
        def __init__(self, T, start, goal, dt):
            self.T = T
            self.start = start
            self.goal = goal
            self.dt = dt
            self.num_states = T + 1
            self.num_controls = T
            self.factors = []

        def add_factor(self, factor):
            self.factors.append(factor)

        def compute_total_cost(self, variables):
            q = variables[:self.num_states * 2].reshape(self.num_states, 2)
            u = variables[self.num_states * 2:].reshape(self.num_controls, 2)
            total_cost = 0
            for factor in self.factors:
                total_cost += factor(q, u)
            return total_cost

    
    def dynamics_factor(dt, t):
        """Dynamics factor: Enforces theta_{t+1} = theta_t + u_t * dt."""
        def factor(q, u):
            theta_t = q[t]
            theta_t1 = q[t + 1]
            u_t = u[t]
            dynamics_residual = wrap_angles(theta_t1 - (theta_t + u_t * dt))
            return np.sum(dynamics_residual**2)
        return factor

    def start_factor(start):
        """Start factor: Enforces theta_0 = start."""
        def factor(q, u):
            residual = wrap_angles(q[0] - start)
            return np.sum(residual**2) * 1e6  
        return factor

    def goal_factor(goal):
        """Goal factor: Enforces theta_T = goal."""
        def factor(q, u):
            residual = wrap_angles(q[-1] - goal)
            return np.sum(residual**2) * 1e6  
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

    
    factor_graph = FactorGraph(T, start, goal, dt)

    
    factor_graph.add_factor(start_factor(start))  
    factor_graph.add_factor(goal_factor(goal))    
    for t in range(T):
        factor_graph.add_factor(control_cost_factor(t))  
        factor_graph.add_factor(dynamics_factor(dt, t))  
    for t in range(T - 1):
        factor_graph.add_factor(acceleration_cost_factor(dt, t))  

    # Initial guess: Straight line in angular space
    initial_positions = np.zeros((factor_graph.num_states, 2))
    for i in range(2):  # For each joint angle
        initial_positions[:, i] = np.linspace(start[i], goal[i], factor_graph.num_states)
    # Wrap initial angles
    initial_positions = wrap_angles(initial_positions)
    # Initial guess for controls
    initial_controls = np.diff(initial_positions, axis=0) / dt
    # Wrap controls
    initial_controls = wrap_angles(initial_controls)

    # Flatten 
    initial_variables = np.hstack((initial_positions.flatten(), initial_controls.flatten()))

    
    result = minimize(
        factor_graph.compute_total_cost,
        initial_variables,
        method="SLSQP",
        options={"disp": True, "maxiter": 1000}
    )

    
    optimized_positions = result.x[:factor_graph.num_states * 2].reshape(factor_graph.num_states, 2)
    optimized_controls = result.x[factor_graph.num_states * 2:].reshape(factor_graph.num_controls, 2)

    
    optimized_positions = wrap_angles(optimized_positions)
    optimized_controls = wrap_angles(optimized_controls)

    # Validation
    if result.success:
        print("Optimization succeeded.")
    else:
        print("Optimization failed:", result.message)

    print("Start matches:", np.allclose(optimized_positions[0], start, atol=1e-2))
    print("Goal matches:", np.allclose(optimized_positions[-1], goal, atol=1e-2))

    
    plt.figure()
    plt.plot(initial_positions[:, 0], initial_positions[:, 1], label="Initial Trajectory", linestyle="--")
    plt.plot(optimized_positions[:, 0], optimized_positions[:, 1], label="Optimized Trajectory", color="blue")
    plt.scatter([start[0], goal[0]], [start[1], goal[1]], c="red", label="Start/Goal")
    plt.title("Two-Link Robot Arm Trajectory Optimization")
    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.legend()
    plt.grid()
    plt.show()


#---Joint angles over time animation---
    # time_steps = np.arange(factor_graph.num_states) * dt

    # plt.figure()
    # plt.plot(time_steps, optimized_positions[:, 0], label='Theta 0')
    # plt.plot(time_steps, optimized_positions[:, 1], label='Theta 1')
    # plt.title('Joint Angles Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Joint Angle (rad)')
    # plt.legend()
    # plt.grid()
    # plt.show()
#---Angular Velocities over time animation---
    # time_steps_u = np.arange(factor_graph.num_controls) * dt

    # plt.figure()
    # plt.plot(time_steps_u, optimized_controls[:, 0], label='Angular Velocity 0')
    # plt.plot(time_steps_u, optimized_controls[:, 1], label='Angular Velocity 1')
    # plt.title('Control Inputs Over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angular Velocity (rad/s)')
    # plt.legend()
    # plt.grid()
    # plt.show()

#---Animation of the robot arm---
    def forward_kinematics(theta):
        """Compute the (x, y) position of the end effector."""
        l1 = 1.0  #first link
        l2 = 1.0  # second link
        theta0, theta1 = theta
        x1 = l1 * np.cos(theta0)
        y1 = l1 * np.sin(theta0)
        x2 = x1 + l2 * np.cos(theta0 + theta1)
        y2 = y1 + l2 * np.sin(theta0 + theta1)
        return (0, x1, x2), (0, y1, y2)

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x_vals, y_vals = forward_kinematics(optimized_positions[i])
        line.set_data(x_vals, y_vals)
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=factor_graph.num_states, init_func=init, blit=True)
    plt.title('2-Link Robot Arm Motion')
    plt.show()




    
    def visualize_factor_graph(T):
        G = nx.Graph()
        # Add nodes for states and controls
        for t in range(T + 1):
            G.add_node(f"theta_{t}")
        for t in range(T):
            G.add_node(f"u_{t}")
        # dynamics
        for t in range(T):
            G.add_edge(f"theta_{t}", f"theta_{t+1}", label="dynamics")
            G.add_edge(f"theta_{t}", f"u_{t}", label="dynamics")
       
        for t in range(T):
            G.add_edge(f"u_{t}", f"u_{t}", label="control cost")
       
        for t in range(T - 1):
            G.add_edge(f"u_{t}", f"u_{t+1}", label="acceleration cost")
        
        G.add_edge("start", "theta_0", label="start")
        G.add_edge(f"theta_{T}", "goal", label="goal")

       
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
        plt.title("Factor Graph Representation for Robot Arm")
        plt.show()

    
    visualize_factor_graph(T=min(5, T))

if __name__ == "__main__":
    main()
