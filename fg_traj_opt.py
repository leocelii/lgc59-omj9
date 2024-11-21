import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
import argparse

def main():
    
    parser = argparse.ArgumentParser(description='Trajectory Optimization via Factor Graph')
    parser.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0], help='Start position (x y)')
    parser.add_argument('--goal', nargs=2, type=float, default=[5.0, 5.0], help='Goal position (x y)')
    parser.add_argument('--T', type=int, default=50, help='Number of time steps')
    args = parser.parse_args()

   
    dt = 0.1  # Time step
    T = args.T    # Number of time steps
    start = np.array(args.start)  
    goal = np.array(args.goal)    

  
    class FactorGraph:
        def __init__(self, T, start, goal, dt):
            self.T = T
            self.start = start
            self.goal = goal
            self.dt = dt
            self.num_states = T + 1  # States from t=0 to t=T
            self.num_controls = T    # Controls from t=0 to t=T-1
            self.factors = []  # List of factors (cost functions)

        def add_factor(self, factor):
            """Add a factor (relationship/constraint) to the graph."""
            self.factors.append(factor)

        def compute_total_cost(self, variables):
            """Compute the total cost of the factor graph for given variables."""
    
            q = variables[:self.num_states * 2].reshape(self.num_states, 2)
            u = variables[self.num_states * 2:].reshape(self.num_controls, 2)
            total_cost = 0
            for factor in self.factors:
                total_cost += factor(q, u)
            return total_cost

  
    def dynamics_factor(dt, t):
        """Dynamics factor: Enforces q_{t+1} = q_t + u_t * dt."""
        def factor(q, u):
            q_t = q[t]
            q_t1 = q[t + 1]
            u_t = u[t]
            dynamics_residual = q_t1 - (q_t + u_t * dt)
            return np.linalg.norm(dynamics_residual)**2
        return factor

    def start_factor(start):
        """Start factor: Enforces q_0 = start."""
        def factor(q, u):
            return np.linalg.norm(q[0] - start)**2 * 1e6  # Large weight to enforce equality
        return factor

    def goal_factor(goal):
        """Goal factor: Enforces q_T = goal."""
        def factor(q, u):
            return np.linalg.norm(q[-1] - goal)**2 * 1e6  
        return factor

    def control_cost_factor(t):
        """Control cost: Penalizes large control inputs."""
        def factor(q, u):
            u_t = u[t]
            return np.linalg.norm(u_t)**2
        return factor

    def acceleration_cost_factor(dt, t):
        """Acceleration cost: Penalizes changes in control (accelerations)."""
        def factor(q, u):
            u_t = u[t]
            u_t1 = u[t + 1]
            acceleration = (u_t1 - u_t) / dt
            return np.linalg.norm(acceleration)**2
        return factor

  
    factor_graph = FactorGraph(T, start, goal, dt)

    
    factor_graph.add_factor(start_factor(start))  
    factor_graph.add_factor(goal_factor(goal))    
    for t in range(T):
        factor_graph.add_factor(control_cost_factor(t)) 
        factor_graph.add_factor(dynamics_factor(dt, t))  
    # Add acceleration cost factors for smoothness except at the last control input
    for t in range(T - 1):
        factor_graph.add_factor(acceleration_cost_factor(dt, t))

    # Initial Straight line from start to goal for positions
    initial_positions = np.linspace(start, goal, factor_graph.num_states)
    # Initial guess for controls Average velocity required to go from start to goal
    average_velocity = (goal - start) / (T * dt)
    initial_controls = np.tile(average_velocity, (factor_graph.num_controls, 1))

    # Flatten
    initial_variables = np.hstack((initial_positions.flatten(), initial_controls.flatten()))

    #bounds if needed
    bounds = [(None, None)] * len(initial_variables)

   
    result = minimize(
        factor_graph.compute_total_cost,
        initial_variables,
        method="SLSQP",
        bounds=bounds,
        options={"disp": True, "maxiter": 1000}
    )

   
    optimized_positions = result.x[:factor_graph.num_states * 2].reshape(factor_graph.num_states, 2)
    optimized_controls = result.x[factor_graph.num_states * 2:].reshape(factor_graph.num_controls, 2)

    
    plt.figure()
    plt.plot(initial_positions[:, 0], initial_positions[:, 1], label="Initial Trajectory")
    plt.plot(optimized_positions[:, 0], optimized_positions[:, 1], label="Optimized Trajectory")
    plt.scatter([start[0], goal[0]], [start[1], goal[1]], c="red", label="Start/Goal", zorder=5)
    plt.title("Trajectory Optimization via Factor Graph")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()
    plt.show()

    #  velocity profile
    velocities = optimized_controls
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    time_steps = np.arange(factor_graph.num_controls) * dt

    plt.figure()
    plt.plot(time_steps, velocity_magnitudes, label='Velocity Magnitude')
    plt.title('Velocity Profile Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Magnitude')
    plt.legend()
    plt.grid()
    plt.show()

    #acceleration profile
    accelerations = np.diff(optimized_controls, axis=0) / dt
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
    time_steps_acc = np.arange(factor_graph.num_controls - 1) * dt

    plt.figure()
    plt.plot(time_steps_acc, acceleration_magnitudes, label='Acceleration Magnitude')
    plt.title('Acceleration Profile Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration Magnitude')
    plt.legend()
    plt.grid()
    plt.show()

   
    def visualize_factor_graph(T):
        G = nx.Graph()
        #nodes for states and controls
        for t in range(T + 1):
            G.add_node(f"q_{t}")
        for t in range(T):
            G.add_node(f"u_{t}")
        
        for t in range(T):
            G.add_edge(f"q_{t}", f"q_{t+1}", label="dynamics")
            G.add_edge(f"q_{t}", f"u_{t}", label="dynamics")
        
        for t in range(T):
            G.add_edge(f"u_{t}", f"u_{t}", label="control cost")
       
        for t in range(T - 1):
            G.add_edge(f"u_{t}", f"u_{t+1}", label="acceleration cost")
        
        G.add_edge("start", "q_0", label="start")
        G.add_edge(f"q_{T}", "goal", label="goal")

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"))
        plt.title("Factor Graph Representation")
        plt.show()

    visualize_factor_graph(T=min(5, T))  

if __name__ == "__main__":
    main()
