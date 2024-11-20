from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gtsam
from typing import List, Optional
# "True" function with its respective parameters
def f(x, a=0.045, b=0.2, c=0.7, d = 4.86):
    return (a * x**3) + (b * x**2) + (c * x) + d
def error_func(y: np.ndarray, x: np.ndarray, this: gtsam.CustomFactor, v:
    gtsam.Values, H: List[np.ndarray]):
    """
    :param y: { Given data point at x: y = f(x) }
    :type y: { array of one element }
    :param x: { Value that produces y for some function f: y = f(x) }
    :type x: { Array of one element }
    :param this: The factor
    :type this: { CustomFactor }
    :param v: { Set of Values, accessed via a key }
    :type v: { Values }
    :param H: { List of Jacobians: dErr/dInput. The inputs of THIS
    factor (the values) }
    :type H: { List of matrices }
    """
    # First, get the keys associated to THIS factor. The keys are in the same order as when the factor is constructed
    key_a = this.keys()[0]
    key_b = this.keys()[1]
    key_c = this.keys()[2]
    key_d = this.keys()[3]
    # Access the values associated with each key. Useful function include: atDouble, atVector, atPose2, atPose3...
    a = v.atDouble(key_a)
    b = v.atDouble(key_b)
    c = v.atDouble(key_c)
    d = v.atDouble(key_d)
    # Compute the prediction (the function h(.))
    yp = (a * x**3) + (b * x**2) + (c*x) + d
    # Compute the error: H(.) - zi. Notice that zi here is "fixed" per factor
    error = yp - y
    # For comp. efficiency, only compute jacobians when requested
    if H is not None:
    # GTSAM always expects H[i] to be matrices. For this simple problem, each J is a 1x1 matrix
        H[0] = np.eye(1) * x**3 # derr / da
        H[1] = np.eye(1) * x**2 # derr / db
        H[2] = np.eye(1) * x # derr / dc
        H[3] = np.eye(1) # derr / dd
    return error

def main():
    parser = argparse.ArgumentParser(description='Fit a 3rd-order polynomial using a factor graph.')
    parser.add_argument('--initial', type=float, nargs=4, required=True, help='Initial guess for a, b, c, d')
    args = parser.parse_args()

    # Extract initial guesses
    a_init, b_init, c_init, d_init = args.initial

    # Generate synthetic data
    T = 21  # Number of data points
    x_values = np.linspace(-10, 10, T)  # x values in [-10, 10]
    sigma_values = [1, 5, 10]  # Noise levels
    GT = [f(x) for x in x_values]
    
    polynomial=[]
    for sigma in sigma_values:
        print(f"\nTesting with noise level σ = {sigma}:")

        # Add Gaussian noise
        Z = [gt + np.random.normal(0.0, sigma) for gt in GT]

        # Create factor graph and values
        graph = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()

        # Create keys for coefficients
        ka = gtsam.symbol('a', 0)
        kb = gtsam.symbol('b', 0)
        kc = gtsam.symbol('c', 0)
        kd = gtsam.symbol('d', 0)

        # Insert initial guesses
        v.insert(ka, a_init)
        v.insert(kb, b_init)
        v.insert(kc, c_init)
        v.insert(kd, d_init)

        # Define noise model
        noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

        # Add factors to the graph
        for i, x in enumerate(x_values):
            keys = gtsam.KeyVector([ka, kb, kc, kd])  # Keys for a, b, c, d
            gf = gtsam.CustomFactor(noise_model, keys, partial(error_func, np.array([Z[i]]), np.array([x])))
            graph.add(gf)

        result = gtsam.LevenbergMarquardtOptimizer(graph, v).optimize()

        # Retrieve and print the results
        a = result.atDouble(ka)
        b = result.atDouble(kb)
        c = result.atDouble(kc)
        d = result.atDouble(kd)
        
        x=np.linspace(-10, 10, 500)
        y=f(x)
        # Define the polynomial function
        def polynomial(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d
        poly=polynomial(x, a, b, c, d)
        plt.plot(x, y, label="Ground-Truth Function", color='blue')
        plt.plot(x, poly, label="Estimated Polynomial", color='green')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"Polynomial Plot - σ:{sigma}")
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # x-axis
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # y-axis
        plt.legend()
        plt.grid()
        print(f"Estimated coefficients: a={a:.6f}, b={b:.6f}, c={c:.6f}, d={d:.6f}")
        print(f"Ground truth coefficients: a=0.045, b=0.2, c=0.7, d=4.86")
        plt.show()


if __name__ == "__main__":
    main()
