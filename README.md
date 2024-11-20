CS 460/560: INTRO TO COMPUTATIONAL ROBOTICS FALL 2024
Assignment 3
Submission Process: You are asked to upload a zip file on Canvas containing all necessary files
of your submission. Only 1 student per team needs to upload a submission.
You should create a top-level folder named with your NETID, (e.g., xyz007-abc001 for teams of
2 students or just xyz007 for students working independently). Include at least the following files
by following this folder structure:
• xyz007-abc001/component_1.py
• (...)
• xyz007-abc001/component_n.py
• xyz007-abc001/component_i.{png|gif|mp4}
• xyz007-abc001/report.pdf
That is, for each  component of this assignment, submit the related source code and any specified
media. A report, in pdf format, detailing your implementation is also needed. You can optionally
create a utils.py file that contains helper functions under the xyz007-abc001 folder.
Please zip this xyz007-abc001 folder and submit the xyz007-abc001.zip on Canvas.
On the first page of the report’s PDF indicate the name and Net ID of the students
in the team (up to 2 students).
Extra Credit for LATEX: You will receive 6% extra credit points if you submit your answers as a
typeset PDF, i.e., one generated using LaTeX. For typeset reports in LaTeX, please also submit your
.tex source file together with the report, e.g., by adding a report.tex file under the xyz007-abc001
folder. There will be a 3% bonus for electronically prepared answers (e.g., using MS Word, etc.)
that are not typeset. Remember that these bonuses are computed as a percentage of your original
grade, i.e., if you were to receive 50 points and you have typeset your report using LaTeX, then
you get a 3-point bonus. If you want to submit a handwritten report, scan it or take photos of it
and submit the corresponding PDF as part of the zip file on Canvas. You will not receive any bonus
in this case. For your reports, do not submit Word documents, raw ASCII text files, or hardcopies
etc. If you choose to submit scanned handwritten answers and we cannot read them, you will not
be awarded any points for the unreadable part of the solution.
Failure to follow the above rules may result in a lower grade on the assignment.
Programming Languages:
You can use any programming language but we encourage you to use either Python or C++.
In general, you can use any library as long as it does not implement components of the
assignment. For example, in this first assignment, for vector operations you can use Numpy
(Python) or Eigen (C++) but cannot use rotation-specific operations.
Your assignment has to run on iLab and if there is any specific instruction to run your assign-
ment, add it to your report. If we cannot run your code on iLab, 0 credits will be awarded for the
programming tasks.
If you are using python, please use python3.10 in ilab using an environment. Numpy and
matplotlib libraries are already installed.
Setup python environment
user: ~ $ source /common/system/venv/python310/bin/activate
(python310) user: ~ $ python
import numpy
import matplotlib
For the factor graph component, use the conda environment (see the corresponding section).
It should be possible for the TAs to check whether your software returns a correct result for the
implemented functions. Be sure to use the function signature given.
Note that the following two assignments will build on top of the infrastructure of the first one.
So take some care with the code you develop. It should be clean, modular and reusable. For all of
your assignments, consider that the units are meters and radians.
Grading: Each component contributes a different percentage to the grade of your assignment.
CS460 The extra part is optional, a correct implementation would give you an extra 20%.
CS560 The extra part are necessary to get the maximum grade.
1 Potential Functions (35 pts)
A potential function is a real-valued function U = RN → R that guides the robot towards the goal. A
potential function consists of two components: an attractive potential U(q) that pulls the system
towards the goal and a repulsive potential Ur (q) that pushes the robot away from obstacles. If the
resulting function U(q) = U(q)+ Ur (q) has a minimum at the desired goal, optimization techniques
can be used. Specifically, the artificial force given by the vector ÏU(q) = [ δU
δq1 (q), · · · , δU
δqN (q)]T
defines the most promising direction of movement.
Using gradient descent (Algorithm 1), implement a potential function to move your rigid body
robot from a collision-free initial position to a desired goal while avoiding obstacles.
Algorithm 1 Gradient descent
q(0) ← qstrt
 ← 0
while ÏU(q()) ̸ = 0 do
q( + 1) ← q() + ÏU(q())
 =  + 1
end while
Create the following python script:
python potential.py --start 0.0 0.0 --goal 5.0 5.0
Where the input arguments are the start and goal configurations of the robot, assume that your
rigid body can only translate (no rotations). You can assume that the start configuration is collision-
free, but not the goal. Your program must produce an animation of the robot moving from the initial
configuration to the goal without collisions, be sure to include the resulting path.
Test your implementation with the 5 environments you generated for the previous assignment
with 5 different start/goals per environment. Report the solutions in terms of success (goal reached
and no collision) and cost (duration) of the resulting trajectory.
Note: the behavior of your potential function depends on the selected parameters (i.e. the re-
pulsive potential needs a distance to obstacles). You need to experiment and find values that work
well enough for some of your environments and use the same values through your experiments.
It is expected that in environments with the higher number of obstacles, some experiments will
fail. Include in your report the examples of the failures.
2 Gtsam: Factor Graphs (10 pts)
Gtsam is the factor graph library that you will use for trajectory optimization. While the library is
written in C++, it provides Python and Matlab bindings. If you do not have conda on ilab, install
miniforge first. In ilab, you can install the library via
conda create -n CS460-fall24
conda activate CS460-fall24
conda install conda-forge::gtsam
(While anaconda.org is blocked, installing new package via conda-forge is allowed).
The following code is provided as a starting example. While this problem is relatively simple to
solve with traditional tools, we can model it as a nonlinear factor graph to solve it. Run the code
and make sure you understand it. Validate that the given solution is correct.
Simple factor graph example: Fitting data to a line
from functools import partial
import numpy as np
import gtsam
from typing import List, Optional
# "True" function with its respective parameters
def f(x, m=0.6, b = 1.5):
return m * x + b
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
# First, get the keys associated to THIS factor. The keys are in the same
order as when the factor is constructed
key_m = this.keys()[0]
key_b = this.keys()[1]
# Access the values associated with each key. Useful function include:
atDouble, atVector, atPose2, atPose3...
m = v.atDouble(key_m)
b = v.atDouble(key_b)
# Compute the prediction (the function h(.))
yp = m * x + b
# Compute the error: H(.) - zi. Notice that zi here is "fixed" per factor
error = yp - y
# For comp. efficiency, only compute jacobians when requested
if H is not None:
# GTSAM always expects H[i] to be matrices. For this simple problem,
each J is a 1x1 matrix
H[0] = np.eye(1) * x # derr / dm
H[1] = np.eye(1) # derr / db
return error
if __name__ == ’__main__’:
graph = gtsam.NonlinearFactorGraph()
v = gtsam.Values()
T = 100
GT = [] # The ground truth, for comparison
Z = [] # GT + Normal(0, Sigma)
# The initial guess values
m = 1;
b = -1;
# Create the key associated to m
km = gtsam.symbol(’m’, 0)
kb = gtsam.symbol(’b’, 0)
# Insert the initial guess of each key
v.insert(km, m)
v.insert(kb, b)
# Create the \Sigma (a n x n matrix, here n=1)
sigma = 1
noise_model = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
for i in range(T):
GT.append(f(i))
Z.append(f(i) + np.random.normal(0.0, sigma)) # Produce the noisy data
# This are the keys associate to each factor.
# Notice that for this simple example, the keys do not depend on T, but
this may not always be the case
keys = gtsam.KeyVector([km, kb])
# Create the factor:
# Noise model - The Sigma associated to the factor
# Keys - The keys associated to the neighboring Variables of the
factor
# Error function - The function that computes the error: h(.) - z
# The function expected by CustomFactor has the
signature
# F(this: gtsam.CustomFactor, v: gtsam.Values,
H: List[np.ndarray])
# Because our function has more parameters (z
and i), we need to *fix* this
# which can be done via partial.
gf = gtsam.CustomFactor(noise_model, keys,
partial(error_func, np.array([Z[i]]),
np.array([i]) ))
# add the factor to the graph.
graph.add(gf)
# Construct the optimizer and call with default parameters
result = gtsam.LevenbergMarquardtOptimizer(graph, v).optimize()
# We can print the graph, the values and evaluate the graph given some
values:
# result.print()
# graph.print()
# graph.printErrors(result)
# Query the resulting values for m and b
m = result.atDouble(km)
b = result.atDouble(kb)
print("m: ", m, " b: ", b)
# Print the data for plotting.
# Should be further tested that the resulting m, b actually fit the data
for i in range(T):
print(i, GT[i], Z[i])
Fit a 3rd order polynomial
Implement the following program:
python fg_polynomial.py --initial 1 2 3 4
Which adapts the previous code to fit a third-order polynomial: ƒ () = 3 + b2 + c + d.
Assume that the ground-truth function is: ƒ () = 0.045 ∗ 3 + 0.2 ∗ 2 + 0.7 + 4.86. The "initial"
input parameter defines the initial guess for the values , b, c, d. Generate new data given the
polynomial with  ∈ [−10, 10] and adapt the function associated with the factor.
Test your implementation with the following noise levels: σ = {1, 5, 10} (this is: z = ƒ () +
N(0, σ). Your solution should always return values within some small ε to the real values. In your
report, include plots showing the ground truth, the noisy values Z, and the estimated polynomial.
3 Trajectory Optimization via Factor Graphs
3.1 Simple Trajectory (15 pts)
Recall that the trajectory optimization problem can be formulated as:
Θ∗ = rg min
Θ Cost(Θ)
s.t.  ̇t = ƒ (t , t )∀t ∈ [0, T]
0 = start state
T ∈ Xgo
(Extra constraints)
Consider a first-order point system qt+1 = qt +  ̇qt ∗ Δt where q ∈ R2. Implement a factor graph
that solves the trajectory optimization for the point system given a start state 0, a goal region
Xgo, and the total number of states T. Assume Δt = 0.1.
python fg_traj_opt.py --start 0 0 --goal 5 5 --T 50
In your report, include a diagram with the implemented factor graph (with only a few states,
not all T states), example visualizations of the generated trajectories and specify how you define
the initial guess.
3.2 Extra Constraints (20 pts)
Adapt the previous trajectory optimization to include two extra constraints:
 T
3
=0
n
 2T
3
=1
n
where 
n are input states. This is, instead of the expected straight line from the previous
problem, now the output trajectory should visit (or be close to) the two given states.
python fg_traj_opt_2.py --start 0 0 --goal 5 5 --T 50 --x0 -1 2 --x1 3 7
In your report, include a diagram of the modified factor graph and specify the function imple-
mented in the new factors. Also, include example visualizations of the trajectories generated by
your program.
3.3 2-Link Robot Arm (20 pts)
Implement the trajectory optimization factor graph for the 2-link arm (without extra constraints).
Notice that the configuration is now q = (θ0, θ1); your factors must consider that these are angles.
You can make use of gtsam’s class gtsam.Rot2.
python fg_traj_opt_arm.py --start 0 0 --goal 3.14 1.57 --T 50
In your report, include a diagram of the modified factor graph and specify the function imple-
mented in the new factors. Also, include example visualizations of the trajectories generated by
your program.
4 Trajectory Optimization in SE(2) (Extra 20 pts)
Implement the previous trajectory optimization factor graph for a system qt+1 = qt +  ̇qt ∗ Δt where
q ∈ SE(2). For this purpose, you can use the SE(2) specific functionality.
python fg_traj_opt_se2.py --start 0 0 0 --goal 5 5 1.57 --T 50 --x0 -1 2
0.78 --x1 3 7 3.14
In your report, include a diagram of the modified factor graph and specify the function imple-
mented in the new factors. Also, include example visualizations of the trajectories generated by
your program.
