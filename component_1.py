import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

##Validating Rotations
# compares given matrix against the identity matrix within a tolerance denoted by epsilon parameter
def isIdentity(matrix, epsilon):
    matrixSize = matrix.shape[0] # grabs the first dimension of the array
    identityMatrix = np.eye(matrixSize) # creates identity matrix
    return np.allclose(matrix, identityMatrix, atol=epsilon) # checks if matrix is equal to identity matrix within given tolerance, atol=epsilon

# creates the transpose of the given matrix
def findTranspose(matrix, size):
    transpose = [row[:] for row in matrix] # copies matrix using list comprehension
    transpose = np.array(transpose)
    for i in range(size):
        for j in range(size):
            transpose[i][j] = matrix[j][i] # switches element to mirror position, row for column and column for row.
    print(transpose)
    return transpose # returns transpose

# checks if given matrix is orthogonal
def isOrthogonal(matrix, epsilon):
    transpose = findTranspose(matrix, matrix.shape[0])
    productMatrix = np.dot(matrix, transpose)
    return isIdentity(productMatrix, epsilon)
    
 ## Steps for check_SOn implementation:
    ## Is matrix orthongal && is determinant of matrix = 1? (both true == m ∊ SO(n))
    ##    - is matrix orthongal? 
    ##        -  find transpose of matrix
    ##        -  multiply transpose with original matrix
    ##        -  if product matrix is identity matrix I, return true, otherwise false
    ##    - does determinant = 1?
    ##        - use np.linalg.det
    ##        - if det = 1, return true, otherwise false.
    
def check_SOn(matrix, epsilon=0.01) -> bool:
    # Finding transpose and multiplying with original and check if resultant matrix is identity
    matrix = np.array(matrix)
    transpose = findTranspose(matrix, matrix.shape[0])
    productMatrix = np.dot(matrix, transpose)
    if not (isIdentity(productMatrix, epsilon)):
        return False

    # Finding the determinant
    determinant = np.linalg.det(matrix)
        #are we allowed to use numpy.linalg.det?
    if not ((determinant >= 1.0 - epsilon) and (determinant <= 1.0 + epsilon)): # within epsilon precision tolerance
        return False # determinant is not within the boudaries of epsilon parameter, [0.99 - 1.01]
    return True

## Steps for check_quaternion implementation
    ## Is vector v ∊ S^3?
    ## check if the vector has 4 elements (check if array is length 4)
    ## compute the sum of squares for all values of the vector
    ## ex: 
    ## v (1,0,0,0)
    ## sum = 1^2 + 0^2 + 0^2 + 0^2
    ## if sum of the squares is 1 than the vector is in S^3
def check_quaternion(vector, epsilon=0.01) -> bool: 
    vectorArray = np.array(vector)
    if (len(vectorArray) == 4):
        sum1 = float((vectorArray[0] ** 2) + (vectorArray[1] ** 2) + (vectorArray[2] ** 2) + (vectorArray[3] ** 2))
        if not((sum1 >= 1.0 - epsilon) and (sum1 <= 1.0 + epsilon)): # within epsilon precision tolerance
            return False
    else:
        return False
    return True    

## Steps for check_SE(n) implementation
    ## is the matrix in SE(2) or SE(3)?
    ## if the matrix is 3x3, then check SE(2) 
    ##    -  first check matrix structure, bottom row of 
    ##          - must be a vector of the form (0, 0, 1)
    ##    -  second check the top left 2x2 matrix is orthogoal
    ##          - x1, x2 (example 2x2)
    ##          - x3, x4
    ##          - check that x1^2 + x2^2 = 1
    ##          - check that x3^2 + x4^2 = 1
    ##          - check that x1*x3 + x2*x4 = 0
    ## if both conditions met than return true, otherwise false
def check_SEn(matrix, epsilon=0.01) -> bool:  
    matrix = np.array(matrix)
    bottomRow = matrix[-1]
    orthogonalSE2 = np.array([0, 0, 1])
    orthogonalSE3 = np.array([0, 0, 0, 1])

    if (len(bottomRow) == 3):
        if not ((bottomRow == orthogonalSE2).all()): # checks if the bottom row is of form (0, 0, 1) SE(2)).
            return False
    elif (len(bottomRow) == 4): 
        if not ((bottomRow == orthogonalSE3).all()): #checks if the bottom row is of form (0, 0, 0, 1) SE(3)).
            return False
    else:
        return False

    orthogonal = matrix[:2, :2] if (matrix.shape[0] == 3) else matrix[:3, :3] # the variable orthogonal will be assigned to either the top left 2x2 matrix of SE(2) matrix, or top left 3x3 matrix of SE(3) matrix.
    if not (isOrthogonal(orthogonal, epsilon)):
        return False
    return True

def random_rotation_matrix(naive: bool) -> np.ndarray:
    if naive:
        # Step 1: Generate random Euler angles
        roll = np.random.uniform(0, 2 * np.pi)
        pitch = np.random.uniform(0, 2 * np.pi)
        yaw = np.random.uniform(0, 2 * np.pi)
        # Rotation matrix for roll (rotation around x-axis)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        #rotation matrix for pitch (y)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        #rotation matrix for yaw (z)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # then Combine the rotation matrices. The order of multiplication matters (R = R_z * R_y * R_x)
        R = R_z @ R_y @ R_x
        return R
    else:
        #Generate a random quaternion
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        #Convert the quaternion to a 3x3 rotation matrix
            #I pray this is correct lol
        R = np.array([
            [1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q4 * q1), 2 * (q2 * q4 + q3 * q1)],
            [2 * (q2 * q3 + q4 * q1), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q2 * q1)],
            [2 * (q2 * q4 - q3 * q1), 2 * (q3 * q4 + q2 * q1), 1 - 2 * (q2**2 + q3**2)]
        ])

        return R

#Rigid body in motion
def interpolate_rigid_body(start_pose, goal_pose) -> np.ndarray:
    
    initialX, initialY, initialTheta = start_pose
    finalX, finalY, finalTheta = goal_pose
    
    #create 5 linear interpolation steps from start to goal position
    pathX = np.linspace(initialX, initialY, 5)
    pathY = np.linspace(finalX, finalY, 5)
    pathTheta = np.linspace(initialTheta, finalTheta, 5)
    
    interpolationSteps = np.stack((pathX, pathY, pathTheta), axis=-1)
    
    return interpolationSteps

def forward_propagate_rigid_body(vector, plan) -> np.ndarray:
    
    finalPath = [np.array(vector)] # add initial velocity vector as first state of path
    for planVector, duration in plan:
        vX, vY, rotation = planVector
        currentX, currentY, currentRot = finalPath[-1] 
        
        deltaX = (vX * duration * np.cos(currentRot)) - (vY * duration * np.sin(currentRot))
        deltaY = (vX * duration * np.sin(currentRot)) + (vY * duration * np.cos(currentRot))
        deltaTheta = rotation * duration
        
        finalX = currentX + deltaX
        finalY = currentY + deltaY
        finalTheta = currentRot + deltaTheta
        finalTheta = finalTheta % (2*np.pi)
	
        if finalTheta < 0:
            finalTheta += 2*np.pi		 # make sure in between [0, 2pi]

        finalPath.append(np.array([finalX, finalY, finalTheta]))
    return np.array(finalPath)

def visualize_path(path):
    robotLength = 0.5
    robotWidth = 0.3
    # Robot corner points relative to the robot's center
    robotCorners = np.array([
        [-(robotLength / 2), -(robotWidth / 2)],
        [(robotLength / 2), -(robotWidth / 2)],
        [(robotLength / 2), (robotWidth / 2)],
        [-(robotLength / 2), (robotWidth / 2)],
    ])

    # Extract x, y positions for plotting the path
    xVals = [pose[0] for pose in path]
    yVals = [pose[1] for pose in path]

    # Setup 
    fig, ax = plt.subplots()
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_title('Robot Path Animation')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True)

    ax.plot(xVals, yVals, 'k--', label='Path')

    #robot polygon
    robotPatch = plt.Polygon(robotCorners, closed=True, fc='blue', ec='black')
    ax.add_patch(robotPatch)

    #updates the robot's position and orientation per frame
    def update(frame):
        vX, vY, theta = path[frame]
        # Compute rotation matrix
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        rotatedCorners = robotCorners @ rotation.T
        robotPosition = rotatedCorners + np.array([vX, vY])
        robotPatch.set_xy(robotPosition)

        return robotPatch,

    ani = FuncAnimation(fig, update, frames=len(path), blit=True, interval=500, repeat=True)
    plt.legend()
    plt.show()


def interpolate_arm(start, goal, steps=10):
    thetaStart0, thetaStart1 = start
    thetaGoal0, thetaGoal1 = goal
    N = steps
    path = []
    delta0 = thetaGoal0 - thetaStart0
    delta1 = thetaGoal1 - thetaStart1

    for k in range(N + 1):
        t = k / N
        theta0 = thetaStart0 + t * delta0
        theta1 = thetaStart1 + t * delta1
        path.append((theta0, theta1))

    return path

def forward_propagate_arm(start_pose, Plan):
    thetaCurrent0, thetaCurrent1 = start_pose  #initialize current joint angles
    path = [start_pose]  #initialize path with the starting configuration

    for command in Plan:
        w0, w1, dT = command  #extract angular velocities and duration

        #compute change in joint angles
        deltaTheta0 = w0 * dT
        deltaTheta1 = w1 * dT
        #update joint angles
        thetaNew0 = thetaCurrent0 + deltaTheta0
        thetaNew1 = thetaCurrent1 + deltaTheta1
        current_pose = (thetaNew0, thetaNew1)
        path.append(current_pose)

        thetaCurrent0 = thetaNew0
        thetaCurrent1 = thetaNew1

    return path


def visualize_arm_path(path):
    L1 = 2.0  
    L2 = 1.5  
    positions = []  #((x0, y0), (x1, y1), (x2, y2))
    # for each configuration in the path
    for t0, t1 in path:
        #Base
        x0, y0 = 0.0, 0.0
        # Position of Joint 1 (end of Link 1)
        x1 = x0 + L1 * np.cos(t0)
        y1 = y0 + L1 * np.sin(t0)
        #end of Link 2
        x2 = x1 + L2 * np.cos(t0 + t1)
        y2 = y1 + L2 * np.sin(t0 + t1)
        positions.append(((x0, y0), (x1, y1), (x2, y2)))

    # Set up the figure and axes
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title('Robotic Arm Movement')
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')

    # Initialize plot elements (lines for links)
    link1_line, = ax.plot([], [], 'o-', lw=4, color='blue', label='Link 1')
    link2_line, = ax.plot([], [], 'o-', lw=4, color='green', label='Link 2')

    #plotting the path of the end-effector
    end_effector_path, = ax.plot([], [], 'r--', lw=1, label='End-Effector Path')
    end_effector_positions = []
    ax.legend()

    # Function to update the plot for each frame
    def update_frame(i):
        ((x0, y0), (x1, y1), (x2, y2)) = positions[i]

        # Update Link 1
        link1_line.set_data([x0, x1], [y0, y1])

        #Link 2
        link2_line.set_data([x1, x2], [y1, y2])
        # Update end-effector path
        end_effector_positions.append((x2, y2))
        xs, ys = zip(*end_effector_positions)
        end_effector_path.set_data(xs, ys)

        return link1_line, link2_line, end_effector_path

    ani = FuncAnimation(fig, update_frame, frames=len(positions),
                        interval=50, blit=True, repeat=True)
    plt.show()



def main():
	
    path2 = [
    (1.0, 0.5, np.pi / 4),
    (3.8284, 3.3284, np.pi / 4),
    (3.8284, 3.3284, np.pi / 3),
    (1.7071, 7.0027, np.pi / 3),
    (1.9659, 7.9686, np.pi / 3)
    ]
    #works yippee!
    visualize_path(path2)
    # -----------------------------------------------
    #armpath test
    start_configuration = (np.radians(30), np.radians(45))  # (t0_start, t1_start)
    goal_configuration = (np.radians(90), np.radians(0))    # (t0_goal, t1_goal)
    arm_path = interpolate_arm(start_configuration, goal_configuration, steps=10)
    print(arm_path)
    # -----------------------------------------------
    start_configuration = (np.deg2rad(45), np.deg2rad(30))
    goal_configuration = (np.deg2rad(90), np.deg2rad(-45))
    path = interpolate_arm(start_configuration, goal_configuration, steps=50)
    visualize_arm_path(path)




if __name__=="__main__":
    main()
