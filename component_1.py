import numpy as np
import math

##Validating Rotations
# compares given matrix against the identity matrix within a tolerance denoted by epsilon parameter
def isIdentity(matrix, epsilon):
    matrixSize = matrix.shape[0] # grabs the first dimension of the array
    identityMatrix = np.eye(matrixSize) # creates identity matrix
    return np.allclose(matrix, identityMatrix, atol=epsilon) # checks if matrix is equal to identity matrix within given tolerance, atol=epsilon
# creates the transpose of the given matrix
def findTranspose(matrix, size):
    transpose = [row[:] for row in matrix] # copies matrix using list comprehension
    for i in range(size):
        for j in range(size):
            transpose[i][j] == matrix[j][i] # switches element to mirror position, row for column and column for row.
    return transpose # returns transpose

 ## Steps for check_SOn implementation:
    ## Is matrix orthongal && is determinant of matrix = 1? (both true == m ∊ SO(n))
    ##    - is matrix orthongal? 
    ##        -  find transpose of matrix
    ##        -  multiply transpose with original matrix
    ##        -  if product matrix is identity matrix I, return true, otherwise false
    ##    - does determinant = 1?
    ##        - use np.linalg.det
    ##        - if det = 1, return true, otherwise false.
def check_SOn(matrix: m, float: epsilon=0.01) -> bool:
    # Finding transpose and multiplying with original and check if resultant matrix is identity
    matrix = np.array(matrix)
    transpose = matrix.T # we may have to code this because they said no rotation specific functions?
    transpose = findTranspose(matrix, matrix.shape[0]) 
    productMatrix = np.dot(matrix, transpose)
    if not (isIdentity(productMatrix, epsilon)):
        return False

    # Finding the determinant
    determinant = np.linalg.det(matrix)
    if not ((determinant >= 1 - epsilon) and (determinant <= 1 + epsilon)): # within epsilon precision tolerance
        return False # determinant is not within the boudaries of epsilon parameter, [0.99 - 1.01]

    print("Matrix is an element of SO(n).")
    return True

## Steps for check_quaternion implementation
    ## Is vector v ∊ S^3?
    ## check if the vector has 4 elements (check if array is length 4)
    ## compute the sum of squares for all values of the vector
    ## ex: 
    ## v (1,0,0,0)
    ## sum = 1^2 + 0^2 + 0^2 + 0^2
    ## if sum of the squares is 1 than the vector is in S^3
def check_quaternion(vector: v, float: epsilon=0.01) -> bool: 
    vectorArray = np.array(vector)
    if (len(vectorArray) == 4):
        sum1 = (vectorArray[0] ** 2) + (vectorArray[1] ** 2) + (vectorArray[2] ** 2) + (vectorArray[3] ** 2)
        if not((sum >= 1 - epsilon) and (sum <= 1 + epsilon)): # within epsilon precision tolerance
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
def check_SEn(matrix: m, float: epsilon=0.01) -> bool:  
  
def main():
  # don't think we need main?
if __name__=="__main__":
    main()
