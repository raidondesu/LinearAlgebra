from testing import exercise, create_empty_matrix
from typing import List

import math, cmath
Matrix = List[List[complex]]

# Matrix Addition
# Commutativity : A + B = B + A 
# Associativity : (A + B) + C = A + (B + C)
#############################################
# Input :
# 1. An n X m  matrix A  represented as a two-dimensional list
# 2. An n x m  matrix B, represented as a two-dimensional list
#############################################
# Output: Return the sum of the matrices A + B an n X m matrix, represented as a two-dimensional list
##############################################
# When representing matrices as lists, each sub-list represents a row.
# For example, list [[1, 2], [3, 4]] represents the following matrix:
#1 2
#3 4
#############################################
@exercise
def matrix_add(a : Matrix, b : Matrix) -> Matrix:
    # Size of matrix
    rows = len(a)
    columns = len(a[0])

    #Init matrix
    c = create_empty_matrix(rows, columns)

    for i in range(rows): 
        for j in range(columns):
            #Access elements
            x = a[i][j]
            y = b[i][j]
            
            #Modify elements
            c[i][j] = a[i][j] + b[i][j]


    return c
#############################################
# Scalar Multiplication - Multiplying the entire matrix by a scalar (real or complex #)
# Associativity: x(yA) = (xy)A
# Distributivity over matrix addition : x(A + B) = xA + xB
# Distributivity over scalar addition : (x + y)A = xA + yA
#############################################
# input :
# 1. A scalar x
# 2. An n x m matrix A
#############################################
# Output:
# Return n x m  matrix (xA)
#############################################
@exercise
def scalar_mult(x : complex, a : Matrix) -> Matrix:
    rows = len(a)
    columns = len(a[0])

    c - create_empty_matrix(rows, columns)

    for i in range(rows):
        for j in range(columns):
           c[i][j] = a[i][j] * x
    
    return c
