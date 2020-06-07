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

#Matrix Multiplication - neither operands nor output are the same size
# [n x m]  matrix multiplied by an [m x k] matrix results in an [n x k] matrix
# Inputs : 
# 1. An n x m matrix A
# 2. An m x k matrix B
#
# Output :
# return the n X k matrix equal to the matrix product A B.
@exercise
def matrix_mult(a : Matrix, b : Matrix) -> Matrix:
    rows = len(a) # the number of rows of the left matrix
    common = len(a[0]) # = len(b) - the common dimension of the matrices
    columns = len(b[0]) # the number of columns of the right matrix

    ans = create_empty_matrix(rows, columns)

    for currentRow in range(rows):
        for currentcolumn in range(columns):
            for k in range(common):
                ans[currentRow][currentColumn] += a [currentRow][k] * b[k][currentColumn]

            return ans
#############################################
# Matrix Inversion
# Input : an invertible 2 x 2 matrix A
# Output: Return the inverse of A, a 2 x 2 matrix (a ** -1)
#############################################
#############################################
# A square n x n  matrix "A" is invertible if it has an inverse n x n  matrix (A ** -1)
# With the following property : (A)(A ** -1) = (A ** -1)(A = (In)
# (A ** -1) acts as the multiplicative inverse of A
# Another equivalent definition highlights what makes this an interesting property.
# For any matrices B and C of compatible sizes:
# (A ** -1)(AB) = A(A ** -1(B)) = B
# (CA)(A ** -1) = ((C)(A ** -1))A = C
#############################################
# A square matrix has a property called the determinant, with the determinant of Matrix A
# Being written as "|A|" A matrix is invertible if and ohnly if its determinant isnt equal to 0
# For a 2 x 2 matrix "A", the determinant is defined as
# |A| = (A[0, 0])(A[1, 1]) - (A[0, 1] * A[1, 0])
# For larger matrices, the determinant is defined through determinants of sub-matrices
@exercise
def matrix_inverse(m : Matrix) -> Matrix:
    #Extract each element of the array into a named variable
    a = m[0][0]
    b = m[0][1]
    c = m[1][0]
    d = m[1][1]

    #Calculate the determinant
    determinant = (a * d) - (b * c)

    # Create the inverse of the matrix following the formula above
    ans = [[d / determinant, -b / determinant], [-c / determinant, a / determinant]]
    return ans
#############################################
# Transpose
# Input : An n x m matrix A
# Output : Return an m x n matrix (A ** T) the transpose of A
#############################################
@exercise
def transpose(a : Matrix) -> Matrix: 
    rows = len(a)
    columns = len(a[0])

    ans = create_empty_matrix(columns, rows)

    for i in range(rows):
        for j in range(columns):
            ans[j][i] = a[i][j]

    return ans
