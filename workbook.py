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
#############################################
# Matrix Conjugate - Take the conjugate of every element of the matrix
# Input :
# An  n x m matrix A
# Output :
# Return an n x m  matrix A, the conjugate of A
@exercise
def conjugate(a : Matrix) -> Matrix:
    rows = len(a)
    columns = len(a[0])

    ans = create_empty_matrix(rows, columns)

    for i in range(rows):
        for j in range(columns):
            ans[i][j] = complex(a[i][j].real, -a[i][j].imag)
    
    return ans
#Adjoint - performing both transpose and conjugate on a matrixkl
#############################################
@exercise
def adjoint(a : Matrix) -> Matrix:
    #Transpose function with the input matrix a
    transp = transpose(a)

    ans = conjugate(transp)

    return ans
#############################################
# Unitary Verification - A matrix is unitary when it is invertible
# Its inverse is equal to its adjoint : (U ** -1) = (U ** t)
# Input :
# An n x n  matrix A
# Output :
# Check if the matrix is unitary and return True if it is, or False if it isnt
#############################################
# 1. Calculate the adjoint of the input matrix
# 2. Multiply it by the input matrix
# 3. Check if the multiplication result is equal to an identity matrix
#############################################
from pytest import approx 
@exercise
def is_matrix_unitary(a : Matrix) -> bool:
    n = len(a)

    #Calculate the adjoint matrix
    adjointA = adjoint(a)

    #Multiply the adjoint matrix by the input matrix
    multipliedMatrix = matrix_mult(a, adjointA)

    #Check whether the multiplication result is (approximiatley) identity matrix
    for i in range(n):
        for j in range(n):
            #An identity matrix has 1's in all the places where the row index and column index are equal...
            if i == j:
                if multipliedMatrix[i][j] != approx(1):
                    return False
            # and zeros in all the places where the row index and column index are different
            else:
                if multipliedMatrix[i][j] != approx(0):
                    return False
            
            return True
#############################################
# Inner product - A matrix multiplied by anothers adjoint
# Vector norm = ||V|| = sqrt(<V,V>)
# Vector normalization - Occurs when vector norm is equal to one
#############################################
@exercise
def inner_prod(v : Matrix, w : Matrix) -> complex:
    # Calculate the adjoint of the v vector
    adjointV = adjoint(v)

    # Multiply the adjoint v and w. The result will be a matrix with only one element
    resultMatrix = matrix_mult(adjointV, w)

    # To get the actual complex number, we have to take one element from the multiplication result
    return resultMatrix[0][0]
#############################################
#############################################
@exercise
def normalize(v : Matrix) -> Matrix:
    norm = math.sqrt(inner_prod(v, v).real)

    n = len(v)
    ans = create_empty_matrix(n, 1)

    # Divide each element of the vector by the norm
    for i in range(n):
        ans[i][0] = v[i][0] / norm

    return ans
#############################################
#############################################
# Outer product
# Inputs:
# An n x 1 vector V
# An m x 1 vector W
# Output :
# Return an n x m  matrix that represents the outer product of V and W
@exercise
def outer_prod(v : Matrix, w : Matrix) -> Matrix:
    # Calculate adjoint of the W
    adjointW = adjoint(w)

    # Multiply V by W adjoint
#############################################
# Tensor Product - Multiplies the second matrix by every element of the first matrix
# Input :
# An n x m  matrix A
# An k x l  matrix B
# Output :
# Return (n * k) * (m * l) matrix A tensor B [ tensor product of a and b]
#############################################
#############################################
@exercise
def tensor_product(a : Matrix, b : Matrix) -> Matrix:
    aRows = len(a) # the number of rows for matrix a
    aColumns = len(a[0]) # the number of columns for matrix a
    bRows = len(b) # the number of rows for matrix b

    ans = create_empty_matrix(aRows * bRows, aColumns * bColumns)

    # Outer pair of loops, iterating tough the elements of the left matrix
    for i in range(aRows):
        for j in range(aColumns):
            # inner pair of loops, iterating through the elements of the right matrix
            for k in range(bRows)
                for l in range(bColumns)
                    ans[i * bRows + k][j * bColumns + 1] = a[i][j] * b[k][l]
    return ans
#############################################
#############################################
# Given a nonzero (n x n) matrix "A"
# A nonzero vector V
# And a scalar x
# if AV = xV then "x" is an eigenvalue of A
# And V is an eigenvector of A corrensponding to that eigenvalue
# Inputs L
# 1. A real valued n x n matrix A
# 2. An eigenvector V of matrix A
# Output : Return a real number - the eigenvalue of A that is associated with the given eigenvector
from pytest import approx

@exercise
def find_eigenvalue(a : Matrix, v : Matrix) -> float:
    n = len(v)
    multiplied = matrix_mult(a, v)

    for i in range(n):
        if (v[i][0] != aprrox(0)):
            return multiplied[i][0] / v[i][0]