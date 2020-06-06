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
    row = len(a)
    columns = len(a[0])
