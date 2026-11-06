import numpy as np
from petsc4py import PETSc

# Initialize PETSc and MPI
comm = PETSc.COMM_WORLD

# Create a PETSc sparse matrix for A
n = 100  # Size of the matrix
A = PETSc.Mat().create(comm=comm)
A.setSizes([n, n])
A.setType('aij')  # Sparse format
A.setUp()

# Fill the tridiagonal matrix
for i in range(n):
    if i > 0:
        A[i, i-1] = 1
    A[i, i] = -2
    if i < n-1:
        A[i, i+1] = 1
A.assemble()

# Create PETSc vectors for b and x
b = PETSc.Vec().create(comm=comm)
b.setSizes(n)
b.setUp()
b.set(1.0)  # Set all elements of b to 1

x = PETSc.Vec().create(comm=comm)
x.setSizes(n)
x.setUp()

# Create a KSP (Krylov Subspace Solver) to solve the system
ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setType('gmres')  # You can choose other solvers like 'cg', 'bcgs', etc.
ksp.setFromOptions()

# Solve the system
ksp.solve(b, x)

# Get the solution as a NumPy array
solution = x.getArray()

print("Solution using PETSc: ", solution)
