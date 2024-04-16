# 1: Jacobi Iterative Method (Algorithm 7.1, Page 453)
from fractions import Fraction
import numpy as np

def jacobi_iteration(A, b, x0, tol=Fraction(1, 1000), max_iter=100):
    n = len(b)
    x_prev = x0.copy()

    # Print initial approximation
    print("Initial approximation:")
    print(f"x^(0) = {[float(x_i) for x_i in x_prev]}")

    for iter_count in range(1, max_iter + 1):
        x = np.zeros_like(x0, dtype=object)  # Use object dtype for fractions

        # Compute Jacobi iteration for each component
        for i in range(n):
            sum_term = np.dot(A[i, :], x_prev) - A[i, i] * x_prev[i]
            x[i] = Fraction(b[i] - sum_term, A[i, i])

        # Convert iteration result to floating-point numbers before printing
        x_float = [float(x_i) for x_i in x]

        # Print iteration results with non-fractional format
        print(f"Iteration {iter_count}:")
        print(f"x^(k) = {x_float}")

        # Check for convergence
        if np.max(np.abs(x - x_prev)) < tol:
            print(f"\nConverged in {iter_count} iterations.")
            return x_float

        x_prev = x.copy()

    print("\nDid not converge within the maximum number of iterations.")
    return x_float

# Define the coefficient matrix A and the constant vector b using Fraction objects
A = np.array([
    [10, -1, 2, 0],
    [-1, 11, -1, 3],
    [2, -1, 10, -1],
    [0, 3, -1, 8]
], dtype=object)  # Use object dtype for fractions

b = np.array([6, 25, -11, 15], dtype=object)  # Use object dtype for fractions

# Initial approximation (starting point) using fractions
x0 = np.array([Fraction(0), Fraction(0), Fraction(0), Fraction(0)], dtype=object)  # Use object dtype for fractions

# Set tolerance and maximum number of iterations
tolerance = Fraction(1, 1000)  # Tolerance as Fraction object
max_iterations = 10  # Adjust the maximum iterations as needed

# Perform Jacobi iteration
solution = jacobi_iteration(A, b, x0, tol=tolerance, max_iter=max_iterations)

# Print the final solution (converted to floating-point format)
print("\nApproximate solution using Jacobi's method (floating-point):")
print(solution)
print("\n")



# 2. Gauss-Seidel Iterative Method (Algorithm 7.2, Page 456)
from fractions import Fraction
import numpy as np

def gauss_seidel_iteration(A, b, x0, tol=Fraction(1, 1000), max_iter=100):
    n = len(b)
    x_prev = x0.copy()

    # Print initial approximation
    print("Initial approximation:")
    print(f"x^(0) = [0. 0. 0. 0.]")

    for iter_count in range(1, max_iter + 1):
        x = x_prev.copy()

        # Update each component using the Gauss-Seidel formula with fractions
        for i in range(n):
            sum_term = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = Fraction(b[i] - sum_term, A[i, i])

        # Print iteration results with non-fractional format
        x_float = [float(x_i) for x_i in x]  # Convert to float before printing
        print(f"Iteration {iter_count}:")
        print(f"x^(k) = {x_float}")

        # Check for convergence
        if max(abs(x[j] - x_prev[j]) for j in range(n)) < tol:
            print(f"\nConverged in {iter_count} iterations.")
            return x_float

        x_prev = x.copy()

    print("\nDid not converge within the maximum number of iterations.")
    return x_float

# Define the coefficient matrix A and the constant vector b using Fraction objects
A = np.array([
    [10, -1, 2, 0],
    [-1, 11, -1, 3],
    [2, -1, 10, -1],
    [0, 3, -1, 8]
], dtype=object)  # Use object dtype for Fraction

b = np.array([6, 25, -11, 15], dtype=object)  # Use object dtype for Fraction

# Initial approximation (starting point) using fractions
x0 = np.array([Fraction(0), Fraction(0), Fraction(0), Fraction(0)], dtype=object)  # Use object dtype for Fraction

# Set tolerance and maximum number of iterations
tolerance = Fraction(1, 1000)  # Tolerance as Fraction object
max_iterations = 100

# Perform Gauss-Seidel iteration with fractions
solution = gauss_seidel_iteration(A, b, x0, tol=tolerance, max_iter=max_iterations)

# Print the final solution (converted to floating-point format)
print("\nApproximate solution using Gauss-Seidel method (floating-point):")
print(solution)
print("\n")



# 3. SOR Method (Algorithm 7.3, Page 467)
import numpy as np

def sor_method(A, b, x0, omega, tol=1e-6, max_iter=1000):
    n = len(b)
    x_prev = x0.copy()

    # Print initial approximation
    print("Initial approximation:")
    print(f"x^(0) = {x_prev}")

    for iter_count in range(1, max_iter + 1):
        x = x_prev.copy()

        for i in range(n):
            sum_term = np.dot(A[i, :], x) - A[i, i] * x[i]
            x[i] = (1 - omega) * x_prev[i] + (omega / A[i, i]) * (b[i] - sum_term)

        # Print iteration results
        print(f"Iteration {iter_count}:")
        print(f"x^(k) = {x}")

        # Check for convergence
        if np.linalg.norm(x - x_prev, np.inf) < tol:
            print(f"\nConverged in {iter_count} iterations.")
            return x

        x_prev = x.copy()

    print("\nDid not converge within the maximum number of iterations.")
    return x

# Define the coefficient matrix A and the constant vector b
A = np.array([
    [4, 3, 0],
    [3, 4, -1],
    [0, -1, 4]
], dtype=float)

b = np.array([24, 30, -24], dtype=float)

# Initial approximation (starting point)
x0 = np.array([1, 1, 1], dtype=float)

# Relaxation parameter (omega)
omega = 1.25

# Set tolerance and maximum number of iterations
tolerance = 1e-6
max_iterations = 1000

# Perform SOR method
solution = sor_method(A, b, x0, omega, tol=tolerance, max_iter=max_iterations)

print("\nApproximate solution using SOR method:")
print(solution)
print("\n")



# 4. Iterative Refinement Method (Algorithm 7.4, Page 474)
import numpy as np

def gaussian_elimination(A, b):
    # Perform Gaussian elimination to solve Ax = b
    n = len(b)
    augmented_matrix = np.concatenate((A, b.reshape(n, 1)), axis=1)

    # Forward elimination
    for i in range(n):
        # Pivot selection (partial pivoting)
        pivot_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

    # Back substitution to get the solution x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:n], x[i+1:])) / augmented_matrix[i, i]

    return x

def iterative_refinement(A, b, max_iterations=100, tolerance=1e-5):
    # Initial approximation
    x_prev = gaussian_elimination(A, b)

    for iteration in range(1, max_iterations + 1):
        # Calculate residual r
        r = b - np.dot(A, x_prev)

        # Solve Ay = r
        y = np.linalg.solve(A, r)

        # Update x
        x_next = x_prev + y

        # Check error
        error = np.max(np.abs(x_next - x_prev))

        # Update for next iteration
        x_prev = x_next

        # Check stopping criterion
        if error < tolerance:
            print(f"Iterative refinement converged after {iteration} iterations.")
            break

    return x_next

# Example usage
A = np.array([[3.3330, 15920, -10.333],
              [2.2220, 16.710, 9.6120],
              [1.5611, 5.1791, 1.6852]])

b = np.array([15913, 28.544, 8.4254])

# Perform iterative refinement
approx_solution = iterative_refinement(A, b)

print("Approximate solution:", approx_solution)
