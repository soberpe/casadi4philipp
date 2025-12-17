"""
CasADi Example for Flat Output Systems and Jacobian Computations

This demonstrates how to efficiently compute:
1. Jacobians of vector functions w.r.t. vectors
2. Recursive derivatives (Jacobian-vector products, higher-order derivatives)
3. Newton solver with automatic differentiation

Key advantage: CasADi uses ALGORITHMIC DIFFERENTIATION, not symbolic.
This means no expression explosion - even with millions of operations!
"""

import casadi as ca
import numpy as np


# ============================================================================
# EXAMPLE 1: Basic Jacobian Computation
# ============================================================================
print("="*70)
print("EXAMPLE 1: Basic Jacobian Computation")
print("="*70)

# Define symbolic variables (can be large vectors)
n = 10  # Size of input vector
m = 8   # Size of output vector

x = ca.SX.sym('x', n)

# Define a vector function f: R^n -> R^m
# Example: nonlinear transformation
f = ca.vertcat(
    ca.sin(x[0]) * x[1]**2 + ca.exp(x[2]),
    x[0]**3 - x[1]*x[2] + x[3],
    ca.norm_2(x[0:4]),
    x[2]**2 + x[3]**2 - x[4],
    ca.tanh(x[5] + x[6]),
    x[0] * x[7] + x[8] * x[9],
    ca.sum1(x),
    ca.sumsqr(x)
)

# Compute Jacobian df/dx (m x n matrix)
J = ca.jacobian(f, x)

# Create a function that evaluates both f and J
f_func = ca.Function('f', [x], [f, J], ['x'], ['f', 'J'])

# Evaluate at a point
x_val = np.random.randn(n)
f_val, J_val = f_func(x_val)

print(f"Input dimension: {n}")
print(f"Output dimension: {m}")
print(f"Jacobian shape: {J_val.shape}")
print(f"Jacobian sparsity: {np.count_nonzero(J_val)}/{J_val.size} nonzero elements")
print(f"\nFirst 3 rows of Jacobian:\n{J_val[:3, :]}")


# ============================================================================
# EXAMPLE 2: Recursive Jacobian-Vector Products
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Recursive Differentiation - Directional Derivatives")
print("="*70)

# Given a function f(x) and direction v, compute:
# 1. First order: J * v  (directional derivative)
# 2. Second order: H * v  where H is Hessian
# 3. Continue recursively...

x = ca.SX.sym('x', 5)
v = ca.SX.sym('v', 5)  # Direction vector

# Define function
f = ca.vertcat(
    x[0]**2 * ca.sin(x[1]) + ca.exp(x[2]*x[3]),
    x[0] * x[1] * x[2] - x[3]**3 + x[4],
    ca.norm_2(x)**2 - ca.sum1(x)
)

# First order: Jacobian times vector (directional derivative)
J = ca.jacobian(f, x)
Jv = ca.mtimes(J, v)

# Second order: Derivative of (J*v) w.r.t. x, times v again
# This gives v^T * H * v for each component (Hessian quadratic form)
Jv_x = ca.jacobian(Jv, x)
Jvv = ca.mtimes(Jv_x, v)

# Third order: Continue the recursion
Jvv_x = ca.jacobian(Jvv, x)
Jvvv = ca.mtimes(Jvv_x, v)

# Create function for all derivatives
derivatives_func = ca.Function('derivatives', 
                              [x, v], 
                              [f, Jv, Jvv, Jvvv],
                              ['x', 'v'],
                              ['f', 'Jv', 'Jvv', 'Jvvv'])

# Evaluate
x_val = np.array([1.0, 2.0, 0.5, -1.0, 0.3])
v_val = np.array([0.1, -0.2, 0.3, 0.1, -0.1])

f_val, Jv_val, Jvv_val, Jvvv_val = derivatives_func(x_val, v_val)

print(f"Function value f(x): {f_val.T}")
print(f"1st order (J*v): {Jv_val.T}")
print(f"2nd order (v'*H*v): {Jvv_val.T}")
print(f"3rd order: {Jvvv_val.T}")


# ============================================================================
# EXAMPLE 3: Forward/Reverse Mode for Efficient Computation
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: Efficient Directional Derivatives (Forward/Reverse Mode)")
print("="*70)

# For Jacobian-vector products, use forward mode (more efficient than computing full Jacobian)
n = 100  # Large input dimension
m = 50   # Large output dimension

x = ca.SX.sym('x', n)
v = ca.SX.sym('v', n)

# Complex function
f = ca.SX.zeros(m)
for i in range(m):
    # Create nonlinear coupling
    f[i] = ca.sum1(ca.sin(x[i:i+min(10, n-i)]))
    if i < n:
        f[i] += x[i]**2

# Forward mode: Compute J*v directly (efficient!)
Jv = ca.jtimes(f, x, v)

# Create function
jv_func = ca.Function('Jv', [x, v], [Jv])

# Compare with computing full Jacobian (slower for large dimensions)
J_full = ca.jacobian(f, x)
J_func = ca.Function('J', [x], [J_full])

print(f"Input dimension: {n}, Output dimension: {m}")
print("Forward mode (jtimes) computes J*v directly - EFFICIENT!")
print("Much faster than computing full Jacobian and then multiplying")


# ============================================================================
# EXAMPLE 4: Newton Solver with Automatic Differentiation
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: Newton Solver for Nonlinear System")
print("="*70)

# Solve F(x) = 0 using Newton's method
# Newton iteration: x_new = x - J^(-1) * F(x)
# where J = dF/dx

n = 5
x = ca.SX.sym('x', n)

# Define system of equations F(x) = 0
# Example: Find equilibrium of a dynamical system
F = ca.vertcat(
    x[0]**2 + x[1]**2 - 4,           # Circle constraint
    x[2] - ca.sin(x[0]) - ca.cos(x[1]),  # Nonlinear relation
    x[3]**3 - x[0]*x[2] + 1,         # Cubic
    x[4] - ca.exp(-x[3]) + 0.5,      # Exponential
    ca.sum1(x) - 2                    # Sum constraint
)

# Create Newton solver using CasADi's rootfinder
# Note: The Jacobian is computed AUTOMATICALLY inside the solver!
# No need to manually compute dF/dx - that's the beauty of CasADi
newton_solver = ca.rootfinder('newton_solver', 'newton', 
                             {'x': x, 'g': F})

# Initial guess
x0 = np.array([1.0, 1.0, 0.5, 0.5, -1.0])

# Solve (pass as dict with initial guess and zero for the 'p' parameter)
solution = newton_solver(x0, [])
print(f"\nSolution: {solution}")

# Verify solution
F_func = ca.Function('F', [x], [F])
residual = F_func(solution)
print(f"Residual |F(x*)|: {np.linalg.norm(residual)}")


# ============================================================================
# EXAMPLE 5: Code Generation for Maximum Efficiency
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 5: Code Generation - No Symbolic Explosion!")
print("="*70)

# Even if symbolic expressions become huge, CasADi can generate
# efficient C code that evaluates them

n = 20
x = ca.SX.sym('x', n)

# Create a very complex expression through recursion
f = x
for iteration in range(10):  # Multiple layers of complexity
    # Nonlinear transformation
    f_temp = ca.SX.zeros(n)
    for i in range(n):
        f_temp[i] = ca.sin(f[i]) + ca.tanh(f[(i+1) % n]) * f[(i-1) % n]
    f = f_temp

# Compute Jacobian of this highly complex function
J = ca.jacobian(f, x)

# Create function
complex_func = ca.Function('complex_func', [x], [f, J])

# OPTION 1: Just evaluate normally (CasADi is already efficient)
# OPTION 2: Generate C code for even faster evaluation
# complex_func.generate('complex_func.c')  # Generates efficient C code

# Evaluate
x_val = np.ones(n) * 0.1
f_val, J_val = complex_func(x_val)

print(f"Complex function evaluated successfully!")
print(f"Function output shape: {f_val.shape}")
print(f"Jacobian shape: {J_val.shape}")
print(f"Function sparsity: {complex_func.n_instructions()} operations")
print("\nKey insight: No matter how complex, CasADi stores as computational graph,")
print("not as symbolic expression. This prevents explosion!")


# ============================================================================
# EXAMPLE 6: Practical Flat Output System
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 6: Flat Output System - Quadrotor Example")
print("="*70)

# Flat output for quadrotor: z = [x, y, z, yaw]
# Full state can be recovered from flat output and derivatives

t = ca.SX.sym('t')  # Time parameter

# Flat outputs (position + yaw trajectory)
sigma = ca.vertcat(
    ca.SX.sym('x', 1),      # x position
    ca.SX.sym('y', 1),      # y position  
    ca.SX.sym('z', 1),      # z position
    ca.SX.sym('yaw', 1)     # yaw angle
)

# Derivatives of flat output
sigma_dot = ca.SX.sym('sigma_dot', 4)
sigma_ddot = ca.SX.sym('sigma_ddot', 4)
sigma_dddot = ca.SX.sym('sigma_dddot', 4)

# Physical parameters
g = 9.81  # gravity
m = 1.0   # mass

# Compute full state from flat output
# Thrust direction from acceleration
a = ca.vertcat(sigma_ddot[0], sigma_ddot[1], sigma_ddot[2] + g)
thrust_mag = ca.norm_2(a)

# Orientation from thrust direction
zB = a / thrust_mag  # Body z-axis

# Pitch and roll from thrust direction and yaw
xC = ca.vertcat(ca.cos(sigma[3]), ca.sin(sigma[3]), 0)  # Desired x in world frame
yB = ca.cross(zB, xC)
yB = yB / ca.norm_2(yB)
xB = ca.cross(yB, zB)

# Full state: position, velocity, orientation, body rates
position = sigma[0:3]
velocity = sigma_dot[0:3]

# This would continue to compute angular velocities from derivatives...
# The point: multiple layers of differentiation and algebraic relations

# Compute Jacobian of position w.r.t. flat output derivatives
J_pos = ca.jacobian(position, sigma)
J_vel = ca.jacobian(velocity, ca.vertcat(sigma, sigma_dot))

print(f"Flat output dimension: {sigma.shape[0]}")
print(f"Position Jacobian w.r.t. flat output: {J_pos.shape}")
print("For flat systems, you need many Jacobians of compositions - CasADi handles this efficiently!")


# ============================================================================
# TIPS FOR YOUR COLLEAGUE
# ============================================================================
print("\n" + "="*70)
print("RECOMMENDATIONS FOR YOUR COLLEAGUE")
print("="*70)
print("""
1. USE SX (scalar expressions) for small-medium problems
   USE MX (matrix expressions) for very large problems

2. For Jacobian-vector products, use ca.jtimes() instead of computing full Jacobian
   - Much more efficient for high-dimensional systems
   
3. For vector-Jacobian products (reverse mode), use ca.jtimes() with reverse=True
   - Efficient when output dimension < input dimension

4. Enable Just-In-Time compilation with .jit() for faster evaluation
   - Can also generate C code with .generate() for production

5. For Newton solver, use ca.rootfinder() with automatic differentiation
   - No need to manually compute or code Jacobians!

6. CasADi stores computational graphs, not symbolic expressions
   - This prevents the "millions of characters" problem from Mathematica
   - Memory usage stays manageable even for very complex expressions

7. Can interface with existing solvers: IPOPT, SNOPT, WORHP, etc.

8. For flat output systems:
   - Define flat outputs and their derivatives as symbolic variables
   - Build state reconstruction as CasADi functions
   - Automatically differentiate through the entire chain
   - Use for trajectory optimization and control

9. Sparse Jacobians are automatically detected and exploited

10. Can save/load functions to avoid recompilation
""")

print("\nGenerated by CasADi - Perfect for automatic differentiation!")
