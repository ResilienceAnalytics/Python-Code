import sympy as sp

def eq0424(F_A, dp_dx, n):
    # Calculate (∂u/∂x)_n
    partial_u_partial_x_at_n = -F_A * (dp_dx / n)
    
    return partial_u_partial_x_at_n
