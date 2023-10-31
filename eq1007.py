import sympy as sp
import numpy as np

def eq10_07(phi_F_A, phi_V):
    """
    Calculate the value of gamma_TM based on the provided phi_F_A and phi_V values.

    :param phi_F_A: The value for Phi_F_A, a parameter in the equation
    :param phi_V: The value for Phi_V, another parameter in the equation
    :return: The calculated value of gamma_TM
    """
    
    # Ensure that the input values are either numbers or SymPy expressions
    phi_F_A = sp.sympify(phi_F_A)
    phi_V = sp.sympify(phi_V)
    
    # Check if the denominator is zero, as division by zero is undefined
    if phi_V == 0:
        raise ValueError("Error: Division by zero. Phi_V must be non-zero.")
    
    # Calculate gamma_TM using the formula gamma_TM = Phi_F_A / Phi_V
    gamma_TM = phi_F_A / phi_V
    
    return gamma_TM

# Example usage:
phi_F_A_value = 10
phi_V_value = 5
result = eq10_07(phi_F_A_value, phi_V_value)
print("Result of eq10-07:", result)  # Expected output: 2.0
