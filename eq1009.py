import sympy as sp

def eq10_09(F_A, V, gamma_TM):
    """
    Calculate the value of k_FV based on the provided F_A, V, and gamma_TM values.

    :param F_A: The value for F_A, a parameter in the equation
    :param V: The value for V, another parameter in the equation
    :param gamma_TM: The value for gamma_TM, another parameter in the equation
    :return: The calculated value of k_FV
    """
    # Ensure that the input values are either numbers or SymPy expressions
    F_A = sp.sympify(F_A)
    V = sp.sympify(V)
    gamma_TM = sp.sympify(gamma_TM)
    
    # Calculate k_FV using the formula k_FV = F_A * V ** gamma_TM
    k_FV = F_A * V ** gamma_TM
    
    return k_FV

# Example usage:
F_A_value = 2
V_value = 3
gamma_TM_value = 4
result = eq10_09(F_A_value, V_value, gamma_TM_value)
print("Result of eq10-09:", result)  # Expected output: 162
