import sympy as sp

def eq10_10(U_P, V, gamma_TM):
    """
    Calculate the value of k_UV based on the provided U_P, V, and gamma_TM values.

    :param U_P: The value for U_P, a parameter in the equation
    :param V: The value for V, another parameter in the equation
    :param gamma_TM: The value for gamma_TM, another parameter in the equation
    :return: The calculated value of k_UV

    # Example usage:
    U_P_value = 2
    V_value = 3
    gamma_TM_value = 4
    result = eq10_10(U_P_value, V_value, gamma_TM_value)
    print("Result of eq10-10:", result)  # Expected output: 54
    """
    # Ensure that the input values are either numbers or SymPy expressions
    U_P = sp.sympify(U_P)
    V = sp.sympify(V)
    gamma_TM = sp.sympify(gamma_TM)
    
    # Calculate k_UV using the formula k_UV = U_P * V ** (gamma_TM - 1)
    k_UV = U_P * V ** (gamma_TM - 1)
    
    return k_UV


