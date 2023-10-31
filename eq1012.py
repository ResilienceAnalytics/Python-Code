import sympy as sp

def eq10_12(U_A, B, M, E_MT):
    """
    Calculate the value of Phi_B based on the partial derivatives of M and E_MT with respect to U_A.

    :param U_A: Symbol representing U_A
    :param B: Symbol representing B, another parameter in the equation
    :param M: Expression representing the function M(U_A, B)
    :param E_MT: Expression representing the function E_MT(U_A, B)
    :return: A tuple containing the calculated values of Phi_B from M and E_MT
    """
    # Calculate the partial derivatives
    partial_M_U_A = sp.diff(M, U_A)
    partial_E_MT_U_A = sp.diff(E_MT, U_A)
    
    # Evaluate the partial derivatives at B
    Phi_B_M = partial_M_U_A.subs(B, B)
    Phi_B_E_MT = partial_E_MT_U_A.subs(B, B)
    
    return Phi_B_M, Phi_B_E_MT

# Define the symbols
U_A = sp.symbols('U_A')
B = sp.symbols('B')  # Other variables that M and E_MT might depend on

# Define the functions M and E_MT as examples (replace with actual functions)
M = U_A**2 + B*U_A + 1
E_MT = U_A**3 - B*U_A**2 + 2

# Example usage
Phi_B_M, Phi_B_E_MT = eq10_12(U_A, B, M, E_MT)
print("Phi_B from M:", Phi_B_M)
print("Phi_B from E_MT:", Phi_B_E_MT)
