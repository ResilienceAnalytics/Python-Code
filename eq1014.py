import sympy as sp

def eq10_14(U_A, F_P, M, L):
    """
    Calculate the value of Phi_{F_P} based on the partial derivatives of M and L with respect to U_A.

    :param U_A: Symbol representing U_A
    :param F_P: Symbol representing F_P, another parameter in the equation
    :param M: Expression representing the function M(U_A, F_P)
    :param L: Expression representing the function L(U_A, F_P)
    :return: A tuple containing the calculated values of Phi_{F_P} from M and L
    """
    # Calculate the partial derivatives
    partial_M_U_A = sp.diff(M, U_A)
    partial_L_U_A = sp.diff(L, U_A)
    
    # Evaluate the partial derivatives at F_P
    Phi_F_P_M = partial_M_U_A.subs(F_P, F_P)
    Phi_F_P_L = partial_L_U_A.subs(F_P, F_P)
    
    return Phi_F_P_M, Phi_F_P_L

# Define the symbols
U_A = sp.symbols('U_A')
F_P = sp.symbols('F_P')  # Other variables that M and L might depend on

# Define the functions M and L as examples (replace with actual functions)
M = U_A**2 + F_P*U_A + 1
L = U_A**3 - F_P*U_A**2 + 2

# Example usage
Phi_F_P_M, Phi_F_P_L = eq10_14(U_A, F_P, M, L)
print("Phi_{F_P} from M:", Phi_F_P_M)
print("Phi_{F_P} from L:", Phi_F_P_L)
