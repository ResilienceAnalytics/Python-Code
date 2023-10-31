import sympy as sp

def eq10_13(U_A, B, M, E_MT, N):
    """
    Calculate the value of phi_B based on the partial derivatives of M and E_MT with respect to U_A, and a constant N.

    :param U_A: Symbol representing U_A
    :param B: Symbol representing B, another parameter in the equation
    :param M: Expression representing the function M(U_A, B)
    :param E_MT: Expression representing the function E_MT(U_A, B)
    :param N: The constant N in the equation
    :return: A tuple containing the calculated values of phi_B from M and E_MT

    # Define the symbols
    U_A = sp.symbols('U_A')
    B = sp.symbols('B')  # Other variables that M and E_MT might depend on

    # Define the constant N
    N = sp.symbols('N')

    # Define the functions M and E_MT as examples (replace with actual functions)
    M = U_A**2 + B*U_A + 1
    E_MT = U_A**3 - B*U_A**2 + 2

    # Example usage
    N_value = 4  # Replace with the actual value
    phi_B_M, phi_B_E_MT = eq10_13(U_A, B, M, E_MT, N_value)
    print("phi_B from M:", phi_B_M)
    print("phi_B from E_MT:", phi_B_E_MT)
    """
    # Calculate the partial derivatives
    partial_M_U_A = sp.diff(M, U_A)
    partial_E_MT_U_A = sp.diff(E_MT, U_A)
    
    # Evaluate the partial derivatives at B and divide by N
    phi_B_M = (1/N) * partial_M_U_A.subs(B, B)
    phi_B_E_MT = (1/N) * partial_E_MT_U_A.subs(B, B)
    
    return phi_B_M, phi_B_E_MT


