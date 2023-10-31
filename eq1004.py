# Importing SymPy for symbolic mathematics
import sympy as sp

# Defining the symbols U_P, F_A, N and the functions T(U_P, F_A), X(U_P, F_A), and phi(F_A)
U_P, F_A, N = sp.symbols('U_P F_A N')
T = sp.Function('T')(U_P, F_A)
X = sp.Function('X')(U_P, F_A)
phi_FA = sp.Function('phi')(F_A)

# Calculating the partial derivatives of T and X with respect to U_P
dT_dUP = sp.diff(T, U_P)
dX_dUP = sp.diff(X, U_P)

# Defining a function to evaluate the derivatives for specific values of U_P, F_A, and N
def eq10_04(U_P_value, F_A_value, N_value):
    """
    This function calculates the partial derivatives of T and X with respect to U_P for given values of U_P, F_A, and N.
    The equation number is Eq10-04, and it represents the following equation:
    ϕ_{F_A} = (1/N) * (∂T/∂U_P)_{F_A} = (1/N) * (∂X/∂U_P)_{F_A}
    
    :param U_P_value: The value of U_P for which the derivatives will be evaluated.
    :param F_A_value: The value of F_A for which the derivatives will be evaluated.
    :param N_value: The value of N.
    :return: A tuple containing the evaluated derivatives of T and X with respect to U_P, and the value of ϕ_{F_A}.
    """
    # Substituting the values of U_P, F_A, and N, and evaluating the derivatives
    dT_dUP_eval = dT_dUP.subs({U_P: U_P_value, F_A: F_A_value})
    dX_dUP_eval = dX_dUP.subs({U_P: U_P_value, F_A: F_A_value})
    phi_FA_eval = dT_dUP_eval / N_value  # Since both derivatives are equal
    
    return dT_dUP_eval, dX_dUP_eval, phi_FA_eval
