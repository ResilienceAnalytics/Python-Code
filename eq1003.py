# Importing SymPy for symbolic mathematics
import sympy as sp


# Defining a function to evaluate the derivatives for specific values of U_P and F_A
def eq10_03(U_P_value, F_A_value):
    """
    This function calculates the partial derivatives of T and X with respect to U_P for given values of U_P and F_A.
    The equation number is Eq10-03, and it represents the following equation:
    Φ_{F_A} = (∂T/∂U_P)_{F_A} = (∂X/∂U_P)_{F_A}
    
    :param U_P_value: The value of U_P for which the derivatives will be evaluated.
    :param F_A_value: The value of F_A for which the derivatives will be evaluated.
    :return: A tuple containing the evaluated derivatives of T and X with respect to U_P, and the value of Φ_{F_A}.

    # Defining the symbols U_P, F_A and the functions T(U_P, F_A), X(U_P, F_A), and Phi(F_A)
    U_P, F_A = sp.symbols('U_P F_A')
    T = sp.Function('T')(U_P, F_A)
    X = sp.Function('X')(U_P, F_A)
    Phi_FA = sp.Function('Phi')(F_A)

    # Calculating the partial derivatives of T and X with respect to U_P
    dT_dUP = sp.diff(T, U_P)
    dX_dUP = sp.diff(X, U_P)
    """
    # Substituting the values of U_P and F_A, and evaluating the derivatives
    dT_dUP_eval = dT_dUP.subs({U_P: U_P_value, F_A: F_A_value})
    dX_dUP_eval = dX_dUP.subs({U_P: U_P_value, F_A: F_A_value})
    Phi_FA_eval = dT_dUP_eval  # Since both derivatives are equal
    
    return dT_dUP_eval, dX_dUP_eval, Phi_FA_eval
