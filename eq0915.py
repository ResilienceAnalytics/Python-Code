# Importing SymPy for symbolic mathematics
import sympy as sp

def eq09_15(F_A_value):
    """
    This function calculates the partial derivatives of T and X with respect to F_A for a given value of F_A.
    The equation number is eq09-15, and it represents the following equation:
    ∂T/∂F_A = ∂X/∂F_A
    
    :param F_A_value: The value of F_A for which the derivatives will be evaluated.
    :return: A tuple containing the evaluated derivatives of T and X with respect to F_A.
    """
    # Defining the symbol F_A and the functions T(F_A) and X(F_A)
    F_A = sp.symbols('F_A')
    T = sp.Function('T')(F_A)
    X = sp.Function('X')(F_A)

    # Calculating the partial derivatives of T and X with respect to F_A
    dT_dFA = sp.diff(T, F_A)
    dX_dFA = sp.diff(X, F_A)

    # Substituting the value of F_A and evaluating the derivatives
    return dT_dFA.subs(F_A, F_A_value), dX_dFA.subs(F_A, F_A_value)
