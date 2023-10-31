# Importing SymPy for symbolic mathematics
import sympy as sp

# Defining the symbols U_P, V and the functions T(U_P, V), E_TM(U_P, V), and Phi(V)
U_P, V = sp.symbols('U_P V')
T = sp.Function('T')(U_P, V)
E_TM = sp.Function('E_TM')(U_P, V)
Phi_V = sp.Function('Phi')(V)

# Calculating the partial derivatives of T and E_TM with respect to U_P
dT_dUP = sp.diff(T, U_P)
dE_TM_dUP = sp.diff(E_TM, U_P)

# Defining a function to evaluate the derivatives for specific values of U_P and V
def eq10_01(U_P_value, V_value):
    """
    This function calculates the partial derivatives of T and E_TM with respect to U_P for given values of U_P and V.
    The equation number is Eq10-01, and it represents the following equation:
    Φ_V = (∂T/∂U_P)_V = (∂E_TM/∂U_P)_V
    
    :param U_P_value: The value of U_P for which the derivatives will be evaluated.
    :param V_value: The value of V for which the derivatives will be evaluated.
    :return: A tuple containing the evaluated derivatives of T and E_TM with respect to U_P, and the value of Φ_V.
    """
    # Substituting the values of U_P and V, and evaluating the derivatives
    dT_dUP_eval = dT_dUP.subs({U_P: U_P_value, V: V_value})
    dE_TM_dUP_eval = dE_TM_dUP.subs({U_P: U_P_value, V: V_value})
    Phi_V_eval = dT_dUP_eval  # Since both derivatives are equal
    
    return dT_dUP_eval, dE_TM_dUP_eval, Phi_V_eval
