# Importing SymPy for symbolic mathematics
import sympy as sp

# Defining a function to evaluate the derivatives for specific values of U_P, V, and N
def eq10_02(U_P_value, V_value, N_value):
    """
    This function calculates the partial derivatives of T and E_TM with respect to U_P for given values of U_P, V, and N.
    The equation number is Eq10-02, and it represents the following equation:
    ϕ_V = (1/N) * (∂T/∂U_P)_V = (1/N) * (∂E_TM/∂U_P)_V
    
    :param U_P_value: The value of U_P for which the derivatives will be evaluated.
    :param V_value: The value of V for which the derivatives will be evaluated.
    :param N_value: The value of N.
    :return: A tuple containing the evaluated derivatives of T and E_TM with respect to U_P, and the value of ϕ_V.

    # Defining the symbols U_P, V, N and the functions T(U_P, V), E_TM(U_P, V), and phi(V)
    U_P, V, N = sp.symbols('U_P V N')
    T = sp.Function('T')(U_P, V)
    E_TM = sp.Function('E_TM')(U_P, V)
    phi_V = sp.Function('phi')(V)

    # Calculating the partial derivatives of T and E_TM with respect to U_P
    dT_dUP = sp.diff(T, U_P)
    dE_TM_dUP = sp.diff(E_TM, U_P)
    """
    # Substituting the values of U_P, V, and N, and evaluating the derivatives
    dT_dUP_eval = dT_dUP.subs({U_P: U_P_value, V: V_value})
    dE_TM_dUP_eval = dE_TM_dUP.subs({U_P: U_P_value, V: V_value})
    phi_V_eval = dT_dUP_eval / N_value  # Since both derivatives are equal
    
    return dT_dUP_eval, dE_TM_dUP_eval, phi_V_eval
