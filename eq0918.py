# Importing SymPy for symbolic mathematics
import sympy as sp

# Defining a function to evaluate the derivatives for a specific value of l
def eq09_18(l_value):
    """
    This function calculates the partial derivatives of M and E_MT with respect to l for a given value of l.
    The equation number is eq09-18, and it represents the following equation:
    ∂M/∂l = ∂E_MT/∂l
    
    :param l_value: The value of l for which the derivatives will be evaluated.
    :return: A tuple containing the evaluated derivatives of M and E_MT with respect to l.

    # Defining the symbol l and the functions M(l) and E_MT(l)
    l = sp.symbols('l')
    M = sp.Function('M')(l)
    E_MT = sp.Function('E_MT')(l)

    # Calculating the partial derivatives of M and E_MT with respect to l
    dM_dl = sp.diff(M, l)
    dE_MT_dl = sp.diff(E_MT, l)
    """
    # Substituting the value of l and evaluating the derivatives
    return dM_dl.subs(l, l_value), dE_MT_dl.subs(l, l_value)
