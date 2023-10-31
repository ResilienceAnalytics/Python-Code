# Importing SymPy for symbolic mathematics
import sympy as sp

# Defining the symbol p and the functions T(p) and E_TM(p)
p = sp.symbols('p')
T = sp.Function('T')(p)
E_TM = sp.Function('E_TM')(p)

# Calculating the partial derivatives of T and E_TM with respect to p
dT_dp = sp.diff(T, p)
dE_TM_dp = sp.diff(E_TM, p)

# Defining a function to evaluate the derivatives for a specific value of p
def eq09_16(p_value):
    """
    This function calculates the partial derivatives of T and E_TM with respect to p for a given value of p.
    The equation number is eq09-16, and it represents the following equation:
    ∂T/∂p = ∂E_TM/∂p
    
    :param p_value: The value of p for which the derivatives will be evaluated.
    :return: A tuple containing the evaluated derivatives of T and E_TM with respect to p.
    """
    # Substituting the value of p and evaluating the derivatives
    return dT_dp.subs(p, p_value), dE_TM_dp.subs(p, p_value)
