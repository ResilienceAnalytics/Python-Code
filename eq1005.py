# Importing SymPy for symbolic mathematics
import sympy as sp



# Defining a function to evaluate the equation for specific values of Phi_V and Phi_FA
def eq10_05(Phi_V_value, Phi_FA_value):
    """
    This function calculates the value of r_P based on the values of Phi_V and Phi_FA.
    The equation number is Eq10-05, and it represents the following equation:
    r_P = Phi_V - Phi_FA
    
    :param Phi_V_value: The value of Phi_V.
    :param Phi_FA_value: The value of Phi_FA.
    :return: The calculated value of r_P.

    # Exemple 
    # Defining the symbols r_P, Phi_V, and Phi_FA
    r_P, Phi_V, Phi_FA = sp.symbols('r_P Phi_V Phi_FA')

    # Defining the equation r_P = Phi_V - Phi_FA
    eq = sp.Eq(r_P, Phi_V - Phi_FA)
    """
    r_P_value = eq.subs({Phi_V: Phi_V_value, Phi_FA: Phi_FA_value})
    return r_P_value
