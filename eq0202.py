import sympy as sp
import numpy as np

def eq0202(n, u, dn, du):
    # Define the symbols
    n_sym, u_sym, dn_sym, du_sym = sp.symbols('n u dn du')
    
    # Define the equation
    equation = sp.Eq(n_sym * du_sym + u_sym * dn_sym)
    
    # Substitute the values
    result = equation.subs({n_sym: n, u_sym: u, dn_sym: dn, du_sym: du})
    
    # Evaluate the expression
    numerical_result = np.float64(result)
    
    return numerical_result
