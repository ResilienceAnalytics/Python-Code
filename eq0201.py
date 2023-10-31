import sympy as sp
import numpy as np

def eq0201(n, u):
    # Define the symbols
    n_sym, u_sym = sp.symbols('n u')
    
    # Define the equation
    equation = sp.Eq(n_sym * u_sym)
    
    # Substitute the values
    result = equation.subs({n_sym: n, u_sym: u})
    
    # Evaluate the expression
    numerical_result = np.float64(result)
    
    return numerical_result
