import sympy as sp

def eq0402(p, t):
    # Define the symbols for p and t if they are not already defined
    if not isinstance(p, sp.Symbol):
        p = sp.symbols('p')
    if not isinstance(t, sp.Symbol):
        t = sp.symbols('t')
    
    # Calculate dp/dt
    dp_dt = sp.diff(p, t)
    
    # Calculate varpi
    varpi = dp_dt
    
    return varpi
