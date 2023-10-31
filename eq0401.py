import sympy as sp

def eq0401(l, t):
    # Define the symbols for l and t if they are not already defined
    if not isinstance(l, sp.Symbol):
        l = sp.symbols('l')
    if not isinstance(t, sp.Symbol):
        t = sp.symbols('t')
    
    # Calculate dl/dt
    dl_dt = sp.diff(l, t)
    
    # Calculate omega
    omega = -dl_dt
    
    return omega
