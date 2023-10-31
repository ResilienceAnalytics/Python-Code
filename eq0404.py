import sympy as sp

def eq0404(p, t):
    # Define the symbols for p and t if they are not already defined
    if not isinstance(p, sp.Symbol):
        p = sp.symbols('p')
    if not isinstance(t, sp.Symbol):
        t = sp.symbols('t')
    
    # Calculate d(varpi)/dt
    varpi = sp.diff(p, t)
    dvarpi_dt = sp.diff(varpi, t)
    
    # Calculate d^2(p)/dt^2
    d2p_dt2 = sp.diff(p, t, 2)
    
    return dvarpi_dt, d2p_dt2
