import sympy as sp

def eq0425(M, t):
    # Calculate dM/dt
    dM_dt = sp.diff(M, t)
    
    # Γ_M is equal to dM/dt
    Gamma_M = dM_dt
    
    return Gamma_M
