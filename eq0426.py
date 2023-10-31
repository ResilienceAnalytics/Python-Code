import sympy as sp

def eq0426(T, t):
    # Calculate dT/dt
    dT_dt = sp.diff(T, t)
    
    # Γ_T is equal to dT/dt
    Gamma_T = dT_dt
    
    return Gamma_T
