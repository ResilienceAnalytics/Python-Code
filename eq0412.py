import sympy as sp

def eq0412(c_A, dvarpi_dt, dV):
    # Calculate dM
    dM = -c_A * dvarpi_dt * dV
    
    return dM
