import sympy as sp

def eq0411(c_P, domega_dt, dB):
    # Calculate dT
    dT = -c_P * domega_dt * dB
    
    return dT
