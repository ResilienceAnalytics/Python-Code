import sympy as sp

def eq0423(c_A, c_P, dvarpi, domega):
    # Calculate the ratio
    ratio = -(c_A / c_P) * (dvarpi / domega)
    
    return ratio
