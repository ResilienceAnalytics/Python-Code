import sympy as sp

def eq0302(R, m, i):
    # Check if i is within the range of m's dimensions
    if i < 0 or i >= len(m):
        raise ValueError("Index i is out of range for vector m")
    
    # Define the symbols for m
    m_symbols = sp.symbols('m0:%d' % len(m))
    
    # Replace m's components with corresponding symbols in R
    R_substituted = R.subs({m[j]: m_symbols[j] for j in range(len(m))})
    
    # Calculate the partial derivative of R with respect to m_i
    dR_m_i = sp.diff(R_substituted, m_symbols[i])
    
    # Check if dR_m_i is greater than 0
    condition = dR_m_i > 0
    
    return dR_m_i, condition
