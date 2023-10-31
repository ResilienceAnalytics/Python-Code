def eq0914(U_A, dA, l, dF_P):
    # Calculate the differential change in L
    dL = U_A * dA + l * dF_P
    
    return dL
