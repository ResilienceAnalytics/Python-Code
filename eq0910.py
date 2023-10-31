def eq0910(U_P, dP, p, dF_A):
    # Calculate the change in X
    dX = -U_P * dP - p * dF_A
    
    return dX
