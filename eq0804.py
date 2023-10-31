def eq0804(F_A, p, r_P, U_P):
    # Calculate the left-hand side of the equation
    lhs = F_A * p
    
    # Calculate the right-hand side of the equation
    rhs = r_P * U_P
    
    # Check if the equation is satisfied
    equation_satisfied = lhs == rhs
    
    return equation_satisfied, lhs, rhs
