def eq0904(delta_P_Entrepreneur, T, U_P0):
    # Calculate the right-hand side of the inequality
    rhs = T / U_P0
    
    # Check if the condition is satisfied
    is_satisfied = delta_P_Entrepreneur <= rhs
    
    return is_satisfied, rhs
