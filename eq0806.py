def eq0806(p_1, U_P1, p_2, U_P2):
    # Calculate the ratios
    ratio_1 = p_1 / U_P1
    ratio_2 = p_2 / U_P2
    
    # Check if the equality is satisfied
    equality_satisfied = ratio_1 == ratio_2
    
    return equality_satisfied, ratio_1, ratio_2
