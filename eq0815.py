def eq0815(F_P1, U_A1, F_P2, U_A2):
    # Calculate the ratios
    ratio_1 = F_P1 / U_A1
    ratio_2 = F_P2 / U_A2
    
    # Check if the equality is satisfied
    equality_satisfied = ratio_1 == ratio_2
    
    return equality_satisfied, ratio_1, ratio_2
