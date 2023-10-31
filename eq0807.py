def eq0807(F_A1, U_P1, F_A2, U_P2):
    # Calculate the ratios
    ratio_1 = F_A1 / U_P1
    ratio_2 = F_A2 / U_P2
    
    # Check if the equality is satisfied
    equality_satisfied = ratio_1 == ratio_2
    
    return equality_satisfied, ratio_1, ratio_2
