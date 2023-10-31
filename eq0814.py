def eq0814(l_1, U_A1, l_2, U_A2):
    # Calculate the ratios
    ratio_1 = l_1 / U_A1
    ratio_2 = l_2 / U_A2
    
    # Check if the equality is satisfied
    equality_satisfied = ratio_1 == ratio_2
    
    return equality_satisfied, ratio_1, ratio_2
