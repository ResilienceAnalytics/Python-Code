def eq0903(delta_P, delta_E_TM, U_P0):
    # Calculate the condition
    condition = delta_P + (delta_E_TM / U_P0)
    
    # Check if the condition is satisfied
    is_satisfied = condition >= 0
    
    return is_satisfied, condition
