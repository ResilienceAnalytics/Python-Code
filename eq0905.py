def eq0905(delta_A, delta_E_MT, U_A0):
    # Calculate the condition
    condition = delta_A - (delta_E_MT / U_A0)
    
    # Check if the condition is satisfied
    is_satisfied = condition >= 0
    
    return is_satisfied, condition
