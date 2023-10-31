def eq0602(U_P, dP, F_A_ext, dV):
    # Calculate the cause-related change
    cause_change = -U_P * dP
    
    # Calculate the effect-related change
    effect_change = F_A_ext * dV
    
    # Calculate dE_TM
    dE_TM = cause_change + effect_change
    
    return dE_TM, cause_change, effect_change
