def eq0604(U_A, dA, F_P_ext, dB):
    # Calculate the cause-related change
    cause_change = U_A * dA
    
    # Calculate the effect-related change
    effect_change = -F_P_ext * dB
    
    # Calculate dE_MT
    dE_MT = cause_change + effect_change
    
    return dE_MT, cause_change, effect_change
