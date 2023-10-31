def eq0503(g_MT):
    # Check if g_MT is zero to avoid division by zero
    if g_MT == 0:
        raise ValueError("g_MT cannot be zero.")
    
    # Calculate g_TM
    g_TM = 1 / g_MT
    
    return g_TM
