def eq0909(X_f, X_i):
    # Calculate the change in X
    delta_X = X_f - X_i
    
    # Check if the condition is satisfied
    is_satisfied = delta_X >= 0
    
    return is_satisfied, delta_X
