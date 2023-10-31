def eq10_08(gamma_TM):
    """
    Check if gamma_TM is less than 1.

    :param gamma_TM: The value of gamma_TM to be checked
    :return: True if gamma_TM is less than 1, False otherwise
    """
    # Ensure that the input value is a number or a SymPy expression
    gamma_TM = sp.sympify(gamma_TM)
    
    # Check if gamma_TM is less than 1
    return gamma_TM < 1

# Example usage:
#gamma_TM_value = 0.5
#result = eq10_08(gamma_TM_value)
#print("Is gamma_TM < 1?", result)  
# Expected output: True
