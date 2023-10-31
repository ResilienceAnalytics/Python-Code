import numpy as np

def eq10_20(F_P, B, gamma_MT):
    """
    Calculate k_FB based on the given values of F_P, B, and gamma_MT.

    :param F_P: Value of F_P
    :param B: Value of B
    :param gamma_MT: Value of gamma_MT
    :return: Calculated value of k_FB
    """
    k_FB = F_P * np.power(B, gamma_MT)
    return k_FB

# Example usage
F_P_value = 4       # Replace with the actual value
B_value = 2         # Replace with the actual value
gamma_MT_value = 3  # Replace with the actual value
k_FB = eq10_20(F_P_value, B_value, gamma_MT_value)
print("k_FB:", k_FB)
