import numpy as np

def eq10_21(U_A, B, gamma_MT):
    """
    Calculate k_UB based on the given values of U_A, B, and gamma_MT.

    :param U_A: Value of U_A
    :param B: Value of B
    :param gamma_MT: Value of gamma_MT
    :return: Calculated value of k_UB

    # Example usage
    U_A_value = 4       # Replace with the actual value
    B_value = 2         # Replace with the actual value
    gamma_MT_value = 3  # Replace with the actual value
    k_UB = eq10_21(U_A_value, B_value, gamma_MT_value)
    print("k_UB:", k_UB)
    """
    k_UB = U_A * np.power(B, gamma_MT - 1)
    return k_UB


