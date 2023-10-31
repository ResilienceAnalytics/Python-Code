import numpy as np

def eq10_22(U_A, F_P, gamma_MT):
    """
    Calculate k_UF based on the given values of U_A, F_P, and gamma_MT.

    :param U_A: Value of U_A
    :param F_P: Value of F_P
    :param gamma_MT: Value of gamma_MT
    :return: Calculated value of k_UF
    """
    k_UF = np.power(U_A, gamma_MT) * np.power(F_P, 1 - gamma_MT)
    return k_UF

# Example usage
#U_A_value = 4       # Replace with the actual value
#F_P_value = 2       # Replace with the actual value
#gamma_MT_value = 3  # Replace with the actual value
#k_UF = eq10_22(U_A_value, F_P_value, gamma_MT_value)
#print("k_UF:", k_UF)
