def eq10_18(Phi_F_P, Phi_B):
    """
    Calculate the value of gamma_MT based on the given values of Phi_F_P and Phi_B.

    :param Phi_F_P: Value of Phi_{F_P}
    :param Phi_B: Value of Phi_B
    :return: Calculated value of gamma_MT

    # Example usage
    Phi_F_P_value = 5  # Replace with the actual value
    Phi_B_value = 3    # Replace with the actual value
    gamma_MT = eq10_18(Phi_F_P_value, Phi_B_value)
    print("gamma_MT:", gamma_MT)
    """
    if Phi_B == 0:
        return "Division by zero error"
    gamma_MT = Phi_F_P / Phi_B
    return gamma_MT


