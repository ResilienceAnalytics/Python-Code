def eq11_01(delta_T_lost, delta_T_useful):
    """
    Calculate the total change in temperature.

    :param delta_T_lost: Change in temperature due to losses
    :param delta_T_useful: Useful change in temperature
    :return: Total change in temperature
    """
    delta_T_total = delta_T_lost + delta_T_useful
    return delta_T_total

# Example usage
delta_T_lost_value = 4       # Replace with the actual value
delta_T_useful_value = 2     # Replace with the actual value
delta_T_total = eq11_01(delta_T_lost_value, delta_T_useful_value)
print("Delta T Total:", delta_T_total)
