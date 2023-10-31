def eq0816(F_P1, l_1, F_P2, l_2):
    # Calculate the products
    product_1 = F_P1 * l_1
    product_2 = F_P2 * l_2
    
    # Check if the equality is satisfied
    equality_satisfied = product_1 == product_2
    
    return equality_satisfied, product_1, product_2
