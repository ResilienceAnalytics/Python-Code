def eq0808(F_A1, p_1, F_A2, p_2):
    # Calculate the products
    product_1 = F_A1 * p_1
    product_2 = F_A2 * p_2
    
    # Check if the equality is satisfied
    equality_satisfied = product_1 == product_2
    
    return equality_satisfied, product_1, product_2
