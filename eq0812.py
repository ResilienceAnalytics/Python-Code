def eq0812(F_P, l, r_A, U_A):
    # Calculate the products
    product_1 = F_P * l
    product_2 = r_A * U_A
    
    # Check if the equality is satisfied
    equality_satisfied = product_1 == product_2
    
    return equality_satisfied, product_1, product_2
