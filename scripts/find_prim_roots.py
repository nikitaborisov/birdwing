from math import gcd

def find_2power_root(p, m, k, odd_part_factors):
    """
    Given a prime p = 2^k * m + 1, returns:
        - g: a primitive root mod p
        - root: a primitive 2^k-th root of unity mod p

    Parameters:
        p: the prime modulus
        m: odd part of p-1
        k: such that p = 2^k * m + 1
        odd_part_factors: list of prime factors of m
    """
    
    # All prime factors of p - 1 = 2^k * m
    all_factors = [2] + odd_part_factors
    
    def is_primitive_root(g):
        for q in all_factors:
            if pow(g, (p - 1)//q, p) == 1:
                return False
        return True

    # Search for a primitive root
    for g in range(2, p):  # start from 2
        if is_primitive_root(g):
            break
    else:
        raise ValueError("No primitive root found (should never happen for prime p).")

    # Compute primitive 2^k-th root of unity: g^m mod p
    root = pow(g, m, p)

    # Verification
    assert pow(root, 2**k, p) == 1, "Not a 2^k-th root of unity!"
    assert pow(root, 2**(k-1), p) != 1, "Order is not exactly 2^k!"

    return g, root

def find_primitive_root(max_root_of_unity, max_order, k, p):
    """
    Given the max root of unity (for a prime p s.t. p - 1 = 2^max_order * m, we know the max_order-th root of unity)
    and also a new k < max_order, find the primitive 2^kth root of unity
    """
    
    omega = pow(max_root_of_unity, 2 ** (max_order - k), p)
    assert pow(omega, 2**k, p) == 1, "Not a 2^k-th root of unity!"
    assert pow(omega, 2**(k-1), p) != 1, "Order is not exactly 2^k!"
    
    return omega

p = 2**43 * 3 * 5**2 + 1
m = 3 * 5**2
max_order = 43
odd_part_factors = [3, 5]

g, root = find_2power_root(p, m, max_order, odd_part_factors)

print("Primitive root:", g)
print("2^k-th root of unity:", root)

k = 26
omega = find_primitive_root(root, max_order, k, p)

print(f"Primitive 2^{k}-th root of unity:", omega)

p = 2**44 * 3**2 * 7 + 1
m = 3**2 * 7
max_order = 44
odd_part_factors = [3, 7]

g, root = find_2power_root(p, m, max_order, odd_part_factors)

print("Primitive root:", g)
print("2^k-th root of unity:", root)

k = 26
omega = find_primitive_root(root, max_order, k, p)

print(f"Primitive 2^{k}-th root of unity:", omega)