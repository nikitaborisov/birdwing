from math import gcd

# p = 2**44 * 3**2 * 7 + 1
# odd_part = 3**2 * 7
# two_power = 2**44

p = 2**43 * 3 * 5**2 + 1
odd_part = 3 * 5**2
two_power = 2**43

def is_primitive_root(g, p):
    # Check if g is a primitive root modulo p
    factors = [2, 3, 5]  # prime factors of p-1
    for factor in factors:
        if pow(g, (p-1)//factor, p) == 1:
            return False
    return True

# Find a primitive root
for g in range(1, 100):
    if is_primitive_root(g, p):
        break

print("Primitive root modulo p:", g)

# Compute the 2^power-th root of unity
root = pow(g, odd_part, p)
print("Primitive 2^power-th root of unity:", root)

# Verify order
assert pow(root, two_power, p) == 1
assert pow(root, two_power//2, p) != 1
print("Verification passed: order is correct")
