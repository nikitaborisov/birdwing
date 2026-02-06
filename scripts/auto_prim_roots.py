from math import gcd

def factorize(n):
    factors = []
    d = 3
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 2
    if n > 1:
        factors.append(n)
    return sorted(set(factors))


def extract_k_m(p):
    x = p - 1
    k = 0
    while x % 2 == 0:
        x //= 2
        k += 1
    return k, x


def find_2power_root(p, m, k, odd_part_factors):
    all_factors = [2] + odd_part_factors

    def is_primitive_root(g):
        for q in all_factors:
            if pow(g, (p - 1)//q, p) == 1:
                return False
        return True

    for g in range(2, p):
        if is_primitive_root(g):
            break
    else:
        raise ValueError("No primitive root found.")

    root = pow(g, m, p)

    assert pow(root, 2**k, p) == 1
    assert pow(root, 2**(k-1), p) != 1

    return g, root


def find_primitive_root(max_root_of_unity, max_order, k, p):
    omega = pow(max_root_of_unity, 2 ** (max_order - k), p)

    assert pow(omega, 2**k, p) == 1
    assert pow(omega, 2**(k-1), p) != 1

    return omega


def find_2nth_root_given_omega(max_root_of_unity, max_order, k, p):
    assert k + 1 <= max_order

    psi = pow(max_root_of_unity, 2 ** (max_order - (k + 1)), p)

    assert pow(psi, 2**(k+1), p) == 1
    assert pow(psi, 2**k, p) != 1

    return psi


def main():
    print("\n=== Auto Root of Unity Finder ===\n")

    p = int(input("Enter prime p = "))
    k_small = int(input("Enter desired k (for 2^k-th root): "))

    k, m = extract_k_m(p)
    odd_part_factors = factorize(m)

    print("\nDetected structure:")
    print(f"p - 1 = 2^{k} * {m}")
    print(f"Odd part prime factors = {odd_part_factors}\n")

    g, root = find_2power_root(p, m, k, odd_part_factors)

    omega = find_primitive_root(root, k, k_small, p)
    psi = find_2nth_root_given_omega(root, k, k_small, p)

    print("Primitive root g:", g)
    print(f"Max 2^{k}-th root of unity:", root)
    print(f"Primitive 2^{k_small}-th root of unity:", omega)
    print(f"Primitive 2^{k_small+1}-th root of unity:", psi)

    assert pow(psi, 2, p) == omega
    print("\nVerification passed: psi^2 = omega")


if __name__ == "__main__":
    main()
