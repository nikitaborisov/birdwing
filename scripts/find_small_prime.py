import sympy as sp

# -------- SETTINGS --------
n = 256          # NTT size (change if you want)
exclude_q = 7681 # your existing modulus
# --------------------------

def find_ntt_prime(n, start_k=1):
    """
    Find prime q = k*(2n) + 1, q != exclude_q
    """
    k = start_k
    while True:
        q = k * (2 * n) + 1
        if q != exclude_q and sp.isprime(q):
            return q
        k += 1


def primitive_root_of_unity(q, n):
    """
    Returns (omega, psi) for modulus q and transform size n
    omega = primitive n-th root
    psi   = primitive 2n-th root (for negacyclic NTT)
    """
    g = sp.primitive_root(q)  # generator of multiplicative group

    omega = pow(g, (q - 1) // n, q)
    psi   = pow(g, (q - 1) // (2 * n), q)

    return omega, psi


def verify_roots(q, n, omega, psi):
    print("Check ω^n mod q =", pow(omega, n, q))
    print("Check ψ^(2n) mod q =", pow(psi, 2*n, q))
    print("Check ψ^n mod q =", pow(psi, n, q))  # should be q-1 (≡ -1)


if __name__ == "__main__":
    q2 = find_ntt_prime(n)
    omega, psi = primitive_root_of_unity(q2, n)

    print(f"Chosen second prime q2 = {q2}")
    print(f"omega (n-th root)      = {omega}")
    print(f"psi   (2n-th root)     = {psi}")
    print("\nVerification:")
    verify_roots(q2, n, omega, psi)
