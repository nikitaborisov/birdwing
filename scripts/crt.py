import random
import math
import time

def egcd(a, b):
    """
    Extended Euclidean algorithm.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b)
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y

    return old_r, old_x, old_y

def modinv(a, m):
    """
    Compute the modular inverse of a modulo m:
        a * a^{-1} ≡ 1 (mod m)
    Raises ValueError if inverse does not exist.
    """
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}, gcd = {g}")
    return x % m

def crt(remainders, moduli):
    """
    Chinese Remainder Theorem for pairwise coprime moduli.

    Given:
        x ≡ remainders[i] (mod moduli[i])  for all i
    returns:
        (x, M) where M = product(moduli)
    such that 0 <= x < M and x is the unique solution modulo M.
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")

    # Optionally: you can assert pairwise coprime moduli here.
    # For understanding, we'll trust the caller or check lightly.
    k = len(moduli)
    M = 1
    for m in moduli:
        M *= m

    x = 0
    for r_i, m_i in zip(remainders, moduli):
        M_i = M // m_i
        inv_M_i = modinv(M_i % m_i, m_i)   # (M / m_i)^(-1) mod m_i
        term = r_i * M_i * inv_M_i
        x += term

    x %= M
    return x, M

# remainders = [2, 3]
# moduli     = [5, 7]

# x, M = crt(remainders, moduli)
# print("x =", x, "mod", M)

# for r, m in zip(remainders, moduli):
#     print(f"{x} % {m} = {x % m} (should be {r})")

def test_basic_cases():
    tests = [
        # (remainders, moduli, expected)
        ([2, 3], [3, 5], 8),
        ([1, 0], [2, 3], 3),
        ([0, 1], [2, 5], 6),
        ([4, 3], [11, 7], 59),
    ]

    for rems, mods, expected in tests:
        x, M = crt(rems, mods)
        assert x == expected, f"Expected {expected}, got {x}"
    print("Basic tests passed!")

def test_random_cases(num_tests=1000):
    for _ in range(num_tests):

        # pick number of moduli
        k = random.choice([2, 3])

        # generate random odd 32-bit moduli
        moduli = []
        while len(moduli) < k:
            m = random.randrange(3, 2**32, 2)  # always odd
            if all(math.gcd(m, other) == 1 for other in moduli):
                moduli.append(m)

        # just generate x in 64-bit range
        x_true = random.getrandbits(64)

        # compute remainders
        rems = [x_true % m for m in moduli]

        # run CRT
        x_rec, M = crt(rems, moduli)

        # since x_rec is modulo M, compare via modulo
        if x_rec % M != x_true % M:
            print("FAIL")
            print("moduli:", moduli)
            print("rems:", rems)
            print("expected mod M:", x_true % M)
            print("got:", x_rec)
            return

    print(f"Randomized test passed for {num_tests} cases!")
    
def test_edge_cases():
    # smallest valid moduli
    rems = [1, 0]
    mods = [2, 3]
    assert crt(rems, mods)[0] == 3

    # large 32-bit prime moduli
    p1 = 4294967291  # largest 32-bit prime
    p2 = 4294967279

    for x in [0, 1, p1-1, p1*p2-1]:
        r1 = x % p1
        r2 = x % p2
        xr, _ = crt([r1, r2], [p1, p2])
        assert xr == x

    print("Edge case tests passed!")

def benchmark_crt(trials=50000):
    print("Benchmarking CRT...")

    # two 32-bit moduli
    m1 = 4294967291
    m2 = 4294967279
    M2 = m1 * m2

    # three 32-bit moduli
    m3 = 4294967231
    M3 = M2 * m3

    start = time.time()
    for _ in range(trials):
        x = random.randrange(0, M2)
        r1, r2 = x % m1, x % m2
        crt([r1, r2], [m1, m2])
    t2 = time.time() - start

    start = time.time()
    for _ in range(trials):
        x = random.randrange(0, M3)
        r1, r2, r3 = x % m1, x % m2, x % m3
        crt([r1, r2, r3], [m1, m2, m3])
    t3 = time.time() - start

    print(f"2-mod CRT: {t2:.3f} sec for {trials} trials, {t2/trials*1e6:.1f} µs per test")
    print(f"3-mod CRT: {t3:.3f} sec for {trials} trials, {t3/trials*1e6:.1f} µs per test")

test_basic_cases()
test_random_cases()
test_edge_cases()
benchmark_crt()
