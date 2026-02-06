def mod_add(a, b, p):
    return (a + b) % p

def mod_sub(a, b, p):
    return (a - b) % p

def mod_mul(a, b, p):
    return (a * b) % p

def mod_inv(a, p):
    return pow(a, -1, p)

def bit_reverse_copy(a):
    n = len(a)
    j = 0
    res = a.copy()
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            res[i], res[j] = res[j], res[i]
    return res

def ntt(a, omega, p):
    n = len(a)
    a = bit_reverse_copy(a)

    length = 2
    while length <= n:
        wlen = pow(omega, n // length, p)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = mod_mul(a[i + j + length//2], w, p)
                a[i + j] = mod_add(u, v, p)
                a[i + j + length//2] = mod_sub(u, v, p)
                w = mod_mul(w, wlen, p)
        length *= 2
    return a

def intt(a, omega_inv, p):
    n = len(a)
    a = bit_reverse_copy(a)

    length = 2
    while length <= n:
        wlen = pow(omega_inv, n // length, p)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length//2]
                a[i + j] = mod_add(u, mod_mul(v, w, p), p)
                a[i + j + length//2] = mod_sub(u, mod_mul(v, w, p), p)
                w = mod_mul(w, wlen, p)
        length *= 2

    n_inv = mod_inv(n, p)
    return [mod_mul(x, n_inv, p) for x in a]

def convolution_mod(a, b, p, omega):
    n = 1
    while n < len(a) + len(b):
        n *= 2

    A = a + [0]*(n - len(a))
    B = b + [0]*(n - len(b))

    print(f"\n--- MOD {p} ---")
    print("Padded A:", A)
    print("Padded B:", B)

    A_ntt = ntt(A, omega, p)
    B_ntt = ntt(B, omega, p)
    print("Forward NTT A:", A_ntt)
    print("Forward NTT B:", B_ntt)

    C_ntt = [(x*y) % p for x,y in zip(A_ntt, B_ntt)]
    print("Pointwise:", C_ntt)

    omega_inv = mod_inv(omega, p)
    c = intt(C_ntt, omega_inv, p)
    print("Inverse NTT:", c)

    return c

def crt3(r0, r1, r2, p0, p1, p2):
    M = p0 * p1 * p2
    m0 = M // p0
    m1 = M // p1
    m2 = M // p2

    inv0 = mod_inv(m0, p0)
    inv1 = mod_inv(m1, p1)
    inv2 = mod_inv(m2, p2)

    return (r0*m0*inv0 + r1*m1*inv1 + r2*m2*inv2) % M

A = [1,2,3,4]
B = [5,6,7,8]

# Your primes
p0 = 2013265921
p1 = 1811939329
p2 = 469762049

# Compute correct omega for N=8 using your root-finding script
# omega0, omega1, omega2 must be primitive 8th roots of unity

omega0 = 1592366214
omega1 = 1452317833
omega2 = 129701348

c0 = convolution_mod(A, B, p0, omega0)
c1 = convolution_mod(A, B, p1, omega1)
c2 = convolution_mod(A, B, p2, omega2)

print("\n--- CRT Reconstruction ---")
result = [crt3(c0[i], c1[i], c2[i], p0, p1, p2) for i in range(len(c0))]
print("Final result:", result)
