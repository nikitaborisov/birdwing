BASE = 1 << 32
MASK = BASE - 1

def carry_propagate_from_crt(coeffs):
    """
    coeffs: list of integers c_k after CRT (each < 2^89)
    returns: list of 32-bit limbs in base 2^32
    """
    out = []
    carry = 0

    for c in coeffs:
        t = c + carry
        out.append(t & MASK)    # t % BASE
        carry = t >> 32         # t // BASE

    while carry:
        out.append(carry & MASK)
        carry >>= 32

    return out
