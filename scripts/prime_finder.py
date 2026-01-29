from gmpy2 import mpz, is_prime

def find_ntt_primes(bits: int, log_size: int, num_primes: int) -> list[mpz]:
    k = 2**(bits-log_size) - 1
    base = 2**log_size
    primes = []
    while len(primes) < num_primes:
        candidate = base*k + 1
        if is_prime(mpz(candidate)):
            primes.append((k,candidate))
        
        k -= 1
        if k < 0:
            print(f"No {num_primes} primes found for bits {bits} and log_size {log_size}")
            break

    return primes

if __name__ == "__main__":
    import sys
    (bits, log_size, num_primes) = map(int, sys.argv[1:])
    primes = find_ntt_primes(bits, log_size, num_primes)
    for k, p in primes:
        print(f"{k}*2**{log_size} + 1 = {p}")
        
# Example usage:
"""
python prime_finder.py 31 26 3
30*2**26 + 1 = 2013265921
27*2**26 + 1 = 1811939329
7*2**26 + 1 = 469762049
"""