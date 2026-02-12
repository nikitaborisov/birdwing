def find_root_of_unity(prime, logn):
    root = 2
    while True:
        x = pow(root, 1 << (logn - 1), prime)
        # Check: x^2 = 1 but x != 1  → x = -1 mod p
        if x != 1 and pow(x, 2, prime) == 1:
            return root
        root += 1


def main():
    print("\n=== Simple Root of Unity Finder ===\n")

    prime = int(input("Enter prime p = "))
    logn = int(input("Enter logn (find 2^logn-th root) = "))

    root = find_root_of_unity(prime, logn)

    print(f"\nPrimitive 2^{logn}-th root of unity: {root}")

    # Verification
    print("Check root^(2^logn) mod p =", pow(root, 1 << logn, prime))
    print("Check root^(2^(logn-1)) mod p =", pow(root, 1 << (logn - 1), prime))


if __name__ == "__main__":
    main()
