#include "crt.h"
#include "config.h"
#include <cstdio>
#include <vector>

#define GREEN "\033[1;32m"
#define RED   "\033[1;31m"
#define RESET "\033[0m"

using namespace std;

// ------------------ Example / test ------------------

int main() {
    // Example 50-bit-ish primes (same as before)
    // TestDataTypeUint p1 = ( (TestDataTypeUint)1 << 43 ) * 3 * 25 + 1;  // 2^43 * 3 * 5^2 + 1
    // TestDataTypeUint p2 = ( (TestDataTypeUint)1 << 44 ) * 9 * 7 + 1;   // 2^44 * 3^2 * 7 + 1

    TestDataTypeUint p1 = 2013265921;
    TestDataTypeUint p2 = 1811939329;
    TestDataTypeUint p3 = 469762049;

    CRT2Params params_2(p1, p2);

    unsigned __int128 x_true = 12345678901234567890ULL; // example
    x_true %= ((unsigned __int128)p1 * (unsigned __int128)p2); // ensure x_true < p1*p2

    TestDataTypeUint x_mod_p1 = (TestDataTypeUint)(x_true % p1);
    TestDataTypeUint x_mod_p2 = (TestDataTypeUint)(x_true % p2);

    // Test 2-prime version
    unsigned __int128 x_recovered_2 = crt_combine_2(params_2, x_mod_p1, x_mod_p2);

    printf("2-prime CRT:\n");
    printf("x_true      = ");
    print_u128(x_true);
    printf("\n");
    printf("x_recovered = ");
    print_u128(x_recovered_2);
    printf("\n");
    if (x_true == x_recovered_2)
        printf(GREEN "2-prime CRT PASSED\n\n" RESET);
    else
        printf(RED "2-prime CRT FAILED\n\n" RESET);

    // Now test multi-prime CRT with the same two primes
    vector<TestDataTypeUint> primes = {p1, p2};
    vector<TestDataTypeUint> residues = {x_mod_p1, x_mod_p2};

    unsigned __int128 x_recovered_many = crt_combine_many(primes, residues);

    printf("multi-prime CRT (k=2):\n");
    printf("x_true      = ");
    print_u128(x_true);
    printf("\n");
    printf("x_recovered = ");
    print_u128(x_recovered_many);
    printf("\n");
    if (x_true == x_recovered_many)
        printf(GREEN "3-prime CRT PASSED\n" RESET);
    else
        printf(RED "3-prime CRT FAILED\n" RESET);


    // Three roughly 40–45-bit NTT-friendly primes
    // p1 = ((TestDataTypeUint)1 << 43) * 3 * 25 + 1;
    // p2 = ((TestDataTypeUint)1 << 44) * 9 * 7 + 1;
    // TestDataTypeUint p3 = ((TestDataTypeUint)1 << 40) * 17 + 1;   // arbitrary ~40-bit prime

    // Check combined size fits in 128 bits
    unsigned __int128 P = (unsigned __int128) p1 * p2 * p3;

    printf("\n=== 3-prime CRT test ===\n");
    printf("p1 = "); print_u128(p1); printf("\n");
    printf("p2 = "); print_u128(p2); printf("\n");
    printf("p3 = "); print_u128(p3); printf("\n");
    printf("Product = "); print_u128(P); printf("\n");

    // Generate a random test value < p1*p2*p3
    x_true =
        ((unsigned __int128)123456789123ULL << 64) ^
        (unsigned __int128)987654321ULL;  // just some ~120-bit number

    x_true %= P; // ensure x_true < product

    // Compute residues
    TestDataTypeUint r1 = (TestDataTypeUint)(x_true % p1);
    TestDataTypeUint r2 = (TestDataTypeUint)(x_true % p2);
    TestDataTypeUint r3 = (TestDataTypeUint)(x_true % p3);

    primes = {p1, p2, p3};
    residues = {r1, r2, r3};

    // Recover using multi-prime CRT
    unsigned __int128 x_rec = crt_combine_many(primes, residues);

    printf("x_true      = "); print_u128(x_true); printf("\n");
    printf("x_recovered = "); print_u128(x_rec);  printf("\n");

    if (x_true == x_rec)
        printf(GREEN "3-prime CRT PASSED\n" RESET);
    else
        printf(RED "3-prime CRT FAILED\n" RESET);

    return 0;
}
