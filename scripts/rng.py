import secrets
import argparse
import struct

# Example usage:
# python rng.py --limbs 64 --output input.bin
# python rng.py --limbs 64 --limb-bits 64 --output input_64bit.bin
# Generates two random numbers and writes them to input.bin in binary format.


def generate_limb_array(num_limbs, limb_bits):
    """Generate an array of random limbs (32- or 64-bit)."""
    if limb_bits == 64:
        return [secrets.randbits(64) for _ in range(num_limbs)]
    return [secrets.randbits(32) for _ in range(num_limbs)]


def write_limb_array(filename, limbs_a, limbs_b, limb_bits):
    """Write two limb arrays to a binary file: a followed by b."""
    pack = struct.Struct("<Q") if limb_bits == 64 else struct.Struct("<I")
    with open(filename, "wb") as f:
        for limb in limbs_a:
            f.write(pack.pack(limb))
        for limb in limbs_b:
            f.write(pack.pack(limb))


def main():
    parser = argparse.ArgumentParser(
        description="Generate two random big integers as limb arrays"
    )
    parser.add_argument(
        "--limbs", type=int, default=32, help="Number of limbs per number"
    )
    parser.add_argument(
        "--limb-bits",
        type=int,
        choices=(32, 64),
        default=32,
        help="Width of each limb in bits (default: 32)",
    )
    parser.add_argument(
        "--output", type=str, default="bigints.bin", help="Output binary file"
    )
    args = parser.parse_args()

    limbs_a = generate_limb_array(args.limbs, args.limb_bits)
    limbs_b = generate_limb_array(args.limbs, args.limb_bits)

    write_limb_array(args.output, limbs_a, limbs_b, args.limb_bits)
    print(
        f"Two random numbers ({args.limbs} x {args.limb_bits}-bit limbs each) "
        f"written to {args.output}."
    )


if __name__ == "__main__":
    main()
