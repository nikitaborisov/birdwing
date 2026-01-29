import secrets
import argparse
import struct

# Example usage:
# python rng.py --limbs 64 --output input.bin
# This generates two random numbers, each with 64 32-bit limbs, and writes them to input.bin in binary format.

def generate_limb_array(num_limbs):
    """Generate an array of random 32-bit integers (limbs)."""
    return [secrets.randbits(32) for _ in range(num_limbs)]

def write_limb_array(filename, limbs_a, limbs_b):
    """Write two limb arrays to a binary file: a followed by b."""
    with open(filename, "wb") as f:
        for limb in limbs_a:
            f.write(struct.pack("<I", limb))  # little-endian 32-bit
        for limb in limbs_b:
            f.write(struct.pack("<I", limb))

def main():
    parser = argparse.ArgumentParser(description="Generate two random numbers as 32-bit limb arrays")
    parser.add_argument("--limbs", type=int, default=32, help="Number of 32-bit limbs per number")
    parser.add_argument("--output", type=str, default="bigints.bin", help="Output binary file")
    args = parser.parse_args()

    limbs_a = generate_limb_array(args.limbs)
    limbs_b = generate_limb_array(args.limbs)

    write_limb_array(args.output, limbs_a, limbs_b)
    print(f"Two random numbers ({args.limbs} limbs each) written to {args.output} in binary format.")

if __name__ == "__main__":
    main()
