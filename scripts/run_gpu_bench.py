#!/usr/bin/env python3
"""Build and run the GPU full-multiply benchmark (32- and/or 64-bit)."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build(bits: int) -> Path:
    target = f"bench_full_{bits}"
    run(["make", target])
    binary = BUILD_DIR / f"bench_full_multiply_{bits}"
    if not binary.exists():
        raise FileNotFoundError(f"Expected binary not found: {binary}")
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GPU full-multiply benchmark for 32-bit and/or 64-bit builds.",
    )
    parser.add_argument(
        "--limb-bits",
        choices=("32", "64", "both"),
        default="32",
        help="Which E2E build to run (default: 32)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip make; assume binaries already exist",
    )
    parser.add_argument(
        "--csv",
        default="gpu_multiply_bench.csv",
        help="Output CSV path (default: gpu_multiply_bench.csv)",
    )
    parser.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the benchmark binary (e.g. 16-24 --iters 20)",
    )
    args = parser.parse_args()

    if args.bench_args and args.bench_args[0] == "--":
        args.bench_args = args.bench_args[1:]

    bits_list = [32, 64] if args.limb_bits == "both" else [int(args.limb_bits)]
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path

    for i, bits in enumerate(bits_list):
        if not args.no_build:
            binary = build(bits)
        else:
            binary = BUILD_DIR / f"bench_full_multiply_{bits}"
            if not binary.exists():
                parser.error(f"binary not found: {binary} (run without --no-build)")

        cmd = [str(binary), "--csv", str(csv_path)]
        if i > 0:
            cmd.append("--append")
        cmd.extend(args.bench_args)

        print(f"\n=== {bits}-bit benchmark ===")
        run(cmd)

    print(f"\nResults in {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
