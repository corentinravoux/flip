#!/usr/bin/env python3
import sys
import traceback

def main() -> int:
    try:
        from flip.data.load_data_test import (
            generate_and_save_covariance_test_reference_values,
        )
        print("Generating covariance test reference values...")
        ref = generate_and_save_covariance_test_reference_values()
        print(f"OK: generated {len(ref)} entries and wrote JSON files.")
        return 0
    except Exception:
        print("ERROR while generating reference values:")
        print("".join(traceback.format_exc()))
        return 1


if __name__ == "__main__":
    sys.exit(main())
