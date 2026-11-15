"""Command-line entry point; heavy dependencies load only when running."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Classify negative workplace reviews and join company statistics.")
    parser.add_argument("--glassdoor-dir", type=Path, default=Path("data/glassdoor reviews dataset"))
    parser.add_argument("--indeed-dir", type=Path, default=Path("data/indeed job listings dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    try:
        from .pipeline import run
        run(args.glassdoor_dir, args.indeed_dir, args.output_dir)
    except ImportError as exc:
        parser.exit(1, f"Missing dependency: {exc}. Install the project; Excel/Parquet inputs need the [formats] extra.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()
