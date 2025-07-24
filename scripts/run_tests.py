#!/usr/bin/env python3
"""Test runner with predefined test suites and options."""

import argparse
import subprocess
import sys

# Test suite definitions
TEST_SUITES = {
    "smoke": {
        "markers": "smoke",
        "description": "Critical path tests only",
        "options": ["--maxfail=1", "-x"],
    },
    "fast": {
        "markers": "fast and not network",
        "description": "Fast unit tests (no network)",
        "options": ["--durations=5"],
    },
    "unit": {"markers": "unit", "description": "Unit tests only", "options": []},
    "integration": {
        "markers": "integration",
        "description": "Integration tests",
        "options": ["--durations=10"],
    },
    "harvesters": {
        "markers": "harvester",
        "description": "All harvester tests",
        "options": [],
    },
    "extractors": {
        "markers": "extractor",
        "description": "All extractor tests",
        "options": [],
    },
    "no-mocks": {
        "path": "tests/extraction/test_*_refactored.py tests/processing/test_*_refactored.py",
        "description": "Refactored tests without heavy mocking",
        "options": [],
    },
    "coverage": {
        "markers": "",
        "description": "Full test suite with coverage report",
        "options": ["--cov-report=html", "--cov-report=term"],
    },
    "parallel": {
        "markers": "not flaky",
        "description": "Run tests in parallel",
        "options": ["-n", "auto"],
    },
    "profile": {
        "markers": "not slow",
        "description": "Run with performance profiling",
        "options": ["--profile", "--durations=20"],
    },
}


def run_tests(suite: str, extra_args: list[str], verbose: bool = False) -> int:
    """Run test suite with pytest."""
    cmd = ["pytest"]

    if suite in TEST_SUITES:
        suite_config = TEST_SUITES[suite]

        # Add markers if specified
        if suite_config.get("markers"):
            cmd.extend(["-m", suite_config["markers"]])

        # Add path if specified
        if "path" in suite_config:
            cmd.extend(suite_config["path"].split())

        # Add suite-specific options
        cmd.extend(suite_config.get("options", []))
    else:
        # Run specific test file/directory
        cmd.append(suite)

    # Add extra arguments
    cmd.extend(extra_args)

    # Add verbosity
    if verbose:
        cmd.append("-vv")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    return subprocess.call(cmd)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run test suites for the literature review pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add suite descriptions to help
    suite_help = "Test suite to run:\n"
    for name, config in TEST_SUITES.items():
        suite_help += f"  {name:12} - {config['description']}\n"

    parser.add_argument("suite", nargs="?", default="fast", help=suite_help)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--list", action="store_true", help="List available test suites"
    )
    parser.add_argument(
        "--no-cov", action="store_true", help="Disable coverage reporting"
    )
    parser.add_argument(
        "--pdb", action="store_true", help="Drop into debugger on failures"
    )
    parser.add_argument("--lf", action="store_true", help="Run last failed tests only")
    parser.add_argument("-k", help="Run tests matching expression")

    args, extra = parser.parse_known_args()

    if args.list:
        print("Available test suites:")
        print("-" * 60)
        for name, config in TEST_SUITES.items():
            print(f"{name:12} - {config['description']}")
        return 0

    # Build extra arguments
    if args.no_cov:
        extra.append("--no-cov")
    if args.pdb:
        extra.append("--pdb")
    if args.lf:
        extra.append("--lf")
    if args.k:
        extra.extend(["-k", args.k])

    # Run tests
    result = run_tests(args.suite, extra, args.verbose)

    # Print summary
    print("-" * 60)
    if result == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

        # Suggest next steps
        if not args.lf:
            print(
                "\nTip: Run './scripts/run_tests.py fast --lf' to re-run failed tests"
            )
        if not args.pdb:
            print("Tip: Add --pdb to drop into debugger on failure")

    return result


if __name__ == "__main__":
    sys.exit(main())
