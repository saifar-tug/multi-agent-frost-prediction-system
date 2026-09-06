# src/test/run_all_tests.py

import pytest


def main():
    return pytest.main(["src/test", "-v"])


if __name__ == "__main__":
    raise SystemExit(main())
