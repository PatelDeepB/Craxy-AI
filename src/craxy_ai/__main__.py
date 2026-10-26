"""Standalone module entrypoint for Craxy-AI.

Allows running the package directly via:
    python -m craxy_ai [options]
"""

import sys

from src.craxy_ai.cli import main


if __name__ == "__main__":
    sys.exit(main())
