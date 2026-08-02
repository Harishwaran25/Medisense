"""Allow `python -m medisense`."""
import sys

from medisense.app import main

if __name__ == "__main__":
    sys.exit(main())