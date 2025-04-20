#!/usr/bin/env python3
import sys
import os

# Backward-compatible wrapper entry point for legacy users
# Inserts src directory into path and delegates to the new CLI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whycast_transcribe.cli import main

if __name__ == '__main__':
    # Pass through all command-line arguments
    sys.exit(main())
