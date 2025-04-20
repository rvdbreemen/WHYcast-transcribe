import sys
import os

def pytest_configure(config):
    # Ensure src directory is on sys.path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(root_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
