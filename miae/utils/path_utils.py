import sys
import os

THIRD_PARTY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../THIRD_PARTY_CODE"))

def add_third_party_to_path():
    if THIRD_PARTY_DIR not in sys.path:
        sys.path.insert(0, THIRD_PARTY_DIR)
