"""Make `from src.pipeline import ...` resolvable when the generator suite is
discovered from the repo root (pytest.ini lists generator/tests as well as
tests/). Running from inside generator/ keeps working unchanged.
"""

import os
import sys

GENERATOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GENERATOR_DIR not in sys.path:
    sys.path.insert(0, GENERATOR_DIR)
