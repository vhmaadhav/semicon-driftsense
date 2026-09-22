"""Drift-Sense: Siamese navigation-error recovery for wafer inspection."""
from driftsense.model import DriftSenseNet

# Single source of truth for the version; pyproject.toml reads it from
# here via [tool.setuptools.dynamic]. 0.x because the Python API is not
# stable -- the CSV output contract of register.py is, and it is
# versioned by the submission, not by this number.
__version__ = "0.1.0"

__all__ = ["DriftSenseNet", "__version__"]
