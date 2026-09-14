"""Absolute paths into the repository, so a script runs from any directory.

Every .dat, .csv and .json this work reads or produces lives in data/.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def data(*parts):
    """A path inside data/."""
    return os.path.join(DATA, *parts)
