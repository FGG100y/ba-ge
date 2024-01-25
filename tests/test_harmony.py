"""
"""

from src.cythonpkg.harmony import _calculate_results


def test_harmonic_mean():
    assert _calculate_results([1, 4]) == 1.6
