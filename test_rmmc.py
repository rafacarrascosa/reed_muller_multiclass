import pytest
import numpy
from reed_muller_multiclass import reed_muller


def test_reed_muller_smallest():
    gm = reed_muller(1, 1)
    assert (gm == numpy.array([
        [1, 1],
        [0, 1]
    ])).all()


def test_reed_muller_invalid_values():
    with pytest.raises(ValueError):
        reed_muller(0, 1)
        reed_muller(0, 0)
        reed_muller(-2, -1)
        reed_muller(10, 9)
