import pytest
import numpy
from reed_muller_multiclass import reed_muller, ReedMuller


def _cm(s):
    s = s.split()
    s.sort(reverse=True)
    return numpy.array([[int(x) for x in xs] for xs in s])


def test_gm_smallest():
    gm = reed_muller(1, 1)
    assert (gm == _cm("11 01")).all()


def test_gm_1_3():
    # according to https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code
    # but shuffling the columns to read left to right.
    correct = """
    11111111
    01010101
    00110011
    00001111
    """
    gm = reed_muller(1, 3)
    assert (gm == _cm(correct)).all()


def test_gm_2_3():
    # according to https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code
    # but shuffling the columns to read left to right.
    correct = """
    11111111
    01010101
    00110011
    00001111
    00010001
    00000101
    00000011
    """
    gm = reed_muller(2, 3)
    assert (gm == _cm(correct)).all()


def test_gm_shape():
    gm = reed_muller(1, 9)
    assert gm.shape == (9 + 1, 2 ** 9)


def test_gm_invalid_values():
    with pytest.raises(ValueError):
        reed_muller(0, 1)
        reed_muller(0, 0)
        reed_muller(-2, -1)
        reed_muller(10, 9)


def test_reed_muller_2_4_back_and_forth():
    rm = ReedMuller(2, 4)
    for i in range(2 ** 11):
        assert i == rm.decode(rm.encode(i))
