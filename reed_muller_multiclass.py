import numpy


def _build_square(A, B, C, D):
    """Build a matrix from submatrices
    A B
    C D
    """
    return numpy.vstack((
        numpy.hstack((A, B)),
        numpy.hstack((C, D))
    ))


def _reed_muller_trivial(m):
    """
    r and m are equal
    """
    if m < 1:
        raise ValueError("m cannot be less than 1")
    if m == 1:
        return numpy.array([
            [1, 1],
            [0, 1]
        ])
    mm = m - 1
    smaller = _reed_muller_trivial(mm)
    return _build_square(
        smaller,
        smaller,
        numpy.zeros((2 ** mm, 2 ** mm), dtype=int),
        smaller
    )


def reed_muller(r, m):
    if r < 1:
        raise ValueError("r cannot be less than 1")
    if r > m:
        raise ValueError("m cannot be smaller than r")
    if r == m:
        return _reed_muller_trivial(m)
    mm = m - 1
    if r == 1:
        smaller = reed_muller(r, mm)
        return _build_square(
            smaller,
            smaller,
            numpy.zeros((1, 2 ** mm), dtype=int),
            numpy.ones((1, 2 ** mm), dtype=int),
        )
    top = reed_muller(r, mm)
    bot = reed_muller(r - 1, mm)
    return _build_square(
        top,
        top,
        numpy.zeros((bot.shape[0], 2 ** mm), dtype=int),
        bot
    )


def _binary_space_matrix(n, bits):
    result = numpy.unpackbits(numpy.arange(n, dtype=">u4").view("uint8"))
    return result.reshape((n, -1))[:, -bits:]


class ReedMuller:
    def __init__(self, r, m, limit=None):
        gm = reed_muller(r, m)
        k, _ = gm.shape
        if limit is None:
            limit = 2 ** k
        codewords = _binary_space_matrix(limit, k)
        codewords = codewords.dot(gm.astype("uint8"))
        codewords %= 2
        self.codewords = codewords

    def encode(self, i):
        if i >= len(self.codewords) or i < 0:
            raise ValueError("Message out of range")
        return self.codewords[i]

    def decode(self, block):
        p = numpy.array(block, dtype=float)
        if (p < 0).any() or (p > 1).any():
            raise ValueError("block must be numbers between 0 and 1")
        scores = self.codewords * p + (1 - self.codewords) * (1 - p)
        scores = numpy.log(scores + 1e-11)
        i = scores.sum(axis=1).argmax()
        return i
