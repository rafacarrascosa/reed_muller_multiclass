from itertools import product
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


def reed_muller_trivial(m):
    """
    r and m are equal
    """
    assert m >= 1
    if m == 1:
        return numpy.array([
            [1, 1],
            [0, 1]
        ])
    mm = m - 1
    smaller = reed_muller_trivial(mm)
    return _build_square(
        smaller,
        smaller,
        numpy.zeros((2 ** mm, 2 ** mm), dtype=int),
        smaller
    )


def reed_muller(r, m):
    assert 1 <= r <= m
    if r == m:
        return reed_muller_trivial(m)
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


def _yield_bin(x, n):
    while n:
        yield x % 2
        x = x >> 1
        n -= 1


def _bin(x, n):
    result = list(_yield_bin(x, n))
    result.reverse()
    return result


def printbin(m):
    if len(m.shape) == 1:
        m = [m]
    s = "\n".join("".join(u"\u25A0" if x else u"\u25A1" for x in row) for row in m)
    print(s)


class ReedMuller:
    def __init__(self, r, m, limit=None):
        gm = reed_muller(r, m)
        self.k, self.n = gm.shape
        self.d = 2 ** (m - r)
        codewords = []
        for i, msg in enumerate(product((0, 1), repeat=self.k)):
            if i == limit:
                break
            msg = numpy.array([x for x in msg])
            codeword = msg.dot(gm) % 2
            codewords.append(codeword)
        self.codewords = numpy.array(codewords)
        self.decodewords = (self.codewords * 2 - 1).T

    def encode(self, msg):
        if isinstance(msg, str) or isinstance(msg, bytes):
            msg = int(msg, 2)
        if isinstance(msg, list):
            x = 0
            for bit in msg:
                x += bit
                x = x << 1
            x = x >> 1
            msg = x
        if msg >= len(self.codewords) or msg < 0:
            raise ValueError("Message out of range")
        return self.codewords[msg]

    def decode(self, block):
        if isinstance(block, str) or isinstance(block, bytes):
            block = [int(x) for x in block]
        for x in block:
            assert x in (0, 1)
        block = numpy.array(block) * 2 - 1
        scores = block.dot(self.decodewords)
        i = scores.argmax()
        if scores[i] <= self.n - self.d:
            raise ValueError("Decoding error, could not recover message")
        return i  # _bin(i, self.k)

    def soft_decode(self, block):
        for x in block:
            assert 0 <= x <= 1  # Message elements are probabilities
        p = numpy.array(block)
        scores = self.codewords * p + (1 - self.codewords) * (1 - p)
        scores = numpy.log(scores)
        i = scores.sum(axis=1).argmax()
        return i
