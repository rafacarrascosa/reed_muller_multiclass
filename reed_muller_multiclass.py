import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args


def _build_square(A, B, C, D):
    """Build a matrix from submatrices
    A B
    C D
    """
    return np.vstack((
        np.hstack((A, B)),
        np.hstack((C, D))
    ))


def _reed_muller_trivial(m):
    """
    r and m are equal
    """
    if m < 1:
        raise ValueError("m cannot be less than 1")
    if m == 1:
        return np.array([
            [1, 1],
            [0, 1]
        ])
    mm = m - 1
    smaller = _reed_muller_trivial(mm)
    return _build_square(
        smaller,
        smaller,
        np.zeros((2 ** mm, 2 ** mm), dtype=int),
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
            np.zeros((1, 2 ** mm), dtype=int),
            np.ones((1, 2 ** mm), dtype=int),
        )
    top = reed_muller(r, mm)
    bot = reed_muller(r - 1, mm)
    return _build_square(
        top,
        top,
        np.zeros((bot.shape[0], 2 ** mm), dtype=int),
        bot
    )


def _binary_space_matrix(n, bits):
    result = np.unpackbits(np.arange(n, dtype=">u4").view("uint8"))
    return result.reshape((n, -1))[:, -bits:]


class ReedMullerCodec:
    def __init__(self, r, m, limit=None):
        gm = reed_muller(r, m)
        k, _ = gm.shape
        if limit is None:
            limit = 2 ** k
        codewords = _binary_space_matrix(limit, k)
        codewords = codewords.dot(gm.astype("uint8"))
        codewords %= 2
        # Remove columns with all zeros or all ones
        codewords = codewords[:, codewords.any(axis=0) & (~codewords).any(axis=0)]
        self.codewords = codewords

    def encode(self, i):
        if i >= len(self.codewords) or i < 0:
            raise ValueError("Message out of range")
        return self.codewords[i]

    def decode_log_proba(self, block):
        p = np.array(block, dtype=float)
        if (p < 0).any() or (p > 1).any():
            raise ValueError("block must be numbers between 0 and 1")
        scores = self.codewords * p + (1 - self.codewords) * (1 - p)
        scores = np.log(scores + 1e-11)
        return scores.sum(axis=1)

    def decode(self, block):
        lp = self.decode_log_proba(block)
        i = lp.argmax()
        return i


class ReedMullerMultiClass(ClassifierMixin):
    @_deprecate_positional_args
    def __init__(self, estimator, *, n_jobs=None):
        # FIXME: Check estimator has predict_proba method
        self.multi_output = MultiOutputClassifier(estimator, n_jobs=n_jobs)

    def fit(self, X, Y, sample_weight=None, **fit_params):
        self.classes_ = np.unique(Y)
        n_classes = len(self.classes_)
        if n_classes < 3:
            pass  # Fixme: Raise warning? Exception?
        self.class_to_index = dict((c, i) for i, c in enumerate(self.classes_))
        # Choose Reed Muller parameters in function of n_classes
        r, m = self._rm_policy(n_classes)
        self.rm = ReedMullerCodec(r, m, limit=n_classes)
        Y = self.encode_labels(Y)
        self.multi_output.fit(X, Y, sample_weight, **fit_params)

    def decision_function(self, X):
        check_is_fitted(self)

        Y = self.multi_output.predict(X)
        return self.decode_log_proba(Y)

    def predict_proba(self, X):
        check_is_fitted(self)

        Y = self.multi_output.predict(X)
        Y = np.exp(self.decode_log_proba(Y))
        Y = Y / Y.sum(axis=1)
        return Y

    def predict(self, X):
        check_is_fitted(self)

        Y = self.multi_output.predict(X)
        Y = self.decode_log_proba(Y).argmax(axis=1)
        return np.array([self.classes_[i] for i in Y])

    def encode_labels(self, Y):
        Y = (self.class_to_index[c] for c in Y)  # Encode classes as integers
        Y = np.array([self.rm.encode(i) for i in Y])  # Encode integers as an RM ECC
        return Y

    def decode_log_proba(self, Y):
        Z = np.empty((len(Y), len(self.classes_)))
        for i, bits in enumerate(Y):
            Z[i] = self.rm.decode_log_proba(bits)
        return Z

    @staticmethod
    def _rm_options():
        m = 0
        # For a small number of classes, give order 1 RM codes
        for m in range(1, 4):
            yield 1, m, m + 1
        # For a larger number of classes, give order 2 RM codes
        m = 3
        while True:
            yield 2, m, int((m * (m - 1)) / 2) + m + 1
            m += 1

    @classmethod
    def _rm_policy(cls, n_classes):
        for r, m, rows in cls._rm_options():
            if 2 ** rows >= n_classes:
                return r, m
