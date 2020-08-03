# Reed Muller multiclass

This is small library to turn binary classifiers into multi class classifiers using a [Reed-Muller error correction code](https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code).

It is compatible with the scikit-learn API for multiclass and it is conceptually similar to  [OutputCodeClassifier](https://scikit-learn.org/stable/modules/multiclass.html#error-correcting-output-codes).

I've tried it on a few datasets and it is useful in the same scenarios as hierarchical softmax. I haven't done extensive testing but it could promising for language modelling.

As far as I know, there is no content on Reed Muller for Machine Learning (papers or code), but if you know of any please let me know! (email is in the commits).
