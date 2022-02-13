from functools import wraps
from typing import Callable, Optional

import numpy as np

__all__ = [
    "Metric",
    "binarize",
    "normalize",
    "Annotator",
]

#: A function that can be called on y_true, y_score and return a floating point result
Metric = Callable[[np.array, np.array], float]


def binarize(metric):
    """
    Binarize the predictions for a ground-truth - prediction vector pair.

    Args:
        metric (function): The metric function which needs a binarization pre-processing step.
    Returns:
        metric_wrapper (function): The function which wraps the metric and binarizes the probability scores.
    """

    @wraps(metric)
    def metric_wrapper(*args, **kwargs):
        y_true = args[0]
        y_score = args[1]
        y_score[:] = argsort_indicator(y_score=y_score, y_true=y_true)
        score = metric(*args, **kwargs)
        return score

    return metric_wrapper


def argsort_indicator(*, y_score: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Construct binary indicators from a list of scores.

    If there are $n$ positively labeled entries in ``y_true``, this function
    assigns the top $n$ highest scores in ``y_score`` as positive and remainder
    as negative.

    :param y_score:
        A 1-D array of the score values
    :param y_true:
        A 1-D array of binary values (1 and 0)
    :return:
        A 1-D array of indicator values

    .. seealso::

        This implementation was inspired by
        https://github.com/xptree/NetMF/blob/77286b826c4af149055237cef65e2a500e15631a/predict.py#L25-L33
    """
    number_pos = np.sum(y_true, dtype=int)
    y_sort = np.flip(np.argsort(y_score))
    y_pred = np.zeros_like(y_true, dtype=int)
    y_pred[y_sort[np.arange(number_pos)]] = 1
    return y_pred


def normalize(metric):
    """
    Normalize the predictions for a ground-truth - prediction vector pair.

    Args:
        metric (function): The metric function which needs a normalization pre-processing step.
    Returns:
        metric_wrapper (function): The function which wraps the metric and normalizes predictions.
    """

    @wraps(metric)
    def metric_wrapper(*args, **kwargs):
        y_true = args[0]
        y_score = args[1]
        y_mean = np.mean(y_true)
        y_std = np.std(y_true)
        y_true[:] = (y_true - y_mean) / y_std
        y_score[:] = (y_score - y_mean) / y_std
        score = metric(*args, **kwargs)
        return score

    return metric_wrapper


class Annotator:
    """A class to wrap annotations that generates a registry."""

    def __init__(self):
        self.funcs = {}

    def __iter__(self):
        return iter(self.funcs.values())

    def annotate(
        self,
        *,
        lower: float,
        upper: float,
        higher_is_better: bool,
        link: str,
        description: str,
        name: Optional[str] = None,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
        binarize: bool = False,
        duplicate_of: Optional[Metric] = None,
    ):
        """Annotate a function."""

        def _wrapper(func):
            self.funcs[func.__name__] = func
            func.name = name or func.__name__.replace("_", " ").title()
            func.lower = lower
            func.lower_inclusive = lower_inclusive
            func.upper = upper
            func.upper_inclusive = upper_inclusive
            func.higher_is_better = higher_is_better
            func.link = link
            func.description = description
            func.binarize = binarize
            func.duplicate_of = duplicate_of
            return func

        return _wrapper

    def duplicate(self, other, *, name: Optional[str] = None, binarize: Optional[bool] = None):
        """Annotate a function as a duplicate."""
        return self.annotate(
            name=name,
            lower=other.lower,
            lower_inclusive=other.lower_inclusive,
            upper=other.upper,
            upper_inclusive=other.upper_inclusive,
            link=other.link,
            description=other.description,
            duplicate_of=other,
            higher_is_better=other.higher_is_better,
            # need to be able to override for sklearn functions
            binarize=binarize if binarize is not None else other.binarize,
        )
