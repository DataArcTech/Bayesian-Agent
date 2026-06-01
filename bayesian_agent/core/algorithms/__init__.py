"""Bayesian belief update algorithms."""

from bayesian_agent.core.algorithms.beta_bernoulli import BetaBernoulliState
from bayesian_agent.core.algorithms.categorical_bayes import CategoricalBayesState, NaiveBayesState, features_from_event

CATEGORICAL_BAYES = "categorical_bayes"
NAIVE_BAYES_ALIAS = "naive_bayes"
BETA_BERNOULLI = "beta_bernoulli"
DEFAULT_ALGORITHM = CATEGORICAL_BAYES
SUPPORTED_ALGORITHMS = (CATEGORICAL_BAYES, NAIVE_BAYES_ALIAS, BETA_BERNOULLI)


def normalize_algorithm(algorithm: str = None) -> str:
    if algorithm in {CATEGORICAL_BAYES, NAIVE_BAYES_ALIAS}:
        return CATEGORICAL_BAYES
    if algorithm == BETA_BERNOULLI:
        return BETA_BERNOULLI
    return DEFAULT_ALGORITHM


def is_categorical_bayes(algorithm: str = None) -> bool:
    return normalize_algorithm(algorithm) == CATEGORICAL_BAYES

__all__ = [
    "BETA_BERNOULLI",
    "BetaBernoulliState",
    "CATEGORICAL_BAYES",
    "CategoricalBayesState",
    "DEFAULT_ALGORITHM",
    "NAIVE_BAYES_ALIAS",
    "NaiveBayesState",
    "SUPPORTED_ALGORITHMS",
    "features_from_event",
    "is_categorical_bayes",
    "normalize_algorithm",
]
