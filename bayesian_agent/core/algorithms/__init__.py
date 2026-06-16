"""Bayesian belief update algorithms."""

from bayesian_agent.core.algorithms.beta_bernoulli import BetaBernoulliState
from bayesian_agent.core.algorithms.categorical_bayes import CategoricalBayesState, NaiveBayesState, features_from_event

CATEGORICAL_BAYES = "categorical_bayes"
NAIVE_BAYES_ALIAS = "naive_bayes"
BETA_BERNOULLI = "beta_bernoulli"
FREQUENTIST = "frequentist"
DEFAULT_ALGORITHM = CATEGORICAL_BAYES
SUPPORTED_ALGORITHMS = (CATEGORICAL_BAYES, NAIVE_BAYES_ALIAS, BETA_BERNOULLI, FREQUENTIST)


def normalize_algorithm(algorithm: str = None) -> str:
    if algorithm in {CATEGORICAL_BAYES, NAIVE_BAYES_ALIAS}:
        return CATEGORICAL_BAYES
    if algorithm == BETA_BERNOULLI:
        return BETA_BERNOULLI
    if algorithm == FREQUENTIST:
        return FREQUENTIST
    return DEFAULT_ALGORITHM


def is_categorical_bayes(algorithm: str = None) -> bool:
    return normalize_algorithm(algorithm) == CATEGORICAL_BAYES


def is_frequentist(algorithm: str = None) -> bool:
    return normalize_algorithm(algorithm) == FREQUENTIST

__all__ = [
    "BETA_BERNOULLI",
    "BetaBernoulliState",
    "CATEGORICAL_BAYES",
    "CategoricalBayesState",
    "DEFAULT_ALGORITHM",
    "FREQUENTIST",
    "NAIVE_BAYES_ALIAS",
    "NaiveBayesState",
    "SUPPORTED_ALGORITHMS",
    "features_from_event",
    "is_categorical_bayes",
    "is_frequentist",
    "normalize_algorithm",
]
