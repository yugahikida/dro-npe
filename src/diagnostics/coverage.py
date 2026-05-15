import warnings
import numpy as np

def compute_coverage(data, post_samples, approximator, alpha, M = 1000,
                     score_true=None, scores_samples=None):
    """
    Compute coverage for given alpha value(s).
    If score_true and scores_samples are provided, reuse precomputed scores.
    Otherwise, compute scores on the fly.
    """
    if score_true is None or scores_samples is None:
        score_true, scores_samples = compute_coverage_scores(data, post_samples, approximator, M)
    
    alpha = np.atleast_1d(alpha)
    n_test = data["parameters"].shape[0]
    n_alpha = len(alpha)
    coverage = np.full((n_alpha, n_test), np.nan)
    n_skipped = 0
    for i in range(n_test):
        if not np.isfinite(score_true[i]) or not np.all(np.isfinite(scores_samples[i])):
            n_skipped += 1
            continue
        thresholds = np.quantile(scores_samples[i], alpha)
        coverage[:, i] = (score_true[i] >= thresholds).astype(float)

    if n_skipped > 0:
        warnings.warn(
            f"{n_skipped}/{n_test} test observations had NaN/Inf log-prob scores and were excluded from coverage."
        )

    mean_coverage = np.nanmean(coverage, axis=1)
    return mean_coverage


def compute_coverage_scores(data, post_samples, approximator, M = 1000):
    f"""
    Compute coverage scores for all test samples once.
    Parameters
    ----------
    data: Mapping[str, np.ndarray]
    post_samples: np.ndarray of shape (n_test, n_post_samples, n_parameters)
    approximator: ContinuousApproximator
    M: int
    Returns:
        score_true: array of shape (n_test,) with true scores
        scores_samples: array of shape (n_test, M) with sample scores
    """
    n_test = data["parameters"].shape[0]
    score_true_list = []
    scores_samples_list = []

    for i in range(n_test):
        theta_star = data["parameters"][i]
        x_star = data["observables"][i]
        post_samples_i = post_samples[i, :M, :]
        x_star_batch = x_star[np.newaxis, :]
        data_true = dict(parameters=theta_star[np.newaxis, :], observables=x_star_batch)
        x_star_broadcast = np.repeat(x_star_batch, post_samples_i.shape[0], axis=0)
        data_samples = dict(parameters=post_samples_i, observables=x_star_broadcast)
        score_true = approximator.log_prob(data_true)
        scores_i = approximator.log_prob(data_samples)
        # Extract scalar from score_true if it's an array
        if isinstance(score_true, np.ndarray):
            score_true = score_true[0]
        score_true_list.append(score_true)
        scores_samples_list.append(scores_i)
    
    return np.array(score_true_list), np.array(scores_samples_list)


def absolute_difference_miscoverage(data, post_samples, approximator, alpha, M = 1000, score_true=None, scores_samples=None):

    empirical_coverage = compute_coverage(data, post_samples, approximator, alpha, M = M, score_true=score_true, scores_samples=scores_samples)
    return np.abs(empirical_coverage - 1 + alpha)