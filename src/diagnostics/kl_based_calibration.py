import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.diagnostics.coverage import compute_coverage_scores
from bayesflow.utils.ecdf.ranks import distance_ranks

def estimate_calibration_kl(u_samples, post_samples, n_splits=5):
    """
    Estimate KL(P_{X,U} || P_X ⊗ Uniform) using classifier (logistic regression)
    
    Args:
        u_samples: shape [n]
        post_samples: shape [n, M, d_theta]
        n_splits: number of folds for cross-fitting
        
    Returns:
        kl_estimate: float
    """
    
    #  Obtain scaler summary of x given by sum of variances of posterior samples over dimensions
    var_per_dim = np.var(post_samples, axis=1)         # [n, d_theta]
    z = np.mean(var_per_dim, axis=1)                  # [n]


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    kl_values = []

    for train_idx, test_idx in kf.split(z):
        
        z_train, z_test = z[train_idx], z[test_idx]
        u_train, u_test = u_samples[train_idx], u_samples[test_idx]

        v_train = np.random.uniform(0.0, 1.0, size=len(train_idx))

        X_real = np.stack([z_train, u_train], axis=1)
        X_fake = np.stack([z_train, v_train], axis=1)

        X_clf = np.concatenate([X_real, X_fake], axis=0)
        y_clf = np.concatenate([
            np.ones(len(train_idx)),
            np.zeros(len(train_idx))
        ])

        scaler = StandardScaler()
        X_clf = scaler.fit_transform(X_clf)
        clf = LogisticRegression(max_iter=1000, C=0.1) 
        clf.fit(X_clf, y_clf)
        
        X_test = np.stack([z_test, u_test], axis=1)
        X_test = scaler.transform(X_test)

        D = clf.predict_proba(X_test)[:, 1]

        # Compute kl based calibration based on classifier
        eps = 1e-6
        D = np.clip(D, eps, 1 - eps)
        log_ratio = np.log(D / (1 - D))

        kl_values.append(np.mean(log_ratio))

    # Average over folds
    return float(np.mean(kl_values))

def obtain_u_samples_q(data, post_samples, approximator, M=1000, score_true=None, scores_samples=None):
    if score_true is None or scores_samples is None:
        score_true, scores_samples = compute_coverage_scores(data, post_samples, approximator, M=M)
    score_true = score_true[:, np.newaxis]
    u_samples = (scores_samples <= score_true).mean(axis=1)
    return u_samples


def obtain_u_samples_distance(data, post_samples):
    distance_rank = distance_ranks(post_samples, data, stacked = True)
    return distance_rank.squeeze(axis = -1)


def kl_cal_q(data, post_samples, approximator, M=1000):
    u_samples = obtain_u_samples_q(data, post_samples, approximator, M=M)
    return estimate_calibration_kl(u_samples, post_samples)