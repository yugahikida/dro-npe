import argparse
import itertools
import pickle
from pathlib import Path
import shutil

import os
import time
from bayes_opt import BayesianOptimization

import numpy as np

def _set_keras_backend_for_method(method_name: str) -> None:
    backend_by_method = {
        "cal": "torch",
        "balanced": "jax",
    }
    backend = backend_by_method.get(method_name)
    if backend is None:
        raise ValueError(f"Unknown method_name='{method_name}' for backend selection.")
    # Must be set before importing keras/bayesflow or any custom approximator modules.
    os.environ["KERAS_BACKEND"] = backend


def _result_dir(simulator_name, method_name, n, seed, comment=""):
    safe_comment = comment.replace(" ", "_") if comment else ""
    comment_suffix = f"_comment_{safe_comment}" if safe_comment else ""
    return Path(
        f"results/{simulator_name}/{method_name}/"
        f"n_{n}_seed_{seed}{comment_suffix}"
    )


def load_validation_data_from_npe(simulator_name, n, seed, val_rate=0.1):
    """Load validation split from shared NPE training data, matching train.py split logic."""
    seed_path = Path(f"results/{simulator_name}/npe/n_{n}_seed_{seed}/training_data.pkl")
    if not seed_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {seed_path}. "
            "Run train.py first to generate npe training data."
        )
    with open(seed_path, "rb") as f:
        full_data = pickle.load(f)

    train_size = int(n * (1 - val_rate))
    return {k: v[train_size:] for k, v in full_data.items()}


def train_and_evaluate(simulator_name, method_name, reg_coef, n, seed, batch_size, epochs, comment="", metric = "kl_cal"):
    from train import train
    from src.diagnostics.kl_based_calibration import (
        estimate_calibration_kl,
        obtain_u_samples_distance,
        obtain_u_samples_q,
    )
    from src.diagnostics.coverage import compute_coverage

    comment = comment + "regcoef_" + str(reg_coef)
    # Use the in-memory approximator returned by train() rather than reloading
    # from disk: on shared filesystems (e.g. /scratch on a cluster) the
    # freshly written model.keras zip can be partially flushed, causing
    # zipfile.BadZipFile: Bad CRC-32 on read.
    _, approximator = train(simulator_name, method_name, 0, n, seed, batch_size, epochs, val_rate = 0.1, comment = comment, reg_coef = reg_coef)
    validation_data = load_validation_data_from_npe(simulator_name, n, seed, val_rate=0.1)

    if metric == "kl_cal":
        post_samples = approximator.sample(
            conditions=validation_data, num_samples=1000
        )["parameters"]
        u_samples = obtain_u_samples_q(validation_data, post_samples, approximator, M=1000)
        kl_cal = estimate_calibration_kl(u_samples, post_samples)

        return kl_cal

    elif metric == "nlpd":
        nlpd = - np.asarray(approximator.log_prob(validation_data)).ravel()
        return float(np.mean(nlpd))

    elif metric == "kl_cal_distance":
        post_samples = approximator.sample(
            conditions=validation_data, num_samples=1000
        )["parameters"]
        u_samples = obtain_u_samples_distance(validation_data["parameters"], post_samples)
        kl_cal = estimate_calibration_kl(u_samples, post_samples)
        return kl_cal

    elif metric == "coverage_error":
        alpha = 0.05
        post_samples = approximator.sample(
            conditions=validation_data, num_samples=1000
        )["parameters"]
        coverage = compute_coverage(validation_data, post_samples, approximator, alpha=alpha, M=1000)
        coverage_error = np.abs(coverage - 1 + alpha)
        return coverage_error.squeeze()

def train_and_optimise_hyperparams(
    simulator_name,
    method_name,
    n,
    seed,
    batch_size,
    epochs,
    comment="",
    coef_min=1e-2,
    coef_max=10,
    maxiter=10,
    delta=0.0,
    optimiser = "BO", # "BO" or "bounded"
    metric = "kl_cal",
):
    from train import train

    comment_suffix = f"_{comment}" if comment else ""

    result_dir = Path(
        f"results/{simulator_name}/{method_name}/"
        f"regcoef_tune_n_{n}_seed_{seed}_maxiter_{maxiter}_delta_{delta}_metric_{metric}{comment_suffix}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    evaluated_runs = []

    if optimiser == "BO":
        def objective_log(log_reg_coef):
            reg_coef = float(round(np.exp(log_reg_coef), 5))
            metric_value = train_and_evaluate(
                simulator_name,
                method_name,
                reg_coef,
                n,
                seed,
                batch_size,
                epochs,
                comment,
                metric,
            )
            comment_trial = comment + "regcoef_" + str(reg_coef)
            run_dir = _result_dir(simulator_name, method_name, n, seed, comment_trial)
            evaluated_runs.append((reg_coef, float(metric_value), run_dir))
            return -float(metric_value)

        pbounds = {'log_reg_coef': (np.log(coef_min), np.log(coef_max))}
        optimizer = BayesianOptimization(
            f=objective_log,
            pbounds=pbounds,
            random_state=seed,
            )
        init_points = min(3, maxiter)
        optimizer.maximize(init_points=init_points, n_iter=max(0, maxiter - init_points))
        best_regcoef = min(evaluated_runs, key=lambda x: x[1])[0]

    # Retrain with BO-best coefficient (epsilon unused for cal/balanced).
    val_rate_retrain = 0.1 if method_name == "cal" else 0
    comment_retrain = comment + "regcoef_" + str(best_regcoef)

    train(
        simulator_name,
        method_name,
        0,
        n,
        seed,
        batch_size,
        epochs,
        val_rate=val_rate_retrain,
        comment=comment_retrain,
        reg_coef=best_regcoef,
    )

    end_time = time.time()
    total_time = end_time - start_time

    with open(result_dir / "best_regcoef.txt", "w") as f:
        f.write(f"{best_regcoef}\n")

    with open(result_dir / "total_time.txt", "w") as f:
        f.write(f"{total_time}\n")

    with open(result_dir / "evaluated_regcoef_kl.csv", "w") as f:
        f.write(f"regcoef,{metric}\n")
        for regcoef, mval, _ in evaluated_runs:
            f.write(f"{regcoef},{mval}\n")

    final_run_dir = _result_dir(simulator_name, method_name, n, seed, comment_retrain)

    if final_run_dir.exists():
        for item in final_run_dir.iterdir():
            target = result_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        try:
            final_run_dir.rmdir()
        except OSError:
            pass

    canonical_dir = _result_dir(simulator_name, method_name, n, seed, comment)
    trial_dirs = {run_dir for _, _, run_dir in evaluated_runs}
    for run_dir in trial_dirs:
        if run_dir.resolve() == canonical_dir.resolve():
            continue
        if run_dir.resolve() == final_run_dir.resolve():
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator_name", type=str, default="lv", choices=["slcp", "lv", "ik", "tm", "cosmo"])
    parser.add_argument("--method_name", type=str, default="cal", choices=["cal", "balanced"])
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    parser.add_argument("--n", type=int, nargs="+", default=[4096])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--optimiser", type=str, default="BO", choices=["BO"])
    parser.add_argument("--coef_min", type=float, default=0.001)
    parser.add_argument("--coef_max", type=float, default=10.0)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.0, help="Tolerance above the optimum KL: pick the largest eps with f(eps) < f(eps*) + delta")
    parser.add_argument("--metric", type=str, default="kl_cal", choices=["kl_cal", "nlpd", "kl_cal_distance", "coverage_error"])
    parser.add_argument("--comment", type=str, default="", help="Optional comment to append to the result directory name")

    args = parser.parse_args()

    _set_keras_backend_for_method(args.method_name)

    for n, seed in itertools.product(args.n, args.seed):
        print(f"Training and optimising hyperparameters for {args.simulator_name} with {args.method_name} on {n} samples with seed {seed}")
        print(f"coef_min: {args.coef_min}, coef_max: {args.coef_max}, maxiter: {args.maxiter}, delta: {args.delta}, optimiser: {args.optimiser}")
        print(f"comment: {args.comment}")
        print(f"batch_size: {args.batch_size}, epochs: {args.epochs}")
        print(f"seed: {seed}")
        print(f"n: {n}")
        print(f"simulator_name: {args.simulator_name}")
        print(f"method_name: {args.method_name}")
        print(f"metric: {args.metric}")
        train_and_optimise_hyperparams(
            args.simulator_name,
            args.method_name,
            n,
            seed,
            args.batch_size,
            args.epochs,
            args.comment,
            args.coef_min,
            args.coef_max,
            args.maxiter,
            args.delta,
            args.optimiser,
            args.metric,
        )