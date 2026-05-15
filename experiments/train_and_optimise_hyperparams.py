import argparse
import itertools
import pickle
from pathlib import Path
import shutil

import bayesflow as bf
import keras

from train import train
from src.approximator.continuous_approximator_dro_ub import ContinuousApproximatorDROUB
from src.diagnostics.kl_based_calibration import estimate_calibration_kl, obtain_u_samples_q, obtain_u_samples_distance
from src.diagnostics.coverage import compute_coverage
from scipy.optimize import minimize_scalar
import time
from bayes_opt import BayesianOptimization

import numpy as np
import os

CUSTOM_OBJECTS = {
    "ContinuousApproximatorDROUB": ContinuousApproximatorDROUB,
}


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


def train_and_evaluate(simulator_name, method_name, epsilon, n, seed, batch_size, epochs, comment="", metric = "kl_cal"):
    """Train then evaluate. Returns ``(metric_value, run_dir)``.
    """
    run_dir, approximator = train(simulator_name, method_name, epsilon, n, seed, batch_size, epochs, val_rate = 0.1, comment = comment)
    validation_data = load_validation_data_from_npe(simulator_name, n, seed, val_rate=0.1)

    if metric == "kl_cal":
        post_samples = approximator.sample(
            conditions=validation_data, num_samples=1000
        )["parameters"]
        u_samples = obtain_u_samples_q(validation_data, post_samples, approximator, M=1000)
        kl_cal = estimate_calibration_kl(u_samples, post_samples)
        
        return kl_cal, run_dir

    elif metric == "nlpd":
        nlpd = - np.asarray(approximator.log_prob(validation_data)).ravel()
        return float(np.mean(nlpd)), run_dir

    elif metric == "kl_cal_distance":
        post_samples = approximator.sample(
            conditions=validation_data, num_samples=1000
        )["parameters"]
        u_samples = obtain_u_samples_distance(validation_data["parameters"], post_samples)
        kl_cal = estimate_calibration_kl(u_samples, post_samples)
        return kl_cal, run_dir

    elif metric == "coverage_error":
        alpha = 0.05
        post_samples = approximator.sample(
            conditions=validation_data, num_samples=1000
        )["parameters"]
        coverage = compute_coverage(validation_data, post_samples, approximator, alpha=alpha, M=1000)
        coverage_error = np.abs(coverage - 1 + alpha)
        return coverage_error.squeeze(), run_dir

def train_and_optimise_hyperparams(
    simulator_name,
    method_name,
    n,
    seed,
    batch_size,
    epochs,
    comment="",
    eps_min=1e-2,
    eps_max=10,
    maxiter=10,
    delta=0.0,
    optimiser = "BO", # "BO" or "bounded"
    metric = "kl_cal",
):
    comment_suffix = f"_{comment}" if comment else ""

    result_dir = Path(
        f"results/{simulator_name}/{method_name}/"
        f"epsilon_tune_n_{n}_seed_{seed}_maxiter_{maxiter}_delta_{delta}_metric_{metric}{comment_suffix}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"maxiter_{maxiter}_delta_{delta}_metric_{metric}"
    run_comment = f"{comment}_{run_tag}" if comment else run_tag

    start_time = time.time()

    evaluated_runs = []

    if optimiser == "BO":
        def objective_log(log_eps):
            eps = float(round(np.exp(log_eps), 5))  # avoid float tail noise
            metric_value, run_dir = train_and_evaluate(
                simulator_name,
                method_name,
                eps,
                n,
                seed,
                batch_size,
                epochs,
                run_comment,
                metric,
            )
            evaluated_runs.append((eps, float(metric_value), run_dir))


            eps_score = (log_eps - np.log(eps_min)) / (
                np.log(eps_max) - np.log(eps_min))

            objective_value = metric_value - delta * eps_score

            return - objective_value # maximise the negative KL


        pbounds = {'log_eps': (np.log(eps_min), np.log(eps_max))}
        optimizer = BayesianOptimization(
            f=objective_log,
            pbounds=pbounds,
            random_state=seed,
            )
        optimizer.maximize(
                init_points=3,
                n_iter= maxiter - 3,
        )
        best_eps = min(evaluated_runs, key=lambda x: x[1])[0]

    elif optimiser == "brent":
        def objective_log(log_eps):
            eps = float(np.exp(log_eps))
            kl_cal, run_dir = train_and_evaluate(
                simulator_name,
                method_name,
                eps,
                n,
                seed,
                batch_size,
                epochs,
                run_comment,
                metric,
            )
            evaluated_runs.append((eps, float(kl_cal), run_dir))
            return kl_cal

        res = minimize_scalar(
            objective_log,
            bounds=(np.log(eps_min), np.log(eps_max)),
            method="bounded",
            options={"maxiter": maxiter},
        )
        best_eps = float(np.exp(res.x))
        best_run_dir = min(evaluated_runs, key=lambda x: x[1])[2]

    with open(result_dir / "best_epsilon.txt", "w") as f:
        f.write(f"{best_eps}\n")

    with open(result_dir / "evaluated_eps_kl.csv", "w") as f:
        f.write("eps,kl_cal\n")
        for eps, kl_cal, _ in evaluated_runs:
            f.write(f"{eps},{kl_cal}\n")

    # Retrain at the canonical path with val_rate=0 (full data).
    final_run_dir, _ = train(
        simulator_name, method_name, best_eps, n, seed,
        batch_size, epochs, val_rate=0, comment=run_comment,
    )

    end_time = time.time()
    total_time = end_time - start_time

    with open(result_dir / "total_time.txt", "w") as f:
        f.write(f"{total_time}\n")

    # Move the final model into result_dir.  Materialize the iterator first
    # (mutating a directory while iterating it via a generator is undefined).
    if final_run_dir.exists():
        for item in list(final_run_dir.iterdir()):
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


    for run_dir in {run_dir for _, _, run_dir in evaluated_runs}:
        if run_dir.resolve() == final_run_dir.resolve():
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator_name", type=str, default="lv", choices=["slcp", "lv", "ik", "tm", "cosmo"])
    parser.add_argument("--method_name", type=str, default="dro_ub", choices=["dro_ub"])
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    parser.add_argument("--n", type=int, nargs="+", default=[4096])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--optimiser", type=str, default="BO", choices=["BO", "brent"])
    parser.add_argument("--eps_min", type=float, default=0.001)
    parser.add_argument("--eps_max", type=float, default=10.0)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.0, help="minimise kl_cal - delta * eps to encourage larger eps")
    parser.add_argument("--metric", type=str, default="kl_cal", choices=["kl_cal", "nlpd", "kl_cal_distance", "coverage_error"])
    parser.add_argument("--comment", type=str, default="", help="Optional comment to append to the result directory name")

    args = parser.parse_args()

    backend_by_method = {
        "dro_ub": "jax",
        "npe": "jax",
        "npe_stop": "jax",
        "cal": "torch",
        "balanced": "jax",
    }
    backend = backend_by_method.get(args.method_name)
    os.environ["KERAS_BACKEND"] = backend

    for n, seed in itertools.product(args.n, args.seed):
        print(f"Training and optimising hyperparameters for {args.simulator_name} with {args.method_name} on {n} samples with seed {seed}")
        print(f"eps_min: {args.eps_min}, eps_max: {args.eps_max}, maxiter: {args.maxiter}, delta: {args.delta}, optimiser: {args.optimiser}")
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
            args.eps_min,
            args.eps_max,
            args.maxiter,
            args.delta,
            args.optimiser,
            args.metric,
        )