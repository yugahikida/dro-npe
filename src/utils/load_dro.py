from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_dro(
    *,
    simulator_name: str,
    n: int,
    seed: int,
    comment: str = "",
    results_root: str | Path = "results",
) -> Any:
    """
    Load a trained DRO approximator saved by `train.py`.

    Expects:
      results/<simulator_name>/dro/n_<n>_seed_<seed>[_comment_<comment>]/model.keras
    """

    # Must be set *before* importing keras (Keras 3 backend selection).
    os.environ.setdefault("KERAS_BACKEND", "jax")

    import keras  # noqa: PLC0415

    from src.approximator.continuous_approximator_dro import (  # noqa: PLC0415
        ContinuousApproximatorDRO,
    )

    safe_comment = comment.replace(" ", "_") if comment else ""
    comment_suffix = f"_comment_{safe_comment}" if safe_comment else ""
    result_dir = Path(results_root) / simulator_name / "dro" / f"n_{n}_seed_{seed}{comment_suffix}"
    model_path = result_dir / "model.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find saved DRO model at '{model_path}'. "
            "Make sure you trained with `--method_name dro` and the same n/seed/comment."
        )

    # Register custom class so deserialization works.
    model = keras.saving.load_model(
        str(model_path),
        custom_objects={"ContinuousApproximatorDRO": ContinuousApproximatorDRO},
    )
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a trained DRO model.keras")
    parser.add_argument("--simulator_name", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--results_root", type=str, default="results")
    args = parser.parse_args()

    model = load_dro(
        simulator_name=args.simulator_name,
        n=args.n,
        seed=args.seed,
        comment=args.comment,
        results_root=args.results_root,
    )
    print(f"Loaded model: {type(model)}")
