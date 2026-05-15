## Set up

1. Install uv https://docs.astral.sh/uv/getting-started/installation/ if you don't have one.
2. `uv sync` to install dependencies

## Demonstration

## Running experiments

### Run experiment with our method

``uv run experiments/train.py --simulator lv --method dro_ub  --epsilon 1.0 --n 1024``

-  **simulator**: `lv` stands for "Lotka–Volterra". You can also choose `tm` (two moons), `slcp` (SLCP), and `ik` (Inverse Kinematics).
-  **method**: `dro_ub` is our method.
-  **epsilon**: Hyperparameters for our method. Larger `epsilon` corresponds to larger Wasserstein ball, leading to more conservative inference.
-  **n**: simulation budget (number of $(\theta, x)$ pair to sample for training)
-  **Other parameters**:
    - seed: setting seed for experiments (sampling from simulator, weight initialisation, batch selection...)
    - epochs: number of epochs to train.
    - val_rate: rate of validation data.

### Run experiment with our method + epsilon selection.
``uv run experiments/train_and_optimise_hyperparams.py --simulator lv --n 1024 --metric kl_cal --optimiser BO --eps_min 0.001 --eps_max 10 --maxiter 10``

- **metric**: metric used to pick `epsilon`. Our propose $kl_\mathrm{cal}$ (`kl_cal`) is used as a default. You can use other metric such as NLPD. 
    - We always use 90% of the data to train and 10% of the data to evaluate the metric.
    - After the optimal `epsilon` is identified, we train using all the data with the optimal `epsilon`.
- **optimiser**: For efficient search of optimal `epsilon`, we use Bayeisan optimisation.
- **eps_min, eps_max**: Lower and upper bound for `epsilon` to be searched.
- **maxiter**: Number of iteration for Bayesian optimisation.



