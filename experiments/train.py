import argparse
import itertools
import numpy as np

from pathlib import Path
import pickle
from time import time
from contextlib import nullcontext
import os

def train(simulator_name, method_name, epsilon, n, seed, batch_size, epochs, val_rate = 0.0, comment="", reg_coef = None):

    backend_by_method = {
        "dro_ub": "jax",
        "npe": "jax",
        "npe_stop": "jax",
        "cal": "torch",
        "balanced": "jax",
    }
    backend = backend_by_method.get(method_name)
    os.environ["KERAS_BACKEND"] = backend

    if backend == "torch":
        from src.approximator.continuous_approximator_cal import ContinuousApproximatorCAL
        # import torch
    else:
        from src.approximator.continuous_approximator_dro_ub import ContinuousApproximatorDROUB
        from src.approximator.continuous_approximator_balanced import ContinuousApproximatorBalanced
    
    # import after backend is set
    import keras
    import bayesflow as bf
    from bayesflow.approximators.continuous_approximator import ContinuousApproximator
    from bayesflow.networks.coupling_flow import CouplingFlow
    from bayesflow.datasets import OfflineDataset

    keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Paths relative to project root (works whether run as `python main.py` or `python toy_exp/main.py`)
    safe_comment = comment.replace(" ", "_") if comment else ""
    comment_suffix = f"_comment_{safe_comment}" if safe_comment else ""
    if method_name in ["dro_ub", "dro_with_C"]:
        result_path = Path(f"results/{simulator_name}/{method_name}/epsilon_{epsilon}_n_{n}_seed_{seed}{comment_suffix}")

    elif method_name in ["npe", "npe_stop", "cal", "balanced", "dro"]:
        result_path = Path(f"results/{simulator_name}/{method_name}/n_{n}_seed_{seed}{comment_suffix}")  # epsilon is not used for npe
    else:
        raise ValueError(f"Method {method_name} not found")
    
    result_path.mkdir(parents=True, exist_ok=True)
   
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("parameters", "inference_variables")
        .rename("observables", "inference_conditions")
    )

    if method_name in ("npe_stop") and val_rate == 0.0:
        raise ValueError("val_rate must be greater than 0.0 for npe_stop and cal")

    def obtain_simulator(simulator_name, rng):
        if simulator_name == "slcp":
            from bayesflow.simulators import SLCP
            return SLCP(rng = rng)

        elif simulator_name == "lv":
            from bayesflow.simulators import LotkaVolterra
            return LotkaVolterra(rng = rng)

        elif simulator_name == "ik":
            from bayesflow.simulators import InverseKinematics
            return InverseKinematics(rng = rng)

        elif simulator_name == "tm":
            from bayesflow.simulators import TwoMoons
            return TwoMoons(rng = rng)

        elif simulator_name == "cosmo":
            return "cosmo"

        raise ValueError(f"Simulator {simulator_name} not found")

    simulator = obtain_simulator(simulator_name, rng)
    inference_network = CouplingFlow(subnet_kwargs = {"activation": "tanh"}) # use tanh activation to satisfy the boundedness.

    # Load or sample training data
    npe_training_data_path = Path(f"results/{simulator_name}/npe/n_{n}_seed_{seed}/training_data.pkl")
    npe_training_data_path.parent.mkdir(parents=True, exist_ok=True)
    if npe_training_data_path.exists():
        with open(npe_training_data_path, "rb") as f:
            training_data = pickle.load(f)
    else:
        training_data = simulator.sample(batch_shape=(n,)) # will not work for cosmo
        with open(npe_training_data_path, "wb") as f:
            pickle.dump(training_data, f)     

    if val_rate > 0.0:
        train_size = int(n * (1 - val_rate)) # validation set is used later (in train_and_optimise_hyperparams.pyå)
        full_data = training_data
        training_data = {k: v[:train_size] for k, v in full_data.items()}
    dataset = OfflineDataset(training_data, batch_size = batch_size, adapter = adapter)


    if method_name == "dro_ub":
        approximator = ContinuousApproximatorDROUB(adapter=adapter, inference_network=inference_network, epsilon=epsilon)
    elif method_name == "npe" or method_name == "npe_stop":
        approximator = ContinuousApproximator(adapter=adapter, inference_network=inference_network)
    elif method_name == "cal":
        from src.utils.calnpe_helper import get_prior_sampler
        prior_sampler = get_prior_sampler(simulator_name)
        gamma = 5 if reg_coef is None else reg_coef
        approximator = ContinuousApproximatorCAL(adapter=adapter, inference_network=inference_network, prior_sampler=prior_sampler, gamma = gamma, L = 16, calibration = False)
    elif method_name == "balanced":
        from src.utils.balnpe_helper import get_prior_sampler
        lmbda = 100 if reg_coef is None else reg_coef
        prior_sampler = get_prior_sampler(simulator_name)
        approximator = ContinuousApproximatorBalanced(
            adapter=adapter,
            inference_network=inference_network,
            prior_sampler=prior_sampler,
            lmbda=lmbda,
        )
    else:
        raise ValueError(f"Method {method_name} not found")

    if val_rate > 0.0:
        validation_data = {k: v[train_size:] for k, v in full_data.items()}
        validation_data = adapter(validation_data)
    else:
        validation_data = None
    if method_name == "npe_stop" and val_rate > 0.0:
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1, mode="min")] # early stopping on the loss on the validation set
    elif method_name == "cal":
        if val_rate > 0.0:
            callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(result_path / "best.weights.h5"),
                monitor="val_loss/inference_loss", # - log q(theta | x)
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                mode="min",
            ), 
            keras.callbacks.TerminateOnNaN(), # terminate the training if NaN is encountered in validation metric
        ]
        else:
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=str(result_path / "epoch_{epoch:04d}.weights.h5"),
                    save_weights_only=True,
                    save_best_only=False,
                    save_freq="epoch",
                ),
                keras.callbacks.TerminateOnNaN()]
    else:
        callbacks = []
        validation_data = None

    learning_rate = 5e-4

    grad_context = nullcontext()
    if backend == "torch":
        import torch
        grad_context = torch.enable_grad()
        
    start_time = time()
    with grad_context:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.1) # don't use any weight decay for easier interpretation of the results
        approximator.compile(optimizer=optimizer)
        history = approximator.fit(
            dataset=dataset,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_data,
        )
    end_time = time()

    if method_name == "cal":
        if val_rate > 0.0:
            best_weights_path = result_path / "best.weights.h5"
            if best_weights_path.exists():
                approximator.load_weights(str(best_weights_path))

        else:
            # load the last checkpoint before the NaN is encountered
            epoch_ckpts = sorted(result_path.glob("epoch_*.weights.h5"))
            if epoch_ckpts:
                last_checkpoint_before_nan = epoch_ckpts[-2]
                approximator.load_weights(str(last_checkpoint_before_nan))


    # only save the final model
    model_path = result_path / "model.keras"
    approximator.save(str(model_path))

    history_dict = history.history
    with open(result_path / "history.pkl", "wb") as f:
        pickle.dump(history_dict, f)

    training_time = end_time - start_time
    total_epochs = len(history.history['loss'])

    with open(result_path / "training_time.txt", "w") as f:
        f.write(f"Total training time: {training_time}\n")
        f.write(f"Total epochs: {total_epochs}\n")
        f.write(f"Time per epochs: {training_time / total_epochs}\n")

    return result_path, approximator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator_name", type=str, default="slcp", choices=["slcp", "lv", "ik", "tm", "cosmo"])
    parser.add_argument("--method_name", type=str, default="dro_ub", choices=["dro_ub", "npe", "npe_stop", "cal", "balanced"])
    parser.add_argument("--seed", type=int, nargs='+', default=[1])
    parser.add_argument("--n", type=int, nargs='+', default=[512])
    parser.add_argument("--epsilon", type=float, nargs='+', default=[0.0])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--val_rate", type=float, default=0.0)
    parser.add_argument("--comment", type=str, default="", help="Optional comment to append to the result directory name")
    parser.add_argument("--reg_coef", type=float, nargs='+', default=[None], help = "lagrange multiplier for baseline method")
    args = parser.parse_args()
    
    for epsilon, n, seed, reg_coef in itertools.product(
        args.epsilon, args.n, args.seed, args.reg_coef
    ):
        train(
            args.simulator_name,
            args.method_name,
            epsilon,
            n,
            seed,
            args.batch_size,
            args.epochs,
            val_rate=args.val_rate,
            comment=args.comment,
            reg_coef=reg_coef,
        )

    print(
        f"Training completed: simulator={args.simulator_name}, method={args.method_name}, "
        f"epsilon={epsilon}, n={n}, seed={seed}, comment={args.comment}, reg_coef={reg_coef}"
    )