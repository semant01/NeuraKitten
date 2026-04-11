import logging
import os
import subprocess

from src.config import NeuraConfig
from src.data_utils import DataFactory, DataScaler, FeatureEngine
from src.model import DeepNeuralNetwork
from src.trainer import fit

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clear_terminal() -> None:
    """Clear terminal, OS based."""
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def main() -> None:
    """Run the main entry point of the script."""
    logging.info("Hello, NeuraKitten!\n")

    cfg = NeuraConfig(
        epochs=501,
        hidden_layers=[32, 24, 16, 12],
        samples=2048,
        batch_size=32,
        data_mode="spirals",  # "spirals", "rhodonea", "multidonut"
        spiral_max_radius=2.0,
        num_spirals=5,
        spiral_turns=5,
        noise=0.03,
        feature_mode="polar",
        view_range=2.5,
        initial_lr=0.001,
        decay_rate=0.01,
    )

    logging.info(
        f"Run for {cfg.epochs} epochs\n\n"
        f"Samples: {cfg.samples} ; batch size: {cfg.batch_size}\n\n"
        f"Data mode: {cfg.data_mode}\n"
        f"Noise: {cfg.noise}\n"
        f"Seed: {cfg.seed}\n\n"
        f"Feature mode: {cfg.feature_mode}\n"
    )

    # 1. Generate the dataset based on cfg.data_mode
    factory = DataFactory(cfg)
    X_raw, targets = factory.generate()

    # 2. Feature Engineering, transform coordinates (e.g., Cartesian to Polar)
    engine = FeatureEngine(cfg)
    X_featured = engine.transform(X_raw)

    # 3. Scaling
    # Normalizes data to [-1, 1] range
    scaler = DataScaler(cfg)
    X_transformed = scaler.fit_transform(X_featured)

    # 4. Initialize the MLP
    # Dimensions are detected automatically from the processed features
    input_dim = X_transformed.shape[1]
    output_dim = targets.shape[1]
    hidden_layers = cfg.hidden_layers
    logging.info(
        f"Neural Network: [{input_dim}] --> {hidden_layers} --> [{output_dim}]\n"
    )
    logging.info(f"Hidden layers activation: {cfg.activation_hidden}")

    brain = DeepNeuralNetwork(
        config=cfg, layers_size=[input_dim] + hidden_layers + [output_dim]
    )

    # 5. Run Training & Visualization
    fit(
        model=brain,
        inputs=X_transformed,
        targets=targets,
        cfg=cfg,
        X_raw=X_raw,  # Original points for plotting
        scaler=scaler,  # Required to de-scale boundaries
        engine=engine,  # Required to re-feature decision grid
    )


if __name__ == "__main__":
    clear_terminal()
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
