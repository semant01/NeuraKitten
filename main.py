import logging
import os
import subprocess

import numpy as np

from src.data_utils import DataFactory, DataScaler, FeatureEngine
from src.model import DeepNeuralNetwork
from src.trainer import fit

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clear_terminal() -> None:
    """Clear terminal, OS based."""
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def main() -> None:
    """Run the main entry point of the script."""
    logging.info("Main entry point of the script")

    np.random.seed(42)

    # 1. Generate the dataset
    # You can choose between factory.generate_spiral() or factory.generate_donut()
    t_samples = 512
    t_noise = 0.05
    factory = DataFactory(samples=t_samples, noise=t_noise)
    X_raw, targets = factory.generate_spiral(turns=2)
    logging.info(f"Dataset generated: Samples - {t_samples}; noise - {t_noise}")
    logging.info(f"Raw data shape. X_raw: {X_raw.shape}; Targets: {targets.shape}")

    # 2. Feature Engineering
    engine = FeatureEngine(
        use_squares=True, use_interaction=True, use_trig=True, mode="cartesian"
    )
    logging.info(f"Mode: {engine.mode}")
    if engine.mode == "cartesian":
        logging.info(
            "Featuring: Squares: %s; Interaction: %s; Trigonometry: %s;",
            engine.use_squares,
            engine.use_interaction,
            engine.use_trig,
        )
    X_featured = engine.transform(X_raw)

    # 3. Scaling
    # Normalizes data to [-1, 1] range, essential for neural network convergence
    logging.info("Scaling...")
    scaler = DataScaler(feature_range=(-1, 1))
    X_transformed = scaler.fit_transform(X_featured)
    logging.info(f"Final Input Shape: {X_transformed.shape}")
    logging.info(
        f"X_transformed Range: [{X_transformed.min():.2f}, {X_transformed.max():.2f}]"
    )

    # 4. Initialize the MLP
    # Define architecture: [Input Nodes, Hidden Layers..., Output Nodes]
    # Input nodes are calculated automatically based on your features
    input_dim = X_transformed.shape[1]
    output_dim = targets.shape[1]
    hidden_layers = [12, 12]
    logging.info(
        f"Create Neural Network: [{input_dim}] --> {hidden_layers} --> [{output_dim}]"
    )

    brain = DeepNeuralNetwork([input_dim] + hidden_layers + [output_dim])

    # 5. Run Training & Visualization
    fit(
        model=brain,
        engine=engine,
        inputs=X_transformed,
        targets=targets,
        scaler=scaler,
        X_raw=X_raw,
        epochs=10001,  # Total training iterations
        batch_size=128,  # Number of samples per gradient update
        base_lr=0.01,  # Initial learning rate
        lr_gradient=0.9998,  # Smooth learning rate decay
        visualize=True,  # Toggle real-time Matplotlib animation
        color_gradient=False,  # Use solid colors or probability gradients
        show_dataset_points=True,  # Show/hide original data points
        show_levels=False,  # Toggle decision boundary contours
    )


if __name__ == "__main__":
    clear_terminal()
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
