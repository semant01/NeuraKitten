import logging
import os
import subprocess

from src.pipeline import NeuraPipeline
from src.structures import NeuraConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clear_terminal() -> None:
    """Clear terminal, OS based."""
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def main() -> None:
    """Entry point for NeuraKitten experiments."""
    logging.info("Hello, NeuraKitten! Initializing pipeline...\n")

    cfg = NeuraConfig(
        epochs=251,
        hidden_layers=[3, 3],
        # samples=2000,
        batch_size=12,
        balanced_batches=True,
        data_mode="iris",
        iris_pca=False,
        feature_mode="cartesian",
        initial_lr=0.005,
        decay_rate=0.01,
        use_interaction=False,
        use_squares=False,
        use_trig=False,
        visualize=True,
        vis_axes=(2, 3),
    )

    pipeline = NeuraPipeline("Test 1", cfg)
    pipeline.run()


if __name__ == "__main__":
    clear_terminal()
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
