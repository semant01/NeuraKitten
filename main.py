import logging
import os
import subprocess

from src.config import NeuraConfig
from src.pipeline import NeuraPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clear_terminal() -> None:
    """Clear terminal, OS based."""
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def main() -> None:
    """Entry point for NeuraKitten experiments."""
    logging.info("Hello, NeuraKitten! Initializing pipeline...\n")

    cfg = NeuraConfig(
        epochs=501,
        hidden_layers=[32, 16, 16, 16, 16, 32],
        samples=1800,
        batch_size=90,
        balanced_batches=True,
        data_mode="multidonut",
        mdonut_radii=[4, 3.5, 2, 1.9, 1.7, 1, 0.5],
        mdonut_r_evenly_dist=True,
        feature_mode="cartesian",
        initial_lr=0.001,
        decay_rate=0.01,
        use_interaction=False,
        use_squares=False,
        use_trig=False,
        view_range=5,
    )

    pipeline = NeuraPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    clear_terminal()
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
