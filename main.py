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
        epochs=101,
        hidden_layers=[16, 16, 16],
        samples=2000,
        batch_size=25,
        balanced_batches=True,
        data_mode="spirals",  # "multidonut", "spirals", "rhodonea"
        spiral_max_radius=1,
        spiral_num_classes=5,
        spiral_turns=3,
        noise=0.03,
        feature_mode="polar",
        initial_lr=0.002,
        decay_rate=0.01,
        use_interaction=False,
        use_squares=False,
        use_trig=False,
        view_range=1.2,
    )

    pipeline = NeuraPipeline("Test 1", cfg)
    pipeline.run()


if __name__ == "__main__":
    clear_terminal()
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
