import logging
import os
import subprocess
import sys

from src.pipeline import NeuraPipeline
from src.structures import NeuraConfig


def clear_terminal() -> None:
    """Clear terminal, OS based."""
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


def setup_logging() -> None:
    """Configure the logging level and format for the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    """Entry point for NeuraKitten experiments."""
    setup_logging()
    logging.info("Hello NeuraKitten... 🐾\n")

    cfg = NeuraConfig(
        epochs=5000,
        hidden_layers=[32, 32, 32, 32],
        samples=2000,
        batch_size=50,
        balanced_batches=True,
        data_mode="spirals",  # "multidonut", "spirals", "rhodonea", "iris"
        # mdonut_r_evenly_dist=False,
        # mdonut_radii=[4, 3, 2, 1],
        spiral_max_radius=1,
        spiral_num_classes=3,
        spiral_turns=7,
        noise=0.03,
        feature_mode="cartesian",
        initial_lr=0.002,
        decay_rate=0.001,
        use_interaction=True,
        use_squares=True,
        use_trig=True,
        visualize=True,
        frame_log=50,
        frame_visual=1,
        save_to_file=True,
        checkpoint_interval=50,
    )

    pipeline = NeuraPipeline(experiment_name="NeuraKitten_Standard", cfg=cfg)

    try:
        pipeline.run()
        logging.info("\nExperiment completed successfully.")
    except KeyboardInterrupt:
        logging.info("\nExperiment interrupted by user. Cleaning up...")
    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}", exc_info=True)


if __name__ == "__main__":
    clear_terminal()
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
