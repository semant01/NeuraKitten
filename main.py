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
        epochs=1000,
        hidden_layers=[128,96,64],
        samples=5000,
        batch_size=50,
        balanced_batches=True,
        data_mode="spirals",  # "multidonut", "spirals", "rhodonea", "iris"
        # mdonut_r_evenly_dist=False,
        # mdonut_radii=[4, 3, 2, 1],
        spiral_max_radius=1,
        spiral_num_classes=5,
        spiral_turns=3,
        noise=0.02,
        feature_mode="cartesian",
        initial_lr=0.001,
        decay_rate=0.000,
        use_interaction=False,
        use_squares=False,
        use_trig=False,
        visualize=False,
        frame_log=100,
        frame_visual=1,
        save_to_file=True,
        checkpoint_interval=1,
    )

    pipeline = NeuraPipeline(experiment_name="NeuraKitten_Standard", cfg=cfg)

    try:
        pipeline.run(
            mode="replay",
            experiment_path="experiments/20260803_173932_NeuraKitten_Standard",
        )
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
