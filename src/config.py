from dataclasses import dataclass, field
from typing import List


@dataclass
class NeuraConfig:
    """Configuration dataset."""

    # --- Network Architecture ---
    hidden_layers: List[int] = field(default_factory=lambda: [12, 12])
    # activation_hidden: str = "leaky_relu"
    # activation_output: str = "sigmoid"

    # --- Hyperparameters & Training ---
    epochs: int = 1001
    batch_size: int = 32
    initial_lr: float = 0.001
    decay_rate: float = 0.01
    min_lr: float = 1e-5

    # --- ADAM Optimizer ---
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # --- Data Generation ---
    samples: int = 2048
    noise: float = 0.05
    data_mode: str = "rhodonea"  # "donut", "spiral", "rhodonea"

    #       Donut
    donut_r_inner: float = 0.3  # Inner radius
    donut_r_outer: float = 0.7  # Outer radius
    donut_r_evenly_dist: bool = False  # distribute evenly along the radius
    #                            or closer to the center

    #       Spiral
    spiral_turns: float = 2.5  # number of turns

    #       Rhodonea
    rose_k: float = 3.0  # 2k petals for rhodonea
    rose_a: float = 1.0  # Maximum radius of rhodonea

    # --- Feature Engineering ---
    feature_mode: str = "polar"  # "cartesian", "polar"
    use_squares: bool = False
    use_interaction: bool = False
    use_trig: bool = False

    # --- Data Scaling ---
    feature_range = (-1, 1)

    # --- UI / UX / Visualization ---
    visualize: bool = True
    view_range: float = 1.5
    color_gradient: bool = False
    show_dataset_points: bool = True
    show_levels: bool = True
    frame_skip: int = 1
    frame_log: int = 100

    # --- Reproducibility ---
    # Reproducibility anchor and the Answer to the Ultimate Question of Life.
    seed: int | None = 42
