from dataclasses import dataclass, field
from typing import List


@dataclass
class NeuraConfig:
    """Configuration dataset."""

    # --- Network Architecture ---
    hidden_layers: List[int] = field(default_factory=lambda: [12, 12])
    activation_hidden: str = "leaky_relu"

    # --- Categorical Cross-Entropy ---
    eps = 1e-15

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
    noise: float = 0
    data_mode: str = "donut"

    #       Multi-donut
    mdonut_radii: List[float] = field(
        default_factory=lambda: [
            1.2,
            0.7,
            0.5,
            0.3,
        ]
    )
    mdonut_r_evenly_dist: bool = False  # distribute evenly along the radius
    #                            or closer to the center

    #       Donut
    donut_r_inner: float = 0.3  # Inner radius
    donut_r_outer: float = 0.7  # Outer radius
    donut_r_evenly_dist: bool = False  # distribute evenly along the radius
    #                            or closer to the center

    #       Spiral
    num_spirals: int = 3
    spiral_turns: float = 2.5  # number of semi-turns
    spiral_max_radius: float = 1.0

    #       Rhodonea
    rhodonea_k: float = 3.0  # 2k petals for rhodonea
    rhodonea_max_radius: float = 1.0  # Maximum radius of rhodonea
    rhodonea_NumOfLayers: int = 1  # Number of classes in addition to background class
    rhodonea_r_evenly_dist: bool = False

    # --- Feature Engineering ---
    feature_mode: str = "polar"  # "cartesian", "polar"
    use_squares: bool = False
    use_interaction: bool = False
    use_trig: bool = False

    # --- Data Scaling ---
    feature_range = (-1, 1)

    # --- UI / UX / Visualization ---
    visualize: bool = True
    resolution: int = 300
    view_range: float = 1.5
    cmap: str = "CMRmap"
    #     cmap options:  "gist_stern", "tab20c", "CMRmap", "nipy_spectral" ,"gnuplot2"
    show_dataset_points: bool = True
    frame_skip: int = 1
    frame_log: int = 100

    # --- Reproducibility ---
    # Reproducibility anchor and the Answer to the Ultimate Question of Life.
    seed: int | None = 42
