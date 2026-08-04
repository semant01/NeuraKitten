import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol for configuration objects to avoid direct coupling."""

    __dict__: dict[str, Any]


class ExperimentManager:
    """Manages the filesystem infrastructure for neural network experiments.

    Handles creation of unique session directories, persists configurations,
    and manages binary storage for datasets and model checkpoints.

    Attributes:
        base_path (Path): The root directory for all experiments.
        exp_dir (Path): The specific directory for the current session.
        checkpoints_dir (Path): Directory for storing epoch-wise model weights.

    """

    def __init__(self, base_path: str = "experiments") -> None:
        """Initialize the ExperimentManager with a base path.

        Args:
            base_path: String path or subfolder where experiments will be stored.

        """
        self.base_path: Path = Path(base_path)
        self.exp_dir: Path = Path()
        self.checkpoints_dir: Path = Path()

    def create_session(self, mode_name: str) -> str:
        """Create a unique directory structure for a new experiment session.

        Generates a timestamped folder and ensures no overwrite by incrementing
        version suffixes if a collision occurs.

        Args:
            mode_name: Name of the experiment or dataset.

        Returns:
            The string path to the created experiment directory.

        """
        clean_name = re.sub(r"[^\w\s.-]", "_", mode_name).strip()
        clean_name = re.sub(r"\s+", "_", clean_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{clean_name}"
        self.exp_dir = self.base_path / folder_name

        counter = 1
        while self.exp_dir.exists():
            self.exp_dir = self.base_path / f"{folder_name}_v{counter}"
            counter += 1

        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        return str(self.exp_dir)

    def save_config(self, config_obj: ConfigProtocol) -> None:
        """Serialize the configuration object to a JSON file.

        Args:
            config_obj: The configuration instance (dataclass) to save.

        """
        # Note: __dict__ is standard for dataclasses, but for nested objects
        # asdict() from dataclasses module would be more robust. - Added
        # data = config_obj.__dict__
        data = asdict(config_obj) if is_dataclass(config_obj) else config_obj.__dict__
        file_path = self.exp_dir / "config.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def save_metrics(self, ctx_obj: ConfigProtocol) -> None:
        """Save the final experiment metrics and history to a JSON file.

        Args:
            ctx_obj: The experiment context instance (dataclass) to save.

        """
        data = asdict(ctx_obj) if is_dataclass(ctx_obj) else ctx_obj.__dict__

        file_path = self.exp_dir / "metrics.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def save_dataset(self, x: np.ndarray, y: np.ndarray) -> None:
        """Save the experiment dataset to a compressed NumPy file.

        Args:
            x: Input feature matrix.
            y: Target labels.

        """
        file_path = self.exp_dir / "dataset.npz"
        np.savez_compressed(file_path, X=x, y=y)

    def save_checkpoint(self, epoch: int, state: dict[str, np.ndarray]) -> None:
        """Persist model weights and biases for a specific epoch.

        Args:
            epoch: The current training epoch number.
            state: Dictionary mapping parameter names to NumPy arrays.

        """
        file_path = self.checkpoints_dir / f"epoch_{epoch:05d}.npz"
        # We use **state to store parameters as named arrays within the .npz file.
        # The # type: ignore is used to suppress false-positive type checker
        # errors regarding the allow_pickle parameter in savez_compressed.
        np.savez_compressed(file_path, **state)  # type: ignore[arg-type]

    def load_config(self, exp_dir: Path) -> dict[str, Any]:
        """Load configuration dictionary from a specific experiment directory."""
        file_path = exp_dir / "config.json"
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dataset(self, exp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load the dataset from a compressed NumPy file."""
        file_path = exp_dir / "dataset.npz"
        data = np.load(file_path)
        return data["X"], data["y"]

    def load_metrics(self, exp_dir: Path) -> list[dict[str, Any]]:
        """Load experiment metrics history from a JSON file."""
        file_path = exp_dir / "metrics.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Assuming metrics are stored in a 'metrics' key within the saved dict
            return data.get("metrics", [])

    def load_checkpoint(self, exp_dir: Path, epoch: int) -> dict[str, np.ndarray]:
        """Load model weights and biases for a specific epoch."""
        file_path = exp_dir / "checkpoints" / f"epoch_{epoch:05d}.npz"
        with np.load(file_path, allow_pickle=True) as data:
            return {name: data[name] for name in data.files}
