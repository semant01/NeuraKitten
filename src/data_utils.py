from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.config import NeuraConfig


class NeuraDataLoader:
    """Manages dataset access by handling batching, shuffling, and class balancing.

    This loader acts as an intermediate layer between the raw data and the
    training loop, ensuring that the model receives stabilized gradients
    through controlled sample distribution.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        balanced: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the data loader with input features and target labels.

        Args:
            inputs (np.ndarray): Input feature matrix of shape (N, features).
            targets (np.ndarray): One-hot encoded target labels of shape (N, classes).
            batch_size (int): Number of samples per training batch.
            balanced (bool): If True, ensures equal class representation in each batch.
            seed (int | None): Random seed for reproducibility.

        """
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.balanced = balanced
        self.rng = np.random.default_rng(seed=seed)

        # Preliminary Class Indexing for rapid Balancing
        labels = np.argmax(targets, axis=1)
        self.class_indices = [np.where(labels == c)[0] for c in np.unique(labels)]
        self.num_classes = len(self.class_indices)

        if self.balanced and self.num_classes > 0:
            self.samples_per_class = batch_size // self.num_classes
            self.real_batch_size = self.samples_per_class * self.num_classes

            # Logging when batch size cannot be fully utilized for balancing purppose
            if self.real_batch_size != batch_size:
                print(
                    "[DataLoader] Batch size adjusted: "
                    f"{batch_size} -> {self.real_batch_size} \n"
                    "             to maintain 1:1 class balance "
                    f"({self.samples_per_class} samples/class).\n"
                )
        else:
            self.samples_per_class = batch_size  # Not used yet
            self.real_batch_size = batch_size

        if self.samples_per_class == 0:
            raise ValueError(
                f"Batch size {batch_size} is too small for "
                f" {self.num_classes} classes. \n"
                f"Minimum required: {self.num_classes}"
            )

    def _get_balanced_batches(self) -> Generator[np.ndarray, None, None]:
        """Generate indices for batches with an equal number of samples from each class.

        Yields:
            np.ndarray: Shuffled array of indices for a balanced mini-batch.

        """
        for idx_list in self.class_indices:
            self.rng.shuffle(idx_list)

        samples_per_class = self.batch_size // self.num_classes
        # Determine the number of batches based on the biggest class
        # to make sure all data are used
        max_samples = max(len(idx) for idx in self.class_indices)
        num_batches = max_samples // samples_per_class

        for i in range(num_batches):
            batch = []
            for c in range(self.num_classes):
                # Repeate small classes until all data used
                indices = self.class_indices[c]
                start = (i * samples_per_class) % len(indices)

                end = start + samples_per_class
                idx_selection = indices[start:end]

                if len(idx_selection) < samples_per_class:
                    needed = samples_per_class - len(idx_selection)
                    idx_selection = np.concatenate([idx_selection, indices[:needed]])

                batch.extend(idx_selection)

            self.rng.shuffle(batch)
            yield np.array(batch)

    def _get_random_batches(self) -> Generator[np.ndarray, None, None]:
        """Generate indices for batches using standard random shuffling.

        Yields:
            np.ndarray: Array of indices for a standard mini-batch.

        """
        indices = np.arange(len(self.inputs))
        self.rng.shuffle(indices)
        for start_idx in range(0, len(indices), self.batch_size):
            yield indices[start_idx : start_idx + self.batch_size]

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Return an iterator over the dataset batches based on the balanced setting.

        Yields:
            np.ndarray: Batch indices for the current iteration.

        """
        if self.balanced and self.num_classes > 1:
            return self._get_balanced_batches()
        return self._get_random_batches()

    def __len__(self) -> int:
        """Calculate the total number of batches available in one epoch.

        Returns:
            int: The number of mini-batches per epoch.

        """
        if self.balanced and self.num_classes > 0:
            samples_per_class = self.batch_size // self.num_classes
            return max(len(idx) for idx in self.class_indices) // samples_per_class
        return int(np.ceil(len(self.inputs) / self.batch_size))


class FeatureEngine:
    """Component for coordinate transformation and feature expansion.

    This class implements various feature engineering techniques, including
    Cartesian to Polar coordinate conversion, polynomial expansions,
    and trigonometric features to enhance the model's ability to learn
    complex decision boundaries.

    Attributes:
        cfg (NeuraConfig): Configuration object containing feature mode
            and expansion flags.

    """

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initialize the FeatureEngine with specific configuration settings.

        Args:
            cfg (NeuraConfig): Configuration instance holding feature
                engineering hyperparameters.

        """
        self.cfg = cfg

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply feature transformations to the raw input data.

        Args:
            X_raw (np.ndarray): Raw input coordinates of shape (N, 2).

        Returns:
            np.ndarray: Transformed and expanded feature matrix.

        """
        feature_mode = self.cfg.feature_mode

        use_squares = self.cfg.use_squares
        use_interaction = self.cfg.use_interaction
        use_trig = self.cfg.use_trig

        x1: np.ndarray = X_raw[:, 0:1]
        x2: np.ndarray = X_raw[:, 1:2]

        if feature_mode == "cartesian":
            features = [X_raw]

            if use_squares:
                features += [x1**2, x2**2]

            if use_interaction:
                features += [x1 * x2]

            if use_trig:
                features += [np.sin(x1), np.cos(x1), np.sin(x2), np.cos(x2)]

            return np.hstack(features)

        elif feature_mode == "polar":
            # polar coordinates use radius and Sin and Cos for smooth approximation
            r: np.ndarray = np.sqrt(x1**2 + x2**2)
            phi: np.ndarray = np.arctan2(x2, x1)
            sn: np.ndarray = np.sin(phi)
            cs: np.ndarray = np.cos(phi)

            return np.hstack([r, sn, cs])

        else:
            raise ValueError("Unknown Feature Mode")


class DataFactory:
    """Generator for synthetic multi-class datasets.

    Responsible for creating various geometric patterns (spirals, donuts,
    rhodonea curves) used for training and testing the neural network's
    classification performance.

    Attributes:
        cfg (NeuraConfig): Configuration object defining data geometry
            and noise levels.
        rng (np.random.Generator): Seeded random number generator for
            reproducibility.

    """

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initialize the DataFactory with a random number generator.

        Args:
            cfg (NeuraConfig): Configuration instance holding dataset parameters.

        """
        self.cfg = cfg

        # Reproducibility anchor.
        self.rng = np.random.default_rng(seed=self.cfg.seed)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Orchestrates dataset generation based on the configured data mode.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing features (X)
                and one-hot encoded targets (y).

        """
        mode: str = self.cfg.data_mode.lower()

        if mode == "multidonut":
            return self.generate_mdonut()
        elif mode == "spirals":
            return self.generate_spirals()
        elif mode == "rhodonea":
            return self.generate_rhodonea()
        else:
            raise ValueError(f"Unknown Data Mode: {mode}")

    def generate_mdonut(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate a multi-layered donut dataset with concentric rings.

        Returns:
            tuple[np.ndarray, np.ndarray]: Generated features and targets
                for the multi-donut pattern.

        """
        samples: int = self.cfg.samples
        radii: list[float] = sorted(self.cfg.mdonut_radii, reverse=True)
        evenly_dist: bool = self.cfg.mdonut_r_evenly_dist

        max_radius = max(radii) * 1.2
        rho, phi = self._generate_random_polar(max_radius, evenly_dist)

        x_col = rho * np.cos(phi)
        y_col = rho * np.sin(phi)
        X = np.hstack([x_col, y_col])

        y_labels = np.zeros((samples, 1), dtype=int)
        num_classes = 0

        for i, r in enumerate(radii):
            if i % 2 == 0:
                num_classes += 1
                current_fill = num_classes
            else:
                current_fill = 0

            y_labels[rho.flatten() <= r] = current_fill

        num_classes = num_classes + 1
        y = np.eye(num_classes)[y_labels.reshape(-1)]

        X = self._apply_noise(X)

        return X, y

    def generate_spirals(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate an intertwined N-armed spiral dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: Generated features and targets
                for the spiral pattern.

        """
        num_spirals = self.cfg.spiral_num_classes
        samples_per_class = self.cfg.samples // num_spirals
        turns = self.cfg.spiral_turns

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        # Progression Parameter (Radius and Angle)
        t = np.linspace(0, 1, samples_per_class)

        for i in range(num_spirals):
            # Phase shift fore each spiral
            angle_offset = (i * 2 * np.pi) / num_spirals
            theta = t * turns * np.pi + angle_offset

            r = t * self.cfg.spiral_max_radius
            x = np.c_[r * np.cos(theta), r * np.sin(theta)]

            x = self._apply_noise(x)

            # Create One-Hot
            y = np.zeros((samples_per_class, num_spirals))
            y[:, i] = 1

            X_list.append(x)
            y_list.append(y)

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        return self._shuffle_and_return(X, y)

    def generate_rhodonea(self, num_layers: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """Generate a multi-layered Rose (Rhodonea) curve dataset.

        Returns:
            tuple[np.ndarray, np.ndarray]: Generated features and targets
                for the rhodonea pattern.

        """
        num_layers = self.cfg.rhodonea_num_classes
        a = self.cfg.rhodonea_max_radius
        k = self.cfg.rhodonea_k
        samples = self.cfg.samples
        evenly_dist = self.cfg.rhodonea_r_evenly_dist

        max_radius = a * 1.2

        rho, phi = self._generate_random_polar(max_radius, evenly_dist)

        X = np.column_stack((rho * np.cos(phi), rho * np.sin(phi)))

        boundary_r = np.abs(a * np.cos(k * phi))

        # Initialize the target matrix (all — Class 0 / Background)
        num_classes = num_layers + 1
        y_one_hot = np.zeros((samples, num_classes))
        y_one_hot[:, 0] = 1

        # Slice the flower into layers
        # Proceed from the outer layer to the inner one to overwrite values
        for i in range(1, num_classes):
            layer_threshold = boundary_r * (1.0 - (i - 1) / num_layers)

            mask = (rho <= layer_threshold).flatten()

            y_one_hot[mask, :] = 0
            y_one_hot[mask, i] = 1

        X = self._apply_noise(X)

        return self._shuffle_and_return(X, y_one_hot)

    def _generate_random_polar(
        self, max_radius: float, evenly_dist: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly generate polar coordinates (rho, phi).

        Args:
            max_radius: float: Feature matrix.
            evenly_dist: bool: Target matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: rho and phi datasets.

        """
        u = self.rng.uniform(0, 1, (self.cfg.samples, 1))

        # Distribute evenly by Area or Radius
        rho = np.sqrt(u) * max_radius if evenly_dist else u * max_radius
        phi = self.rng.uniform(0, 2 * np.pi, (self.cfg.samples, 1))

        return rho, phi

    def _apply_noise(self, X: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise to the dataset if configured.

        Args:
            X (np.ndarray): The input feature matrix.

        Returns:
            np.ndarray: Feature matrix with added noise.

        """
        if self.cfg.noise > 0:
            X += self.rng.normal(0, self.cfg.noise, X.shape)
        return X

    def _shuffle_and_return(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly shuffle the feature and target matrices in unison.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: Shuffled datasets.

        """
        indices = self.rng.permutation(len(X))
        return X[indices], y[indices]


class DataScaler:
    """Min-Max normalizer for feature scaling.

    Ensures that input features are scaled to a specific range (typically [-1, 1])
    to stabilize the neural network's training process and prevent
    gradient explosion or vanishing.

    Attributes:
        cfg (NeuraConfig): Configuration object defining the target scale range.
        range (tuple): Target min and max values for scaling.
        min_ (np.ndarray): Memorized minimum values per feature.
        max_ (np.ndarray): Memorized maximum values per feature.
        diff (np.ndarray): Computed range (max - min) per feature.

    """

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initialize the DataScaler with target range parameters.

        Args:
            cfg (NeuraConfig): Configuration instance containing feature range.

        """
        self.cfg = cfg
        self.range = self.cfg.feature_range

        self.min_ = None
        self.max_ = None

        self.diff = None

    def fit(self, X: np.ndarray) -> None:
        """Compute and memorizes the scaling parameters from the input data.

        Args:
            X (np.ndarray): Input data used to calculate min and max values.

        """
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.diff = np.where((self.max_ - self.min_) == 0, 1, self.max_ - self.min_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale the input data using previously memorized parameters.

        Args:
            X (np.ndarray): Input data to be scaled.

        Returns:
            np.ndarray: Scaled feature matrix.

        """
        if self.min_ is None or self.max_ is None:
            raise RuntimeError(
                "DataScaler error: The scaler has not been fitted yet. "
                "Call fit() or fit_transform() before calling transform()."
            )

        X_std = (X - self.min_) / self.diff
        X_scaled = X_std * (self.range[1] - self.range[0]) + self.range[0]
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Learn scaling parameters and transforms the data in a single step.

        Args:
            X (np.ndarray): Input data for both fitting and transformation.

        Returns:
            np.ndarray: Scaled feature matrix.

        """
        self.fit(X)
        return self.transform(X)
