import numpy as np

from src.config import NeuraConfig


class FeatureEngine:
    """Transform input data.

    X_raw: input array with shape data (N, 2), columns are x1, x2 - cartesian
    [r, phi] - polar
    """

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initiate parameters for the Feature Engine class."""
        self.cfg = cfg

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Transfor X_raw data and return modified data."""
        feature_mode = self.cfg.feature_mode

        use_squares = self.cfg.use_squares
        use_interaction = self.cfg.use_interaction
        use_trig = self.cfg.use_trig

        x1 = X_raw[:, 0:1]
        x2 = X_raw[:, 1:2]

        if feature_mode == "cartesian":
            features = [X_raw]

            if use_squares:
                features.append(x1**2)
                features.append(x2**2)

            if use_interaction:
                features.append(x1 * x2)

            if use_trig:
                features.append(np.sin(x1))
                features.append(np.cos(x1))
                features.append(np.sin(x2))
                features.append(np.cos(x2))

            return np.hstack(features)

        elif feature_mode == "polar":
            # polar coordinates use radius and Sin and Cos for smooth approximation
            r = np.sqrt(x1**2 + x2**2)
            phi = np.arctan2(x2, x1)
            sn = np.sin(phi)
            cs = np.cos(phi)

            return np.hstack([r, sn, cs])

        else:
            raise ValueError("Unknown Feature Mode")


class DataFactory:
    """Generate input data with parameters."""

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initiate parameters for the DataFactory class."""
        self.cfg = cfg

        # Reproducibility anchor.
        self.rng = np.random.default_rng(seed=self.cfg.seed)

    def generate(self) -> tuple:
        """Generate dataset by choosing a mode."""
        mode = self.cfg.data_mode.lower()

        if mode == "multidonut":
            return self.generate_mdonut()
        elif mode == "spirals":
            return self.generate_spirals()
        elif mode == "rhodonea":
            return self.generate_rhodonea()
        else:
            raise ValueError(f"Unknown Data Mode: {mode}")

    def generate_mdonut(self) -> tuple:
        """Generate points, multiple classes - shape Target.

        Points can be distributed evenly along the radius, or closer to the center.
        """
        samples = self.cfg.samples
        noise = self.cfg.noise
        radii = sorted(self.cfg.mdonut_radii, reverse=True)
        evenly_dist = self.cfg.mdonut_r_evenly_dist

        max_radius = max(radii) * 1.2
        u = self.rng.uniform(0, 1, (samples, 1))

        if evenly_dist:
            rho = np.sqrt(u) * max_radius
        else:
            rho = u * max_radius

        phi = self.rng.uniform(0, 2 * np.pi, (samples, 1))

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

        if noise > 0:
            X += self.rng.normal(0, noise, X.shape)

        return X, y

    def generate_spirals(self) -> tuple:
        """Generate dataset for N intertwined spirals with One-Hot encoding."""
        num_spirals = self.cfg.num_spirals
        samples_per_class = self.cfg.samples // num_spirals
        noise = self.cfg.noise
        turns = self.cfg.spiral_turns

        X_list = []
        Y_list = []

        # Progression Parameter (Radius and Angle)
        t = np.linspace(0, 1, samples_per_class)

        for i in range(num_spirals):
            # Phase shift fore each spiral
            angle_offset = (i * 2 * np.pi) / num_spirals
            theta = t * turns * np.pi + angle_offset

            r = t * self.cfg.spiral_max_radius
            x = np.c_[r * np.cos(theta), r * np.sin(theta)]

            # Add noise
            if noise > 0:
                x += self.rng.normal(0, noise, x.shape)

            # Create One-Hot
            y = np.zeros((samples_per_class, num_spirals))
            y[:, i] = 1

            X_list.append(x)
            Y_list.append(y)

        X = np.vstack(X_list)
        Y = np.vstack(Y_list)

        return self._shuffle_and_return(X, Y)

    def generate_rhodonea(self, num_layers: int = 2) -> tuple:
        """Generate Points inside a Rose (Rhodonea curve) with multiple layers.

        Layers:
        - Class 0: Background (Outside the flower)
        - Class 1 to num_layers: Concentric zones inside the petals
        """
        num_layers = self.cfg.rhodonea_NumOfLayers
        a = self.cfg.rhodonea_max_radius
        k = self.cfg.rhodonea_k
        samples = self.cfg.samples
        noise = self.cfg.noise
        evenly_dist = self.cfg.rhodonea_r_evenly_dist

        max_radius = a * 1.2
        u = self.rng.uniform(0, 1, (samples, 1))
        if evenly_dist:
            rho = np.sqrt(u) * max_radius
        else:
            rho = u * max_radius

        phi = self.rng.uniform(0, 2 * np.pi, (samples, 1))

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

        if noise > 0:
            X += self.rng.normal(0, noise, X.shape)

        return self._shuffle_and_return(X, y_one_hot)

    def _shuffle_and_return(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Shuffle data. Support method."""
        indices = self.rng.permutation(len(X))
        return X[indices], y[indices]


class DataScaler:
    """Scale data using Min-Max to fit into (-1;1) range."""

    def __init__(self, cfg: NeuraConfig) -> None:
        """Initiate parameters for the DataScaler class."""
        self.cfg = cfg
        self.range = self.cfg.feature_range

        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> None:
        """Memorize scale parameters for data X when scaling to min-max range."""
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.diff = np.where((self.max_ - self.min_) == 0, 1, self.max_ - self.min_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data X to min-max range memorized with fit method."""
        X_std = (X - self.min_) / self.diff
        X_scaled = X_std * (self.range[1] - self.range[0]) + self.range[0]
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Call fit function to memorize scale parameters and fit data to range."""
        self.fit(X)
        return self.transform(X)
