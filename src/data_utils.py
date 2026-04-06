import numpy as np


class FeatureEngine:
    """Transform input data.

    X_raw: input array with shape data (N, 2), columns are x1, x2 - cartesian
    [r, phi] - polar
    """

    def __init__(
        self,
        use_squares: bool = False,  # cartesian only
        use_interaction: bool = False,  # cartesian only
        use_trig: bool = False,  # cartesian only
        mode: str = "cartesian",  # polar
    ) -> None:
        """Initiate parameters for the Feature Engine class."""
        # 'cartesian' -> [x, y]
        # 'polar'     -> [r_norm, phi_norm]
        self.use_squares = use_squares
        self.use_interaction = use_interaction
        self.use_trig = use_trig
        self.mode = mode

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """Transfor X_raw data and return modified data."""
        x1 = X_raw[:, 0:1]
        x2 = X_raw[:, 1:2]

        if self.mode == "cartesian":
            features = [X_raw]

            if self.use_squares:
                features.append(x1**2)
                features.append(x2**2)

            if self.use_interaction:
                features.append(x1 * x2)

            if self.use_trig:
                features.append(np.sin(x1))
                features.append(np.cos(x1))
                features.append(np.sin(x2))
                features.append(np.cos(x2))

            return np.hstack(features)

        elif self.mode == "polar":
            # polar coordinates use radius and Sin and Cos for smooth approximation
            r = np.sqrt(x1**2 + x2**2)
            phi = np.arctan2(x2, x1)
            sn = np.sin(phi)
            cs = np.cos(phi)

            return np.hstack([r, sn, cs])

        else:
            raise ValueError("Unknown mode")


class DataFactory:
    """Generate input data with parameters."""

    def __init__(
        self, samples: int = 2000, noise: float = 0.05, seed: None = None
    ) -> None:
        """Initiate parameters for the DataFactory class."""
        self.samples = samples
        self.noise = noise
        if seed is not None:
            np.random.seed(seed)

    def generate_spiral(self, turns: float = 2.5) -> tuple:
        """Generate dataset for two intertwined spirals."""
        n = self.samples // 2

        # The parameter *t* determines the radius and the angle simultaneously.
        t = np.linspace(0, 1, n)
        theta = t * turns * np.pi  # Number of turns

        # Spiral 1 (Class 0)
        r1 = t * 1.0
        x1 = np.c_[r1 * np.cos(theta), r1 * np.sin(theta)]
        x1 += np.random.normal(0, self.noise, x1.shape)
        y1 = np.zeros((n, 1))

        # Spiral 2 (Class 1) — Rotated 180 degrees
        r2 = t * 1.0
        x2 = np.c_[-r2 * np.cos(theta), -r2 * np.sin(theta)]
        x2 += np.random.normal(0, self.noise, x2.shape)
        y2 = np.ones((n, 1))

        return self._shuffle_and_return(np.vstack([x1, x2]), np.vstack([y1, y2]))

    def generate_donut(
        self, inner: float = 0.3, outer: float = 0.7, evenly: bool = False
    ) -> tuple:
        """Generate Points inside the ring and outside/inside the hole (0).

        Points can be distributed evenly along the radius, or closer to the center.
        """
        max_radius = outer + (outer - inner)
        u = np.random.uniform(0, 1, (self.samples, 1))

        if evenly:
            rho = np.sqrt(u) * max_radius
        else:
            rho = u * max_radius

        phi = np.random.uniform(0, 2 * np.pi, (self.samples, 1))

        x_col = rho * np.cos(phi)
        y_col = rho * np.sin(phi)
        X = np.hstack([x_col, y_col])

        y = np.logical_and(rho >= inner, rho <= outer).astype(float).reshape(-1, 1)
        if self.noise > 0:
            X += np.random.normal(0, self.noise, X.shape)

        return X, y

    def generate_xor(self) -> tuple:
        """Generate classic XOR: different coordinate signs."""
        X = np.random.uniform(-1, 1, (self.samples, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)
        return X, y

    def _shuffle_and_return(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Shuffle data. Support method."""
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]


class DataScaler:
    """Scale data using Min-Max to fit into (-1;1) range."""

    def __init__(self, feature_range: tuple = (-1, 1)) -> None:
        """Initiate parameters for the DataScaler class."""
        self.min_ = None
        self.max_ = None
        self.range = feature_range

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
