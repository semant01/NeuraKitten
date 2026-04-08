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

        if mode == "donut":
            return self.generate_donut()
        elif mode == "spiral":
            return self.generate_spiral()
        elif mode == "rhodonea":
            return self.generate_rhodonea()
        else:
            raise ValueError(f"Unknown Data Mode: {mode}")

    def generate_spiral(self) -> tuple:
        """Generate dataset for two intertwined spirals."""
        n = self.cfg.samples // 2
        noise = self.cfg.noise
        turns = self.cfg.spiral_turns

        # The parameter *t* determines the radius and the angle simultaneously.
        t = np.linspace(0, 1, n)
        theta = t * turns * np.pi  # Number of turns

        # Spiral 1 (Class 0)
        r1 = t * 1.0
        x1 = np.c_[r1 * np.cos(theta), r1 * np.sin(theta)]
        x1 += self.rng.normal(0, noise, x1.shape)
        y1 = np.zeros((n, 1))

        # Spiral 2 (Class 1) — Rotated 180 degrees
        r2 = t * 1.0
        x2 = np.c_[-r2 * np.cos(theta), -r2 * np.sin(theta)]
        x2 += self.rng.normal(0, noise, x2.shape)
        y2 = np.ones((n, 1))

        return self._shuffle_and_return(np.vstack([x1, x2]), np.vstack([y1, y2]))

    def generate_donut(self) -> tuple:
        """Generate Points inside the ring and outside/inside the hole (0).

        Points can be distributed evenly along the radius, or closer to the center.
        """
        samples = self.cfg.samples
        noise = self.cfg.noise
        inner = self.cfg.donut_r_inner
        outer = self.cfg.donut_r_outer
        evenly_dist = self.cfg.donut_r_evenly_dist

        max_radius = outer + (outer - inner)
        u = self.rng.uniform(0, 1, (samples, 1))

        if evenly_dist:
            rho = np.sqrt(u) * max_radius
        else:
            rho = u * max_radius

        phi = self.rng.uniform(0, 2 * np.pi, (samples, 1))

        x_col = rho * np.cos(phi)
        y_col = rho * np.sin(phi)
        X = np.hstack([x_col, y_col])

        y = np.logical_and(rho >= inner, rho <= outer).astype(float).reshape(-1, 1)
        if noise > 0:
            X += self.rng.normal(0, noise, X.shape)

        return X, y

    def generate_rhodonea(self) -> tuple:
        """Generate Points inside a Rose (Rhodonea curve).

        a - maximum radius
        k - number of petals (2k petals will be generated)

        """
        a = self.cfg.rose_a
        k = self.cfg.rose_k
        samples = self.cfg.samples
        noise = self.cfg.noise

        max_radius = a * 1.3

        u = self.rng.uniform(0, 1, (samples, 1))
        rho = np.sqrt(u) * max_radius
        phi = self.rng.uniform(0, 2 * np.pi, (samples, 1))

        x_col = rho * np.cos(phi)
        y_col = rho * np.sin(phi)
        X = np.hstack([x_col, y_col])

        boundary_r = np.abs(a * np.cos(k * phi))
        y = (rho <= boundary_r).astype(float).reshape(-1, 1)

        if noise > 0:
            X += self.rng.normal(0, noise, X.shape)

        return X, y

    def generate_xor(self) -> tuple:
        """Generate classic XOR: different coordinate signs."""
        samples = self.cfg.samples
        X = self.rng.uniform(-1, 1, (samples, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)
        return X, y

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
