# Changelog

All notable changes to the "NeuraKitten" project will be documented in this file.  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.2.0] - 2026-04-08

### Added
- Added option Rose (Rhodonea curve) for generating datasets. 
- Added `src/config.py` with `NeuraConfig` dataclass supporting parameters for Network architecture, Hyperparameters, ADAM, Generating data, Feature Engineering and UI/UX.

### Fixed
- Cleaned up all classes to adapt a single `config.py` architectre.


## [1.1.0] - 2026-04-06
### Added
- Functional **ADAM Optimizer** with bias correction.
- **Time-based Learning Rate Decay** to stabilize convergence in late training stages.
- Added a toggle for point distribution in the "Donut" generator: Area-proportional (uniform density) vs. Radius-proportional (concentrated at the center).

### Changed
- Gradient descent mathematics refactor: shifted to classical notation `(predictions - targets)`.
- Updated parameter update logic from `+=` to `-=` to align with standard Gradient Descent conventions.
- Optimized `fit` function: removed batch-size-dependent LR scaling in favor of ADAM's adaptive nature.

### Fixed
- Fixed weight update order: error backpropagation now uses weights from the current forward pass before they are updated by ADAM.
- Eliminated late-epoch oscillations by implementing a `decay_rate` decay rate of `0.01` and a `min_lr` floor of `1e-5`.


## [1.0.0] - 2026-04-01
### Added
- Initial architecture of a Multi-Layer Perceptron (MLP) using NumPy.
- Core activation functions: Sigmoid and Leaky ReLU (including derivatives).
- Weight initialization using **He Initialization** method.
- Standard Backpropagation algorithm using Mean Squared Error (MSE) loss.
- Real-time training visualization (Live Plot) for the "Donut" and "Spirals" classification task.
- Synthetic data generator for circular and ring-based point distributions.