# Changelog

All notable changes to the "NeuraKitten" project will be documented in this file.  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.3.0] - 2026-04-21

### Added
- Integrated Fisher's Iris dataset via `sklearn.datasets` into the `DataFactory` class.
- Developed an internal `_manual_pca` method using eigenvector decomposition for 2D dimensionality reduction, providing an alternative to `sklearn.decomposition`.
- Added `x_min/max` and `y_min/max` parameters to `NeuraConfig` for adaptive visualization boundaries.
- Added `vis_axes` parameter to `NeuraConfig` for visualization of selected axis.

### Changed
- Refactored `visualization.py` to utilize dynamic coordinate bounds instead of a fixed view_range.
- Optimized `requirements.txt` to include `scikit-learn` while removing redundant sub-dependencies.

### Fixed
- Fixed an issue where data points outside the `data_range` were not rendered on the main chart.


## [2.2.0] - 2026-04-21

### Added
- Created `ExperimentContext` dataclass to centralize runtime metadata. Integrated `ExperimentContext` into the `NeuraPipeline` and `fit` loop.
- Implemented `GridSpec` to create a multi-pane dashboard (Main Plot, Metrics, and Info).

### Changed
- Refactored project structure: moved `NeuraConfig` and `ExperimentContext` to a dedicated `src/structures.py` file for better modularity (replaces `config.py`).
- Optimized UI performance by moving layout and axis initialization outside the training loop.
- Updated `README.md` to reflect the new project structure and visualization features.

### Fixed
- Resolved a critical memory leak and window "jumping" caused by redundant axis creation.


## [2.1.0] - 2026-04-12

### Added
- Created `NeuraDataLoader` class to encapsulate batching logic, supporting both standard and balanced data distribution.
- Added `balanced_batches` flag to `NeuraConfig`, allowing global control over class balancing strategies from a single entry point.
- Implemented automated batch size adjustment with informative logging to maintain perfect class ratios during training.
- Implemented comprehensive Python type hints and Google-style docstrings across all modules to enhance maintainability and IDE support.

### Changed
- Streamlined `main.py` by delegating data processing and training orchestration to the centralized `NeuraPipeline`.
- Refactored Core Architecture: Decomposed the `DeepNeuralNetwork` class to separate concerns and improve code readability without altering the underlying mathematical approach.
- Redesigned `trainer.py`, `data_utils.py` and `visualization.py` to minimize computational overhead and provide a cleaner, more modular code structure.
- Enhanced `NeuraDataLoader` with an oversampling mechanism to prevent data loss in minority classes while maintaining strict balance.
- Migrated from global `np.random` state to isolated `Generator` instances (PCG64) to ensure robust experiment reproducibility.


### Fixed
- Unified `epsilon` usage. The same constant is now consistently applied to prevent division by zero in the ADAM optimizer and to stabilize Categorical Cross-Entropy loss calculations via `np.clip`.


## [2.0.0] - 2026-04-11

### Added
- Multi-class Support: Data generators now support an arbitrary number of classes (e.g., multi-layered roses, N-armed spirals).
- One-Hot Encoding: Integrated automatic One-Hot labels generation for all datasets.
- Categorical Cross-Entropy: Added as the primary loss function for multi-class optimization.

### Changed
- Architecture Unified: Replaced Sigmoid with Softmax activation in the output layer for all models.
- Visualizer Upgrade: `live_plot` now handles discrete decision boundaries for $N$ classes with high-resolution grid mapping (`resolution=300` by default in `config.py`).
- Standardized Pipeline: All internal logic now defaults to a multi-class workflow, simplifying the API.

### Fixed
- Numerical Stability: Implemented `shift_x` in Softmax and `epsilon` clipping in Cross-Entropy to prevent `NaN` values.
- Backprop Delta: Corrected the output layer gradient calculation for the Softmax/Cross-Entropy pair.


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