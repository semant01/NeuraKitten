# NeuraKitten
A lightweight, NumPy-based Multi-Layer Perceptron (MLP) implementation built from scratch for educational purposes.

This project demonstrates how a simple neural network can classify complex non-linear patterns. By leveraging **Feature Engineering**, NeuraKitten transforms standard Cartesian coordinates into Polar coordinates to achieve high-precision boundary classification with minimal architecture.

![Neural Network Decision Boundary Animation](demo.png)

## Key Features
- **Custom Data Factory**: Generates synthetic "Donut", "Spiral" or "Rhodonea curve" datasets with adjustable noise and density.
- **Feature Engineering Engine**: Implements transformations including Polar coordinates $(r, sin(\phi), cos(\phi))$, polynomial features $(x^2, y^2, x \cdot y)$, and trigonometric expansions.
- **Why Polar?** By linearizing circular boundaries, we allow a simpler network to achieve 99%+ accuracy significantly faster than using raw $(x, y)$ data.
- **Live Visualization**: Real-time plotting of decision boundaries and loss convergence during the training process.
- **Modular Architecture**: Clean separation between model logic, data processing, and training loops.
- **Advanced Optimization**: Implements the ADAM (Adaptive Moment Estimation) optimizer with bias correction for faster and more stable convergence.
- **Comprehensive Versioning**: Detailed project evolution tracked in CHANGELOG.md following SemVer standards.

## Mathematical Foundations
- **Optimization**: ADAM (Adaptive Moment Estimation) with time-based learning rate decay.  
- **Initialization**: He Initialization for stable variance in deep networks.  
- **Activation**: Leaky ReLU for hidden layers and Sigmoid for the output layer.  
- **Loss Function**: Mean Squared Error (MSE).  

## Project Structure
```text
NeuraKitten/
├── src/
│   ├── config.py           # Configuration parameters for Network architecture, Hyperparameters, etc.
│   ├── data_utils.py       # DataFactory, FeatureEngine, DataScaler
│   ├── model.py            # DeepNeuralNetwork class
│   ├── trainer.py          # Training loop logic
│   └── visualization.py    # Live plotting with Matplotlib
├── main.py                 # Entry point
├── requirements.txt        # Project dependencies
├── CHANGELOG.md            # Revision history
└── README.md               # Documentation
```

## Getting Started
### Prerequisites
    Python 3.8+
    pip (Python package manager)


### Installation
#### 1. Clone the repository:
    git clone https://github.com/semant01/NeuraKitten.git
    cd NeuraKitten

#### 2. Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate   # Windows

#### 3. Install dependencies:
    pip install -r requirements.txt


## Running the Project
Simply execute the main script to start the training and visualization:
```
python main.py
```

### Configuration.
You can customize the entire pipeline from data geometry to neural architecture using the `NeuraConfig` dataclass.


```
import numpy as np

from src.config import NeuraConfig
from src.data_utils import DataFactory, DataScaler, FeatureEngine
from src.model import DeepNeuralNetwork
from src.trainer import fit


def main() -> None:
    """Run the main entry point of the script."""

    cfg = NeuraConfig(
        hidden_layers=[16, 16],
        samples=2048,
        batch_size=16,
        data_mode="rhodonea",  # "donut", "spiral", "rhodonea"
        rose_k=2.5,
        noise=0.03,
        feature_mode="polar",
        color_gradient=False,
        show_levels=True,
        show_dataset_points=True,
    )

    # 1. Generate the dataset based on cfg.data_mode
    factory = DataFactory(cfg)
    X_raw, targets = factory.generate()

    # 2. Feature Engineering, transform coordinates (e.g., Cartesian to Polar)
    engine = FeatureEngine(cfg)
    X_featured = engine.transform(X_raw)

    # 3. Scaling
    # Normalizes data to [-1, 1] range
    scaler = DataScaler(cfg)
    X_transformed = scaler.fit_transform(X_featured)

    # 4. Initialize the MLP
    # Dimensions are detected automatically from the processed features
    input_dim = X_transformed.shape[1]
    output_dim = targets.shape[1]
    hidden_layers = cfg.hidden_layers

    brain = DeepNeuralNetwork(
        config=cfg, layers_size=[input_dim] + hidden_layers + [output_dim]
    )

    # 5. Run Training & Visualization
    fit(
        model=brain,
        inputs=X_transformed,
        targets=targets,
        cfg=cfg,
        X_raw=X_raw,  # Original points for plotting
        scaler=scaler,  # Required to de-scale boundaries
        engine=engine,  # Required to re-feature decision grid
    )
```

## Tech Stack
- *NumPy*: For high-performance matrix operations.
- *Matplotlib*: For real-time training visualization.
- *Python*: Clean, object-oriented implementation.

## License
Distributed under the MIT License.