# NeuraKitten
A lightweight, NumPy-based **Multi-class** Multi-Layer Perceptron (MLP) implementation built from scratch for educational purposes.

This project demonstrates how a neural network can learn to classify complex, non-linear multi-class patterns. By leveraging **Feature Engineering**, NeuraKitten transforms standard Cartesian coordinates into Polar coordinates to achieve high-precision boundary classification even with a compact architecture.

![Neural Network Decision Boundary Animation](demo.png)

## Key Features
- **Multi-class Data Factory**: Generates synthetic "Donut", "N-armed Spiral", or "Multi-layered Rose" datasets with an arbitrary number of labels.
- **Feature Engineering Engine**: Implements transformations including Polar coordinates $(r, sin(\phi), cos(\phi))$, polynomial features $(x^2, y^2, x \cdot y)$, and trigonometric expansions.
- **Why Polar?** By linearizing circular and spiral boundaries, we allow the network to achieve high accuracy significantly faster than using raw $(x, y)$ data.
- **Softmax Classification**: Proper multi-class probabilistic output, ensuring the sum of all class probabilities equals 1.0.
- **Live Visualization**: Real-time plotting of discrete decision boundaries and loss convergence with high-resolution grid mapping.
- **Advanced Optimization**: Implements the ADAM (Adaptive Moment Estimation) optimizer with numerical stability tricks like `epsilon` clipping and `shift_x` Softmax.

## Mathematical Foundations
- **Optimization**: ADAM (Adaptive Moment Estimation) with time-based learning rate decay.  
- **Initialization**: He Initialization for stable variance in deep architectures.  
- **Activation**: Leaky ReLU for hidden layers and Softmax for the output layer.  
- **Loss Function**: Categorical Cross-Entropy for robust multi-class training. 

## Project Structure
```text
NeuraKitten/
├── src/
│   ├── config.py           # Hyperparameters and data geometry settings.
│   ├── data_utils.py       # DataFactory, FeatureEngine, DataScaler
│   ├── model.py            # DeepNeuralNetwork class
│   ├── trainer.py          # Training loop logic (fit)
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

### Example Configuration.
You can customize the entire pipeline from data geometry to neural architecture using the `NeuraConfig` dataclass.


```
from src.config import NeuraConfig
from src.data_utils import DataFactory, DataScaler, FeatureEngine
from src.model import DeepNeuralNetwork
from src.trainer import fit


def main() -> None:
    """Run the main entry point of the script."""

    cfg = NeuraConfig(
        epochs=501,
        hidden_layers=[32, 24, 16, 12],
        samples=2048,
        batch_size=32,
        data_mode="spirals",  # "spirals", "rhodonea", "multidonut"
        spiral_max_radius=2.0,
        num_spirals=5,
        spiral_turns=5,
        noise=0.03,
        feature_mode="polar",
        view_range=2.5,
        initial_lr=0.001,
        decay_rate=0.01,
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