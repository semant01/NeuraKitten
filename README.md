# NeuraKitten
A lightweight, NumPy-based Multi-Layer Perceptron (MLP) implementation built from scratch for educational purposes.

This project demonstrates how a simple neural network can classify complex non-linear patterns. By leveraging **Feature Engineering**, NeuraKitten transforms standard Cartesian coordinates into Polar coordinates to achieve high-precision boundary classification with minimal architecture.

![Neural Network Decision Boundary Animation](demo.png)

## Key Features
* **Custom Data Factory**: Generates synthetic "Donut" and "Spiral" datasets with adjustable noise and density.
* **Feature Engineering Engine**: Implements transformations including Polar coordinates $(r, \phi)$, polynomial features $(x^2, y^2, x \cdot y)$, and trigonometric expansions.
* **Why Polar?** By linearizing circular boundaries, we allow a simpler network to achieve 99%+ accuracy significantly faster than using raw $(x, y)$ data.
* **Live Visualization**: Real-time plotting of decision boundaries and loss convergence during the training process.
* **Modular Architecture**: Clean separation between model logic, data processing, and training loops.


## Project Structure
```text
NeuraKitten/
├── src/
│   ├── data_utils.py       # DataFactory, FeatureEngine, DataScaler
│   ├── model.py            # DeepNeuralNetwork class
│   ├── trainer.py          # Training loop logic
│   └── visualization.py    # Live plotting with Matplotlib
├── main.py                 # Entry point
├── requirements.txt        # Project dependencies
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
You can customize the training behavior in `main.py`:

```
import numpy as np

from src.data_utils import DataFactory, DataScaler, FeatureEngine
from src.model import DeepNeuralNetwork
from src.trainer import fit


def main():
    np.random.seed(42)

    # 1. Generate the dataset
    # You can choose between factory.generate_spiral() or factory.generate_donut()
    factory = DataFactory(samples=512, noise=0.05)
    X_raw, targets = factory.generate_spiral(turns=2.5)

    # 2. Feature Engineering
    # Experiment with "polar" mode or add squares/trig features in "cartesian" mode
    engine = FeatureEngine(
        mode="cartesian", use_squares=False, use_interaction=False, use_trig=False
    )
    X_featured = engine.transform(X_raw)

    # 3. Scaling
    # Normalizes data to [-1, 1] range, essential for neural network convergence
    scaler = DataScaler(feature_range=(-1, 1))
    X_transformed = scaler.fit_transform(X_featured)

    # 4. Initialize the MLP
    # Define architecture: [Input Nodes, Hidden Layers..., Output Nodes]
    # Input nodes are calculated automatically based on your features
    input_dim = X_transformed.shape[1]
    output_dim = targets.shape[1]
    hidden_layers = [12, 12]
    brain = DeepNeuralNetwork([input_dim] + hidden_layers + [output_dim])

    # 5. Run Training & Visualization
    fit(
        model=brain,
        engine=engine,
        inputs=X_transformed,
        targets=targets,
        scaler=scaler,
        X_raw=X_raw,
        epochs=20001,  # Total training iterations
        batch_size=128,  # Number of samples per gradient update
        base_lr=0.03,  # Initial learning rate
        lr_gradient=0.9998,  # Smooth learning rate decay
        visualize=True,  # Toggle real-time Matplotlib animation
        color_gradient=False,  # Use solid colors or probability gradients
        show_dataset_points=True,  # Show/hide original data points
    )
```

## Tech Stack
- *NumPy*: For high-performance matrix operations.
- *Matplotlib*: For real-time training visualization.
- *Python*: Clean, object-oriented implementation.

## License
Distributed under the MIT License.