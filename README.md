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
│   ├── config.py         # Single source of truth for all hyperparameters and settings.
│   ├── data_utils.py     # Dataset generation, feature engineering, and scaling tools.
│   ├── model.py          # Core Neural Network and ADAM implementation using NumPy.
│   ├── pipeline.py       # Orchestrator for the experiment lifecycle and component flow.
│   ├── trainer.py        # Training loop management and dynamic Learning Rate logic.
│   └── visualization.py  # Real-time decision boundary plotting and live metrics.
├── main.py               # Entry point to trigger the NeuraPipeline.
├── requirements.txt      # Project dependencies and environment requirements.
├── CHANGELOG.md          # Record of all notable changes and refactoring history.
└── README.md             # Project documentation and usage guide.
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
from src.pipeline import NeuraPipeline


def main() -> None:
    cfg = NeuraConfig(
        epochs=501,
        hidden_layers=[32, 24, 16, 12],
        samples=2048,
        batch_size=32,
        balanced_batches=True,
        data_mode="rhodonea",
        rhodonea_k=3,
        rhodonea_num_classes=5,
        feature_mode="polar",
        initial_lr=0.001,
        decay_rate=0.01,
        use_interaction=True,
        use_squares=True,
        use_trig=True,
    )

    pipeline = NeuraPipeline(cfg)
    pipeline.run()
```

## Tech Stack
- *NumPy*: For high-performance matrix operations.
- *Matplotlib*: For real-time training visualization.
- *Python*: Clean, object-oriented implementation.

## License
Distributed under the MIT License.