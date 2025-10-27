# Micrograd-Engine

My implementation of micrograd engine.

## Key Features & Benefits

- **Value Class:** Core data structure that stores a scalar value, its gradient, and the computation graph used for automatic differentiation.
- **Mathematical Operations:** Supports addition, subtraction, multiplication, division, exponentiation, and activation functions (tanh, exp) with gradient tracking.
- **Backpropagation:** Implements a full reverse-mode autodiff system by building a topological graph and performing gradient propagation through `_backward` functions.
- **Neuron Class:** Represents a single neuron with randomly initialized weights and bias, computes the weighted sum of inputs, and applies a tanh activation.
- **Layer Class:** Comprises multiple neurons forming a single layer; handles forward propagation for each neuron and manages their parameters collectively.
- **MLP Class:** Defines a multi-layer perceptron built from multiple `Layer` instances; supports flexible architectures through configurable layer sizes.
- **Training Loop:** Demonstrates forward and backward passes, gradient resets, and parameter updates to train the MLP using a simple squared loss function.
- **Visualization Support:** Designed to work with a `draw_dot()` helper function to visualize the computation graph for educational and debugging purposes.

## Prerequisites & Dependencies

- Python 3.x
- numpy
- matplotlib
- torch

Install the dependencies using pip:

```bash
pip install numpy matplotlib torch
```

## Installation & Setup Instructions

1.  Clone the repository:

    ```bash
    git clone https://github.com/saady789/Micrograd-Engine.git
    cd Micrograd-Engine
    ```

2.  No further installation is required. The `engine.py` file contains the implementation, and `torch_compare.py` can be run to verify the results.

## Usage Examples & API Documentation

### Value Class

```python
from engine import Value

# Create Value objects
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'

# Perform backpropagation
L.backward()

# Access data and gradient
print(f"Data: {L.data}, Gradient: {L.grad}")
print(f"Data: {d.data}, Gradient: {d.grad}")
print(f"Data: {c.data}, Gradient: {c.grad}")
print(f"Data: {f.data}, Gradient: {f.grad}")

```

### Running the `torch_compare.py` script

This script compares the gradients computed by `engine.py` with those computed by PyTorch.

```bash
python torch_compare.py
```

This will print the output from PyTorch and the calculated gradients. Compare the PyTorch values to values calculated with your engine.

## Configuration Options

There are no specific configuration options for this project, as it is a basic implementation. You can modify the parameters in `engine.py` or `torch_compare.py` to test different scenarios.

## Contributing Guidelines

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Commit your changes with clear, descriptive commit messages.
4.  Push your changes to your fork.
5.  Submit a pull request.

## License Information

License not specified. All rights reserved.

## Acknowledgments

This project is based on the micrograd engine concepts.
