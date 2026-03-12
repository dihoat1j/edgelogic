# EdgeLogic

EdgeLogic is a specialized neural network inference framework designed for game engines and edge devices where memory and CPU cycles are at a premium. It focuses on 8-bit quantized execution of feed-forward networks, ideal for game agent behavior trees, NPC sensors, and procedural content generation.

## Key Features

*   Int8 Quantized Inference: Drastically reduces memory footprint and increases throughput on CPUs.
*   Zero-Dependency Runtime: The core engine is self-contained with minimal overhead.
*   Game Engine Ready: Designed to be easily integrated into update loops (Unity, Godot, or custom C++ engines).
*   Agent-Centric: Optimized for small, high-frequency "brain" models rather than massive LLMs.

## Installation

```bash
pip install edgelogic
```

## Quick Start

### 1. Training and Exporting
```python
import numpy as np
from edgelogic.compiler import ModelCompiler

# Define a simple agent model (layer_sizes)
model_data = {
    "weights": [np.random.randn(10, 4), np.random.randn(4, 2)],
    "biases": [np.zeros(4), np.zeros(2)]
}

compiler = ModelCompiler()
quantized_model = compiler.compile(model_data)
quantized_model.save("agent_brain.edge")
```

### 2. Runtime Inference
```python
from edgelogic.runtime import InferenceEngine

engine = InferenceEngine("agent_brain.edge")
observation = np.array([0.5, -0.1, 0.9, 0.1, 0.0, 1.0, -0.5, 0.2, 0.3, -0.1])
action = engine.predict(observation)
print(f"Agent Action: {action}")
```

## Architecture

EdgeLogic uses Symmetric Quantization ($Q = round(f / scale)$).
*   Weights: Quantized to static scales per layer.
*   Activations: Dynamic scaling or pre-computed calibration.
*   Layers: Supports Dense (Linear) and ReLU activations.

## Performance
On a standard ARM-based mobile device, EdgeLogic can execute a 3-layer MLP in under 15 microseconds, allowing hundreds of agents to run simultaneously without dropping frames.
