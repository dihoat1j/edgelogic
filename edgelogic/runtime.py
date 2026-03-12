import numpy as np
import struct
import os

class InferenceEngine:
    """Executes quantized inference for agent decision loops."""
    
    def __init__(self, model_path: str):
        self.layers = self._load_binary(model_path)

    def _load_binary(self, path: str):
        layers = []
        with open(path, 'rb') as f:
            num_layers_raw = f.read(4)
            if not num_layers_raw: return []
            num_layers = struct.unpack('<I', num_layers_raw)[0]
            
            for _ in range(num_layers):
                rows, cols, w_scale, b_scale = struct.unpack('<IIff', f.read(16))
                w_size = rows * cols
                weights = np.frombuffer(f.read(w_size), dtype=np.int8).reshape(rows, cols)
                biases = np.frombuffer(f.read(cols), dtype=np.int8)
                layers.append({
                    'W': weights, 'b': biases,
                    'ws': w_scale, 'bs': b_scale
                })
        return layers

    def predict(self, input_vec: np.ndarray) -> np.ndarray:
        """Forward pass using optimized int8 matrix multiplication."""
        x = input_vec
        for i, layer in enumerate(self.layers):
            # Dequantize for intermediate calc (simulating hardware SIMD behavior)
            # In a full C++ impl, we would use fixed-point arithmetic
            W = layer['W'].astype(np.float32) * layer['ws']
            b = layer['b'].astype(np.float32) * layer['bs']
            
            x = np.dot(x, W) + b
            
            # Activation: ReLU for hidden layers
            if i < len(self.layers) - 1:
                x = np.maximum(0, x)
                
        return x
