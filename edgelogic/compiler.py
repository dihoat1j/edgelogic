import numpy as np
import struct
import json
from typing import Dict, Any

class ModelCompiler:
    """Compiles floating point weights into quantized int8 format."""
    
    def compile(self, model_params: Dict[str, Any]) -> 'CompiledModel':
        quantized_layers = []
        for W, b in zip(model_params['weights'], model_params['biases']):
            # Calculate scale: max(abs) / 127
            w_scale = np.max(np.abs(W)) / 127.0 if np.max(np.abs(W)) > 0 else 1.0
            b_scale = np.max(np.abs(b)) / 127.0 if np.max(np.abs(b)) > 0 else 1.0
            
            w_int8 = np.round(W / w_scale).astype(np.int8)
            b_int8 = np.round(b / b_scale).astype(np.int8)
            
            quantized_layers.append({
                'weights': w_int8,
                'biases': b_int8,
                'w_scale': float(w_scale),
                'b_scale': float(b_scale)
            })
        
        return CompiledModel(quantized_layers)

class CompiledModel:
    def __init__(self, layers):
        self.layers = layers

    def save(self, path: str):
        """Saves weights in a compact binary format."""
        with open(path, 'wb') as f:
            # Header: Number of layers
            f.write(struct.pack('<I', len(self.layers)))
            for layer in self.layers:
                rows, cols = layer['weights'].shape
                f.write(struct.pack('<IIff', rows, cols, layer['w_scale'], layer['b_scale']))
                f.write(layer['weights'].tobytes())
                f.write(layer['biases'].tobytes())
