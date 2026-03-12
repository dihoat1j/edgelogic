import numpy as np

def calculate_mse(original, quantized):
    """Measures precision loss after quantization."""
    return np.mean((original - quantized)**2)

def simulate_quantization(data):
    """Helper to visualize quantization error."""
    scale = np.max(np.abs(data)) / 127.0
    q = np.round(data / scale).astype(np.int8)
    dq = q.astype(np.float32) * scale
    return dq, calculate_mse(data, dq)
