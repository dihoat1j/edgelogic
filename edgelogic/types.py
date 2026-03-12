from typing import List, Protocol
import numpy as np

class EdgeModel(Protocol):
    def predict(self, input_vec: np.ndarray) -> np.ndarray:
        ...

class LayerMetadata:
    def __init__(self, rows: int, cols: int, w_scale: float, b_scale: float):
        self.rows = rows
        self.cols = cols
        self.w_scale = w_scale
        self.b_scale = b_scale
