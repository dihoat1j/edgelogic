import numpy as np
from edgelogic.compiler import ModelCompiler
from edgelogic.runtime import InferenceEngine
import time

def benchmark():
    # Large MLP for an agent: 128 -> 64 -> 32 -> 16
    layers = [
        (128, 64),
        (64, 32),
        (32, 16)
    ]
    
    weights = [np.random.randn(*l).astype(np.float32) for l in layers]
    biases = [np.random.randn(l[1]).astype(np.float32) for l in layers]
    
    compiler = ModelCompiler()
    model = compiler.compile({"weights": weights, "biases": biases})
    model.save("bench.edge")
    
    engine = InferenceEngine("bench.edge")
    input_data = np.random.randn(128).astype(np.float32)
    
    # Warmup
    for _ in range(10): engine.predict(input_data)
    
    start = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        engine.predict(input_data)
    end = time.perf_counter()
    
    avg_ms = (end - start) / iters * 1000
    print(f"Average Inference Time (Int8): {avg_ms:.4f} ms")

if __name__ == "__main__":
    benchmark()
