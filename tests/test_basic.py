import numpy as np
from edgelogic.compiler import ModelCompiler
from edgelogic.runtime import InferenceEngine
import os

def test_inference_flow():
    # Setup dummy model
    W1 = np.random.randn(5, 10).astype(np.float32)
    b1 = np.random.randn(10).astype(np.float32)
    
    model_data = {"weights": [W1], "biases": [b1]}
    compiler = ModelCompiler()
    compiled = compiler.compile(model_data)
    
    path = "test_model.edge"
    compiled.save(path)
    
    # Check if file exists
    assert os.path.exists(path)
    
    # Run inference
    engine = InferenceEngine(path)
    input_data = np.random.randn(5).astype(np.float32)
    output = engine.predict(input_data)
    
    assert output.shape == (10,)
    os.remove(path)

if __name__ == "__main__":
    test_inference_flow()
    print("Test Passed!")
