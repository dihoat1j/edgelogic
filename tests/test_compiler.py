from edgelogic.compiler import ModelCompiler
import numpy as np
import pytest

def test_compiler_scaling():
    compiler = ModelCompiler()
    # Test with known scale
    W = np.array([[127.0, -127.0], [0.0, 63.5]], dtype=np.float32)
    b = np.array([0.0, 0.0], dtype=np.float32)
    
    params = {"weights": [W], "biases": [b]}
    model = compiler.compile(params)
    
    layer = model.layers[0]
    assert layer['w_scale'] == 1.0
    assert np.array_equal(layer['weights'], np.array([[127, -127], [0, 64]], dtype=np.int8))

if __name__ == "__main__":
    pytest.main([__file__])
