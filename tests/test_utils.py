import unittest
import numpy as np
from edgelogic.utils import calculate_mse, simulate_quantization

class TestUtils(unittest.TestCase):
    def test_mse(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 4])
        self.assertAlmostEqual(calculate_mse(a, b), 1/3)

    def test_sim_quant(self):
        data = np.random.randn(100)
        dq, mse = simulate_quantization(data)
        self.assertTrue(mse < 0.1)
        self.assertEqual(dq.shape, (100,))

if __name__ == "__main__":
    unittest.main()
