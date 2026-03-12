import numpy as np
from edgelogic.compiler import ModelCompiler
from edgelogic.runtime import InferenceEngine

def run_agent_simulation():
    print("--- EdgeLogic Agent Simulation ---")
    
    # Model: Input(3) -> Hidden(8) -> Output(2)
    # Inputs: [DistanceToEnemy, Health, Ammo]
    # Outputs: [MoveSpeed, ShootProbability]
    
    W1 = np.random.uniform(-1, 1, (3, 8))
    b1 = np.random.uniform(-0.1, 0.1, 8)
    W2 = np.random.uniform(-1, 1, (8, 2))
    b2 = np.random.uniform(-0.1, 0.1, 2)
    
    model_data = {"weights": [W1, W2], "biases": [b1, b2]}
    compiler = ModelCompiler()
    compiled = compiler.compile(model_data)
    compiled.save("brain.edge")
    
    engine = InferenceEngine("brain.edge")
    
    # Simulated input observations
    observations = [
        np.array([10.0, 100.0, 50.0]), # Far, Healthy, High Ammo
        np.array([2.0, 10.0, 5.0]),   # Close, Dying, Low Ammo
    ]
    
    for obs in observations:
        action = engine.predict(obs)
        print(f"Obs: {obs} -> Action: {action}")

if __name__ == "__main__":
    run_agent_simulation()
