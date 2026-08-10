import numpy as np
import pygicp
import sys

print(f"Testing pygicp from: {pygicp.__file__}")

# Create dummy point clouds
N = 100
source = np.random.rand(N, 3).astype(np.float64)
target = source + np.array([0.1, 0.0, 0.0])

# Initialize FastGICP
reg = pygicp.FastGICP()
reg.set_input_source(source)
reg.set_input_target(target)

# Create weights
weights = np.ones(N, dtype=np.float32)
weights[0:50] = 0.1 # Low weight for first half

# Set weights
try:
    reg.set_target_weights(weights)
    print("Successfully set target weights!")
except AttributeError:
    print("ERROR: set_target_weights method not found!")
    sys.exit(1)
except Exception as e:
    print(f"Failed to set target weights: {e}")
    sys.exit(1)

# Align
T = reg.align()
print("Alignment successful")
print(T)
