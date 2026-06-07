# Basic usage example for Aurelis
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aurelis

print("=== Testing Aurelis Python Bindings ===")

# Test tensor creation
print("\n1. Testing Tensor Creation")
t = aurelis.Tensor.zeros([3, 4])
print(f"Shape: {t.shape}")
print(f"Numel: {t.numel()}")

# Test setting values
for i in range(t.numel()):
    t[i] = float(i + 1)

# Test getting values
print("\n2. Testing Tensor Values")
for i in range(t.numel()):
    print(f"t[{i}] = {t[i]:.1f}")

# Test saving
print("\n3. Testing Tensor Saving")
t.save("test_tensor.aurelis")
print("Saved tensor successfully")

# Test loading
print("\n4. Testing Tensor Loading")
t2 = aurelis.Tensor.load("test_tensor.aurelis")
print("Loaded tensor successfully")

# Test config
print("\n5. Testing Config")
config = aurelis.AurelisConfig()
config.lens.D = 64
config.lens.vocab_size = 16
config.save("test_config.json")
print("Config saved")

config2 = aurelis.AurelisConfig.load("test_config.json")
print(f"Loaded config: D={config2.lens.D}, vocab_size={config2.lens.vocab_size}")

print("\n=== All tests passed! ===")
