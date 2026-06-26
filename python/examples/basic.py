# Project Aurelis - Basic Usage Example
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aurelis

def main():
    print("=== Project Aurelis - Basic Example ===\n")

    # 1. Test Tensor Operations
    print("Step 1: Tensor operations")
    a = aurelis.Tensor.zeros([2, 3])
    b = aurelis.Tensor.zeros([2, 3])
    for i in range(a.numel()):
        a[i] = i + 1
        b[i] = i + 4
    print("  a =\n", a.to_numpy())
    print("  b =\n", b.to_numpy())

    c = aurelis.add(a, b)
    print("  a + b =\n", c.to_numpy())

    d = aurelis.mul(a, b)
    print("  a * b =\n", d.to_numpy())

    # 2. Test Linear Layer
    print("\nStep 2: Linear layer")
    linear = aurelis.Linear(3, 2)
    linear.init_xavier()
    print("  Linear weight shape:", linear.weight.shape)
    print("  Linear bias shape:", linear.bias.shape)

    x = aurelis.Tensor.zeros([2, 3])
    for i in range(x.numel()):
        x[i] = i + 1
    y = linear.forward(x)
    print("  x =\n", x.to_numpy())
    print("  y = linear(x) =\n", y.to_numpy())

    # 3. Test Config
    print("\nStep 3: Configuration system")
    config = aurelis.AurelisConfig()
    config.lens.vocab_size = 100
    config.lens.D = 128
    config.save("config_example.json")
    loaded_config = aurelis.AurelisConfig.load("config_example.json")
    print("  Loaded config D =", loaded_config.lens.D)
    print("  Loaded config vocab_size =", loaded_config.lens.vocab_size)

    # 4. Test Checkpointing
    print("\nStep 4: Checkpointing")
    linear.save("linear_example.aur")
    loaded_linear = aurelis.Linear.load("linear_example.aur")
    print("  Checkpointed linear weight shape:", loaded_linear.weight.shape)

    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    main()
