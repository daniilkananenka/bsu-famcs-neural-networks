import math
import random

XOR_DATA = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

def sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def run_task_3a():
    print("=" * 60)
    print("A: 2 Layers + Bias (Sigmoid, MSE)")
    print("=" * 60)

    best_weights = None
    min_error = float('inf')

    for _ in range(100000):
        w = [random.uniform(-15, 15) for _ in range(9)]
        
        current_error = 0
        for i1, i2, target in XOR_DATA:
            h1 = sigmoid(i1 * w[0] + i2 * w[1] + w[2])
            h2 = sigmoid(i1 * w[3] + i2 * w[4] + w[5])
            out = sigmoid(h1 * w[6] + h2 * w[7] + w[8])
            
            current_error += (target - out) ** 2
        
        if current_error < min_error:
            min_error = current_error
            best_weights = w
            if min_error < 0.001:
                break

    print(f"Best error (MSE): {min_error:.5f}")
    print(f"Weights: {[round(x, 3) for x in best_weights]}")
    print("-" * 20)
    print("Check:")
    
    w = best_weights
    for i1, i2, target in XOR_DATA:
        h1 = sigmoid(i1 * w[0] + i2 * w[1] + w[2])
        h2 = sigmoid(i1 * w[3] + i2 * w[4] + w[5])
        out = sigmoid(h1 * w[6] + h2 * w[7] + w[8])
        print(f"[{i1}, {i2}] -> {out:.3f} (Target: {target})")

def run_task_3b():
    print("\n" + "=" * 60)
    print("B: 1 Layer + Bias")
    print("=" * 60)

    best_weights = None
    min_error = float('inf')

    for _ in range(50000):
        w = [random.uniform(-10, 10) for _ in range(3)]

        current_error = 0
        for i1, i2, target in XOR_DATA:
            out = sigmoid(i1 * w[0] + i2 * w[1] + w[2])
            current_error += (target - out) ** 2
        
        if current_error < min_error:
            min_error = current_error
            best_weights = w

    print(f"Best error (MSE): {min_error:.5f}")
    print(f"Weights: {[round(x, 3) for x in best_weights]}")
    print("-" * 20)
    print("Check:")
    
    w = best_weights
    for i1, i2, target in XOR_DATA:
        out = sigmoid(i1 * w[0] + i2 * w[1] + w[2])
        print(f"[{i1}, {i2}] -> {out:.3f} (Target: {target})")

if __name__ == "__main__":
    run_task_3a()
    run_task_3b()