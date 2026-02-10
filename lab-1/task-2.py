import math
import random

# --- Общие настройки ---
# Таблица истинности XOR: (Input1, Input2) -> Target
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

def run_task_a():
    print(f"\n{'='*60}")
    print("А: 2 Layers (Sigmoid, MSE)")
    print(f"{'='*60}")

    best_weights = None
    min_error = float('inf')

    for _ in range(100000):
        w = [random.uniform(-10, 10) for _ in range(6)]
        
        current_error = 0
        for i1, i2, target in XOR_DATA:
            # H1 = sigmoid(i1*w1 + i2*w3)
            h1 = sigmoid(i1 * w[0] + i2 * w[2])
            # H2 = sigmoid(i1*w2 + i2*w4)
            h2 = sigmoid(i1 * w[1] + i2 * w[3])
            
            # Out = sigmoid(h1*w5 + h2*w6)
            out = sigmoid(h1 * w[4] + h2 * w[5])
            
            current_error += (target - out) ** 2
        
        if current_error < min_error:
            min_error = current_error
            best_weights = w
            if min_error < 0.01:
                break

    print(f"Best error (MSE): {min_error:.5f}")
    print("New weights:", [round(x, 3) for x in best_weights])
    print("-" * 20)
    print("Check:")
    for i1, i2, target in XOR_DATA:
        h1 = sigmoid(i1 * best_weights[0] + i2 * best_weights[2])
        h2 = sigmoid(i1 * best_weights[1] + i2 * best_weights[3])
        out = sigmoid(h1 * best_weights[4] + h2 * best_weights[5])
        print(f"[{i1}, {i2}] -> {out:.3f} (Target: {target})")


def run_task_b():
    print(f"\n{'='*60}")
    print("B: 1 Layer")
    print(f"{'='*60}")

    best_weights = None
    min_error = float('inf')

    # Схема весов: [w1, w2] + Bias (смещение), чтобы дать шанс, хотя это не поможет
    for _ in range(50000):
        w1 = random.uniform(-10, 10)
        w2 = random.uniform(-10, 10)
        bias = random.uniform(-10, 10)
        
        current_error = 0
        for i1, i2, target in XOR_DATA:
            # Out = sigmoid(i1*w1 + i2*w2 + bias)
            val = i1 * w1 + i2 * w2 + bias
            out = sigmoid(val)
            current_error += (target - out) ** 2
        
        if current_error < min_error:
            min_error = current_error
            best_weights = [w1, w2, bias]

    print(f"Best error (MSE): {min_error:.5f}")
    print("Weights:", [round(x, 3) for x in best_weights])
    print("-" * 20)
    print("Check:")
    for i1, i2, target in XOR_DATA:
        val = i1 * best_weights[0] + i2 * best_weights[1] + best_weights[2]
        out = sigmoid(val)
        print(f"[{i1}, {i2}] -> {out:.3f} (Target: {target})")


def run_task_c():
    print(f"\n{'='*60}")
    print("V: 2 Layers, 3 neurons, Tanh (activation), Arctan (error)")
    print(f"{'='*60}")

    best_weights = None
    min_arctan_error = float('inf')

    for _ in range(100000):
        w = [random.uniform(-5, 5) for _ in range(9)]
        
        # w[0]..w[5]
        # w[6]..w[8]
        
        total_error = 0
        for i1, i2, target in XOR_DATA:
            # H1
            h1_in = i1 * w[0] + i2 * w[3] 
            h1 = math.tanh(h1_in)
            
            # H2
            h2_in = i1 * w[1] + i2 * w[4] 
            h2 = math.tanh(h2_in)
            
            # H3
            h3_in = i1 * w[2] + i2 * w[5] 
            h3 = math.tanh(h3_in)

            # Out
            out_in = h1 * w[6] + h2 * w[7] + h3 * w[8]
            out = math.tanh(out_in)
            
            # Error
            diff = target - out
            err = math.atan(diff)
            total_error += abs(err) 
        
        if total_error < min_arctan_error:
            min_arctan_error = total_error
            best_weights = w
            if min_arctan_error < 0.01:
                break

    print(f"Best error (Sum |Arctan(diff)|): {min_arctan_error:.5f}")
    print("Weights:", [round(x, 2) for x in best_weights])
    print("-" * 20)
    print("Check:")
    for i1, i2, target in XOR_DATA:
        h1 = math.tanh(i1 * best_weights[0] + i2 * best_weights[3])
        h2 = math.tanh(i1 * best_weights[1] + i2 * best_weights[4])
        h3 = math.tanh(i1 * best_weights[2] + i2 * best_weights[5])
        out = math.tanh(h1 * best_weights[6] + h2 * best_weights[7] + h3 * best_weights[8])
        
        err_val = math.atan(target - out)
        print(f"[{i1}, {i2}] -> {out:.3f} (Target: {target}) | Err(atan): {err_val:.3f}")

if __name__ == "__main__":
    run_task_a()
    run_task_b()
    run_task_c()