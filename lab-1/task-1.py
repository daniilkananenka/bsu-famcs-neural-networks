import math
import random

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, weights):
        self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = weights
        
        self.last_inputs = None
        self.h1_out = None
        self.h2_out = None
        self.o1_out = None

    def predict(self, i1, i2):
        self.last_inputs = (i1, i2)
        
        # H1
        h1_input = i1 * self.w1 + i2 * self.w3
        self.h1_out = sigmoid(h1_input)
        
        # H2
        h2_input = i1 * self.w2 + i2 * self.w4
        self.h2_out = sigmoid(h2_input)
        
        # O1
        o1_input = self.h1_out * self.w5 + self.h2_out * self.w6
        self.o1_out = sigmoid(o1_input)
        
        return self.o1_out

# From example
initial_weights = [0.45, 0.78, -0.12, 0.13, 1.5, -2.3]
input1 = 1
input2 = 0
target_example = 1

nn = NeuralNetwork(initial_weights)
result = nn.predict(input1, input2)
mse_error = (target_example - result) ** 2

print("="*30)
print("Task 1")
print("="*30)
print(f"Inputs: I1={input1}, I2={input2}")
print(f"Weights: {initial_weights}")
print(f"Result by NN: {result:.2f}")  # 0.33
print(f"Error (MSE): {mse_error:.2f}") # 0.45
print(f"Is equal with example: {'Yes' if round(result, 2) == 0.33 else 'No'}")
print("\n")
