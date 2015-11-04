import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Returns the cost J and the gradient vector
# def cost_function(training, theta):

# def cost_function_reg(training, theta, lambda):
