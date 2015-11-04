import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Returns the cost J and the gradient vector
def cost_function(training, classes):
    m = classes.size

    # add the bias unit to the training set
    training = np.concatenate(np.ones((training.shape[0],1)),training, axis=1)

    theta = np.zeros((training.shape[1],1))

    hypothesis = sigmoid(np.dot(training,theta))

    j_1 = np.dot(np.transpose(classes), math.log(hypothesis))
    j_2 = np.dot(np.transpose(np.subtract(1,classes)), math.log(np.subtract(1,hypothesis)))
    J = np.multiply(1/m, np.multiply(-1,np.add(j_1, j_2)))

    gradient = np.multiply(1/m, np.dot(np.traspose(np.subtract(hypothesis,classes)),training))

    return (J, gradient)


def cost_function_reg(training, classes, lambda):
    m = classes.size

    # theta2 excludes the first parameter in orther to be reguarized
    theta2 = np.zeros((training.shape[1],1))

    # add the bias unit to the training set
    training = np.concatenate(np.ones((training.shape[0],1)),training, axis=1)

    theta = np.zeros((training.shape[1],1))

    hypothesis = sigmoid(np.dot(training,theta))

    j_1 = np.dot(np.transpose(classes), math.log(hypothesis))
    j_2 = np.dot(np.transpose(np.subtract(1,classes)), math.log(np.subtract(1,hypothesis)))
    j_3 = np.multiply(lambda/2*m,np.sum(theta2**2))
    J = np.multiply(1/m, np.multiply(-1,np.add(j_1, j_2))) + j_3

    gradient_1 = np.multiply(1/m, np.dot(np.traspose(np.subtract(hypothesis,classes)),training))
    gradient_2 = np.multiply(lambda/m, theta)
    gradient = np.transpose(gradient_1) + gradient_2
    gradient[0] = np.multiply(1/m, np.dot(np.traspose(np.subtract(hypothesis,classes)),training[:0]))

    return (J, gradient)
