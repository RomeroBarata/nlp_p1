import math
import numpy

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Returns the cost J and the gradient vector
def cost_function(training, theta, classes):
    m = classes.size

    hypothesis = sigmoid(numpy.dot(training,theta))

    j_1 = numpy.dot(numpy.transpose(classes), math.log(hypothesis))
    j_2 = numpy.dot(numpy.transpose(numpy.subtract(1,classes)), math.log(numpy.subtract(1,hypothesis)))
    J = numpy.multiply(1/m, numpy.multiply(-1,numpy.add(j_1, j_2)))

    gradient = numpy.multiply(1/m, numpy.dot(numpy.traspose(numpy.subtract(hypothesis,classes)),training))

    return (J, gradient)


# def cost_function_reg(training, theta, lambda):
