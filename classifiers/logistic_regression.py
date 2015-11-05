import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

vsigmoid = np.vectorize(sigmoid)

# Returns the cost J and the gradient vector
def cost_function(training, classes, theta):
    m = len(classes)

    hypothesis = vsigmoid(np.dot(training,theta))

    j_1 = np.dot(np.transpose(classes), np.log(hypothesis))
    j_2 = np.dot(np.transpose(np.subtract(1, classes)), np.log(np.subtract(1,hypothesis)))
    J = -1/m * (j_1[0][0] + j_2[0][0])
    
    return J


def cost_function_reg(training, classes, regLambda, theta):
    m = len(classes)

    # theta2 excludes the first parameter in orther to be reguarized
    theta2 = theta[range(1,theta.size)]

    hypothesis = vsigmoid(np.dot(training,theta))

    j_1 = np.dot(np.transpose(classes), np.log(hypothesis))
    j_2 = np.dot(np.transpose(np.subtract(1,classes)), np.log(np.subtract(1,hypothesis)))
    j_3 = (regLambda/(2*m)) * np.sum(theta2**2)
    J = (1/m) * (-1) * np.add(j_1, j_2) + j_3    

    return J


def gradient_descent(training, classes, theta, alpha, num_iterations):
    m = len(classes)

    for i in range(0,num_iterations):
        print (i)

        hypothesis = vsigmoid(np.dot(training,theta))
        
        gradient = np.multiply(1/m, np.dot(np.transpose(np.subtract(hypothesis, classes)),training))

        theta = np.subtract(theta, np.transpose(np.multiply(alpha, gradient)))

    return theta


def gradient_descent_reg(training, classes, theta, alpha, num_iterations, regLambda):
    m = len(classes)

    for i in range(0,num_iterations):

        hypothesis = vsigmoid(np.dot(training,theta))

        gradient_1 = np.multiply(1/m, np.dot(np.transpose(np.subtract(hypothesis, classes)),training))
        gradient_2 = np.multiply((regLambda/m), theta)
        gradient = np.transpose(gradient_1) + gradient_2
        gradient[0] = (1/m) * np.dot(np.traspose(hypothesis - classes),training[:,0])

        theta = np.subtract(theta, np.transpose(np.multiply(alpha, gradient)))

    return theta

