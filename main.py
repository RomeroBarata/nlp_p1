from nltk.corpus import reuters
from feature_extractors.feature_extraction import *
from classifiers.logistic_regression import *

# Constants
NUM_MOST_FREQUENT = 200

# Extract the fileids of the necessary subset of documents.
necessary_documents_fileids = reuters.fileids(['earn', 'acquisitions', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn'])

# Split into training and testing.
training_fileids = [w for w in necessary_documents_fileids if w.startswith('training')]
testing_fileids = [w for w in necessary_documents_fileids if w.startswith('test')]

# Extract features.
most_frequent_words = extract_most_frequent_words(training_fileids, NUM_MOST_FREQUENT)
training_featureset = [(document_features(reuters.words(fileid), most_frequent_words), reuters.categories(fileid)) for fileid in training_fileids]
testing_featureset = [(document_features(reuters.words(fileid), most_frequent_words), reuters.categories(fileid)) for fileid in testing_fileids]



# Train a classifier.

# training_matrix = [x.values() for (x,y) in training_featureset]
# TODO: Add the bias unit to the documents matrix

# The initial theta is a vector of zeros of size n
# theta = np.zeros ((n + 1, 1))
# thetaReg = np.zeros ((n + 1, 1))
# regLambda = 1

# Initial cost
# cost = cost_function(training, classes, theta)

iterations = 1000
alpha = 0.01

# theta = gradient_descent(training, classes, theta, alpha, iterations)
# thetaReg = gradient_descent_reg(training, classes, theta, alpha, num_iterations, regLambda)

# Assess the results.
