from nltk.corpus import reuters
from feature_extractors.feature_extraction import *
from classifiers.logistic_regression import *

# Constants
NUM_MOST_FREQUENT = 200
TOP_CLASSES = ['earn', 'acquisitions', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']

# Extract the fileids of the necessary subset of documents.
necessary_documents_fileids = reuters.fileids(TOP_CLASSES)

# Split into training and testing.
training_fileids = [w for w in necessary_documents_fileids if w.startswith('training')]
testing_fileids = [w for w in necessary_documents_fileids if w.startswith('test')]

# Extract features.
most_frequent_words = extract_most_frequent_words(training_fileids, NUM_MOST_FREQUENT)
training_featureset = [(document_features(reuters.words(fileid), most_frequent_words), reuters.categories(fileid)) for fileid in training_fileids]
testing_featureset = [(document_features(reuters.words(fileid), most_frequent_words), reuters.categories(fileid)) for fileid in testing_fileids]

print "going to the classifier part"

# Train a classifier.

training_matrix = [x.values() for (x,y) in training_featureset]
classes_vector = [y for (x,y) in training_featureset]
n = NUM_MOST_FREQUENT
m = len(training_matrix)
# add the bias unit to the training matrix
training_matrix = np.concatenate((np.ones((m,1)) , training_matrix) , axis=1)

# creating ten classifiers
classifiers = dict( (c, np.zeros((n+1,1)) ) for c in TOP_CLASSES )

# regLambda = 1

iterations = 100
alpha = 0.1

c = TOP_CLASSES[0]
mapClasses = [[1] if c in cl else [-1] for cl in classes_vector]
print "initial cost"
cost = cost_function(training_matrix, mapClasses, classifiers[c])
print cost
bla = gradient_descent(training_matrix, mapClasses, classifiers[c], alpha, iterations)
print "trained cost"
cost2 = cost_function(training_matrix, mapClasses, bla)
print cost2

# Assess the results.
