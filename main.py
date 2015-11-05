from nltk.corpus import reuters
from feature_extractors.feature_extraction import *
from classifiers.logistic_regression import *
from assessment.assessment_metrics import *
import pprint

## Constants ##
NUM_MOST_FREQUENT = 700
CATEGORIES = ['earn', 'acquisitions', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']
# Logistic Regression parameters
ITERATIONS = 150
ALPHA = 0.1
THRESHOLD = 0.5
#LAMBDA = 1

## Extract the fileids of the necessary subset of documents ##
documents_fileids = reuters.fileids(CATEGORIES)

## Split into training and testing ##
training_fileids = [w for w in documents_fileids if w.startswith('training')]
testing_fileids = [w for w in documents_fileids if w.startswith('test')]

## Extract features ##
most_frequent_words = extract_most_frequent_words(training_fileids, NUM_MOST_FREQUENT)
training_featureset = [(document_features(reuters.words(fileid), most_frequent_words), reuters.categories(fileid)) for fileid in training_fileids]
testing_featureset = [(document_features(reuters.words(fileid), most_frequent_words), reuters.categories(fileid)) for fileid in testing_fileids]

## Train a classifier ##

# Create a matrix with the values of the features to send to the classifier
training_matrix = np.array([list(x.values()) for x, y in training_featureset])
# Create a matrix with only the classes labels of each document
classes_vector = [y for x, y in training_featureset]
n = training_matrix.shape[1] # number of features
m = len(training_matrix) # number of training documents
# add the bias unit to the training matrix
training_matrix = np.concatenate((np.ones((m,1)), training_matrix), axis = 1)

# creating ten classifiers
classifiers = dict( (c, np.zeros((n+1,1)) ) for c in CATEGORIES )

# training the ten classifiers
for c in CATEGORIES:
    print ("Training Class: " + c)
    # map the classes vector to a vetor of 1s and 0s (1 if the class is equal to c)
    mapClasses = [[1] if c in cl else [0] for cl in classes_vector]
    mapClasses = np.array(mapClasses)

    cost = cost_function(training_matrix, mapClasses, classifiers[c])
    print ("Initial Cost: " + str(cost))

    # update the parameter in order to minimize the cost
    classifiers[c] = gradient_descent(training_matrix, mapClasses, classifiers[c], ALPHA, ITERATIONS)
    
    cost = cost_function(training_matrix, mapClasses, classifiers[c])
    print ("Final Cost: " + str(cost))

## Classify unseen examples and assess the results ##
testing_matrix = np.array([list(x.values()) for x, y in testing_featureset])
testing_matrix = np.concatenate((np.ones((len(testing_matrix), 1)), testing_matrix), axis = 1)
testing_true_classes = [y for x, y in testing_featureset]

performances = []
for category in CATEGORIES:
    predicted_class = classify(testing_matrix, classifiers[category], THRESHOLD, category)
    performances.append(compute_metrics(testing_true_classes, predicted_class, category))

macro_metrics = macro_average(performances)
micro_metrics = micro_average(performances)

## Print the results ##
pprint.pprint(performances)
pprint.pprint(macro_metrics)
pprint.pprint(micro_metrics)
