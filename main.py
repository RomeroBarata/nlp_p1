from nltk.corpus import reuters
from feature_extractors.feature_extraction import *

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
# add the bias unit to the training set

# Assess the results.
