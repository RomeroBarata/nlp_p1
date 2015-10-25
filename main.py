import nltk
from nltk.corpus import reuters

# Extract the fileids of the necessary subset of documents.
necessary_documents_fileids = reuters.fileids(['earn', 'acquisitions', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn'])

# Split into training and testing.

# Train a classifier.

# Assess the results.
