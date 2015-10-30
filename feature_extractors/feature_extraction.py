from nltk.corpus import reuters
from nltk.probability import FreqDist

def extract_most_frequent_words(fileids, num_most_frequent):
    fdist = FreqDist()
    for fileid in fileids:
        for word in reuters.words(fileid):
            fdist[word.lower()] += 1

    most_frequent_words = [k for k, v in fdist.most_common(num_most_frequent)]
    return most_frequent_words

    
def document_features(document, most_frequent_words):
    features = {}
    for word in most_frequent_words:
        features.setdefault(word, 0)

    for word in document:
        if word.lower() in most_frequent_words:
            features[word.lower()] += 1
            
    return features
