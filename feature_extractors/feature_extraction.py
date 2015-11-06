from nltk.corpus import reuters
from nltk.probability import FreqDist

# Constants
MOST_COMMON_WORDS_ENGLISH = ['the', 'be', 'am', 'are', 'is', 'was', 'were', 'been', 'being', 'to', 'of', 'and', 'a', 'in', 'that',
                             'have', 'has', 'had', 'having', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 
                             'do', 'does', 'did', 'doing', 'at', 'this', 'but', 'his', 'by', 'from', 
                             'they', 'we', 'say', 'says', 'said', 'saying', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 
                             'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'gets', 'got', 'getting', 'which', 'go', 
                             'goes', 'went', 'gone', 'me', 'when', 'make', 'can', 'could', 'like', 'time', 'no', 'just',
                             'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
                             'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use',
                             'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
                             'give', 'day', 'most', 'us']

def extract_most_frequent_words(fileids, num_most_frequent):
    ''' Function to extract the most frequent words from a corpus.
        Args:
              fileids: file ids for the documents in the reuters corpus.
              num_most_frequent: Number of most frequent words the user
                                 wish to compute.
    '''
    fdist = FreqDist()
    for fileid in fileids:
        for word in reuters.words(fileid):
            fdist[word.lower()] += 1

    most_frequent_words = [k for k, v in fdist.most_common(num_most_frequent)
                           if k.isalpha() and len(k) > 2 
                           and k not in MOST_COMMON_WORDS_ENGLISH]
    return most_frequent_words

    
def document_features(document, most_frequent_words):
    ''' Function to extract the features from a document.
        Args:
              document: A list of words describing the document.
              most_frequent_words: A list of the most frequent words (the features).
    '''
    features = {}
    for word in most_frequent_words:
        features.setdefault(word, 0)

    for word in document:
        if word.lower() in most_frequent_words:
            features[word.lower()] += 1
            
    return features
