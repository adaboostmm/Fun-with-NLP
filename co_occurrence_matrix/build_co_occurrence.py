import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import nltk
nltk.download('reuters')
from nltk.corpus import reuters
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)



def read_corpus(category="crude"):
    """ Read files from NLTK  Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]
    
    
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    corpus_set = set()
    for list1 in corpus:
        for s in list1:
            corpus_set.add(s)
    corpus_words = sorted(list(corpus_set))
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words
    
    
 def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix) :Co-occurence matrix of word counts. 
   
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2ind = {}
    
    # ------------------

    def get_left_list(ind1, n, list1):
        result = []
        count = 0
        while ind1 > 0:
            if count == n:
                break;

            result.append(list1[ind1-1])
            ind1 = ind1 - 1
            count = count +1
        return result

    def get_right_list(ind1, n, list1):
        result = []
        count = 0
        while ind1 < len(list1) - 1:
            if count == n:
                break;

            result.append(list1[ind1+1])
            ind1 = ind1 + 1
            count = count +1
        return result

    n = window_size
    total_list = {}
    for word in words:
        total_list[word] = []
    for docs in corpus:
        doc1 = docs
        for s in range(len(doc1)):
            left_list = get_left_list(s, n, doc1)
            right_list = get_right_list(s, n, doc1)
            total_list[doc1[s]] = total_list[doc1[s]] + left_list + right_list
        
    mlist = []
    for w in words:
        result = []
        values = total_list.get(w)
        for w2 in words:
            count = 0
            if w2 in values:
                for j in values:
                    if j == w2:
                        count = count + 1
                result.append(count)            
            else:
                result.append(0)
        mlist.append(result)   
    M = np.array(mlist)
    
    word2ind = dict(zip(words, range(len(words))))
    
    # ------------------

    return M, word2ind
    
 def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix) co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix): matrix of k-dimensioal word embeddings.
                    In terms of SVD, this actually returns U * S
    """    
    n_iters = 10   
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
        # ------------------
    
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components = k, n_iter = n_iters, random_state = 42)
    M_reduced = svd.fit_transform(M)
    
        # ------------------

    print("Done.")
    return M_reduced
    
def plot_embeddings(M_reduced, word2ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        
        Params:
            M_reduced (numpy matrix): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
    plt.figure(figsize=(6,6))
    plt.scatter(M_reduced[:,0], M_reduced[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, M_reduced):
        plt.text(x+0.05, y+0.05, word)

    # ------------------
    
#---------------------------------
# PUTTING IT ALL TOGETHER
#---------------------------------
reuters_corpus = read_corpus()
M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']

plot_embeddings(M_normalized, word2ind_co_occurrence, words)
