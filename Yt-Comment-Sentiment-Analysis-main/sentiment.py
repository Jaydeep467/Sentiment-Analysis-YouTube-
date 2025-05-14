import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


##<<<<<<< HEAD
documents_f = open("pickled_algos/documents.pickle", "rb")
#========
documents_f = open("new_pickled_algos/documents.pickle", "rb")
##>>>>>>> 3b1de88 (Reran the pickled algos)
documents = pickle.load(documents_f)
documents_f.close()




##<<<<<<< HEAD
word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
#========
word_features5k_f = open("new_pickled_algos/word_features5k.pickle", "rb")
##>>>>>>> 3b1de88 (Reran the pickled algos)
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# featuresets=[(find_features(rev),category) for (rev,)]

##<<<<<<< HEAD
# featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
#========
# featuresets_f = open("new_pickled_algos/featuresets.pickle", "rb")
##>>>>>>> 3b1de88 (Reran the pickled algos)
# featuresets = pickle.load(featuresets_f)
# featuresets_f.close()
# featuresets=find_features(documents_f)

# random.shuffle(featuresets)
# print(len(featuresets))

# testing_set = featuresets[10000:]
# training_set = featuresets[:10000]



#<<<<<<< HEAD
open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
#========
open_file = open("new_pickled_algos/originalnaivebayes5k.pickle", "rb")
#>>>>>>> 3b1de88 (Reran the pickled algos)
classifier = pickle.load(open_file)
open_file.close()


#<<<<<<< HEAD
open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
#========
open_file = open("new_pickled_algos/MNB_classifier5k.pickle", "rb")
#>>>>>>> 3b1de88 (Reran the pickled algos)
MNB_classifier = pickle.load(open_file)
open_file.close()



#<<<<<<< HEAD
open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
#========
open_file = open("new_pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
#>>>>>>> 3b1de88 (Reran the pickled algos)
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


#<<<<<<< HEAD
open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
#========
open_file = open("new_pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
#>>>>>>> 3b1de88 (Reran the pickled algos)
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


#<<<<<<< HEAD
open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
#========
open_file = open("new_pickled_algos/LinearSVC_classifier5k.pickle", "rb")
#>>>>>>> 3b1de88 (Reran the pickled algos)
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


#<<<<<<< HEAD
open_file = open("pickled_algos/SGDC_classifier5k.pickle", "rb")
#========
open_file = open("new_pickled_algos/SGDC_classifier5k.pickle", "rb")
#>>>>>>> 3b1de88 (Reran the pickled algos)
SGDC_classifier = pickle.load(open_file)
open_file.close()




voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)




def sentimentAnalysis(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
#<<<<<<< HEAD
#========

# print(sentimentAnalysis("Wow this video is good amazing spectacular joyous and stright up bussin no cap."))
#>>>>>>> 3b1de88 (Reran the pickled algos)
