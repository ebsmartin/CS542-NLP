# CS542 Fall 2021 Programming Assignment 2
# Logistic Regression Classifier

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random
from collections import Counter

'''
Computes the logistic function.
'''
def sigma(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, n_features=2):

        # holds the class labels for each document
        self.class_dict = {'neg': 0, 'pos': 1}

        # Added this to hold the common words and bigrams for use in featurize
        self.negatives, self.positives = self.generate_common_words(1000)
        self.negative_bigrams, self.positive_bigrams = self.generate_common_bigrams(1000)

        # Added this to hold words that may indicate the conclusion of a review (e.g. "in conclusion", "to summarize", etc.)
        self.conclusive_words = []

        # self.feature_dict holds the index of each feature in my feature vector
        self.feature_dict = {'num_pos_words': 0, 'num_neg_words': 1, 'num_pos_words_conclusion': 2, 'num_neg_words_conclusion': 3, 'num_pos_bigrams': 4, 'num_neg_bigrams': 5, 'num_pos_bigrams_conclusion': 6, 'num_neg_bigrams_conclusion': 7}


        self.n_features = n_features
        self.theta = np.zeros(n_features + 1) # weights (and bias)

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()

        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # your code here
                    # BEGIN STUDENT CODE
                    filenames.append(name) 
                    classes[name] = self.class_dict[os.path.basename(root)] # store class of document with filename as key and class as index
                    document_words = f.read().split() # read in document into list of words
                    documents[name] = self.featurize(document_words)  # send a doc as a list of words to be featurized
                    # END STUDENT CODE
        return filenames, classes, documents

    # BEGIN STUDENT CODE
    ''' 
    This function finds the most common words in the training set and returns them as a list of strings
    It returns two lists, one for negative reviews and one for positive reviews
    '''
    def generate_common_words(self, top_n_words, path="movie_reviews/train"):
        # Initialize counters for negative and positive reviews to hold count of each word occurance
        # I cuold have made my own counter but this is easier
        negative_counter, positive_counter = Counter(), Counter()

        # Open and iterate through negative reviews adding words to the counter
        for root, dirs, files in os.walk(os.path.join(path, "neg")):
            for name in files:
                with open(os.path.join(root, name), 'r') as f:
                    # Filter out words that are non-letter characters (this helped a lot)
                    words = [word for word in f.read().split() if word.isalpha()] # this grabs the list of all words that are alphabetic
                    negative_counter.update(words)  # this sums the count of each word in the list

        # Does the same as above but for positive reviews
        for root, dirs, files in os.walk(os.path.join(path, "pos")):
            for name in files:
                with open(os.path.join(root, name), 'r') as f:
                    # Filter out words that are non-letter characters
                    words = [word for word in f.read().split() if word.isalpha()]
                    positive_counter.update(words)

        # Identify words that appear in both positive and negative reviews
        # I did this to remove words that are common to both positive and negative reviews
        # This way the words in the lists should be indicative of pos or neg
        common_words = set(negative_counter.keys()).intersection(set(positive_counter.keys()))

        # Remove common words from the counters
        for word in common_words:
            del negative_counter[word]  
            del positive_counter[word]  

        # Get top N words from each counter
        negatives = [item[0] for item in negative_counter.most_common(top_n_words)]
        positives = [item[0] for item in positive_counter.most_common(top_n_words)]

        return negatives, positives
    # END STUDENT CODE

    # BEGIN STUDENT CODE
    ''' 
    This function finds the most common bigrams in the training set. I am trying to increase accuracy of the classifier.
    This is pretty much the same as above except 
    '''

    def generate_common_bigrams(self, top_n_bigrams, path="movie_reviews/train"):
        negative_counter, positive_counter = Counter(), Counter()

        # Helper function to extract bigrams from a text
        def extract_bigrams(text):
            # Split the text into words
            words = text.split()

            # Only include bigrams where both words are alphabetic
            bigrams = []
            for i in range(len(words) - 1):
                word1 = words[i]
                word2 = words[i + 1]
                if word1.isalpha() and word2.isalpha():
                    bigrams.append((word1, word2))
            return bigrams

        # Iterate through negative reviews
        for root, dirs, files in os.walk(os.path.join(path, "neg")):
            for name in files:
                with open(os.path.join(root, name), 'r') as f:
                    bigrams = extract_bigrams(f.read())
                    negative_counter.update(bigrams)

        # Iterate through positive reviews
        for root, dirs, files in os.walk(os.path.join(path, "pos")):
            for name in files:
                with open(os.path.join(root, name), 'r') as f:
                    bigrams = extract_bigrams(f.read())
                    positive_counter.update(bigrams)

        # Identify bigrams that appear in both positive and negative reviews
        common_bigrams = set(negative_counter.keys()).intersection(set(positive_counter.keys()))

        # Remove common bigrams from the counters
        for bigram in common_bigrams:
            del negative_counter[bigram]
            del positive_counter[bigram]

        # Get top N bigrams from each counter
        negative_bigrams = [item[0] for item in negative_counter.most_common(top_n_bigrams)]
        positive_bigrams = [item[0] for item in positive_counter.most_common(top_n_bigrams)]

        return negative_bigrams, positive_bigrams
    # END STUDENT CODE


    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        # BEGIN STUDENT CODE
        # count all neg and pos words in document
        ''' 
        So the thought process here was originally to check if the last two sentences of each doc was reached
        and if so, the check if any conclusive words were hit. If they were, then I would weight the sentiment hits
        higher since they may be more indicative of the overall sentiment of the review.

        I didn't want to add any return functions to the featurize function so I instead of checking for the last two
        sentences, I just check for the conclusive words and then set a flag to true which prob isn't as good... but oh well.
        '''
        conclusive_word_reached = False
        for word in document:
            if word in self.conclusive_words and conclusive_word_reached == False:
                conclusive_word_reached = True
                # print('conclusive word reached')
                continue 
            if word in self.negatives:
                if conclusive_word_reached:
                    vector[self.feature_dict['num_neg_words_conclusion']] += 2  # weight words in conclusion higher
                vector[self.feature_dict['num_neg_words']] += 1
            elif word in self.positives:
                if conclusive_word_reached:
                    vector[self.feature_dict['num_pos_words_conclusion']] += 2  # weight words in conclusion higher
                vector[self.feature_dict['num_pos_words']] += 1

        # count all neg and pos bigrams in document
        for i in range(len(document) - 1):
            bigram = (document[i], document[i + 1])
            if bigram in self.negative_bigrams:
                if conclusive_word_reached:
                    vector[self.feature_dict['num_neg_bigrams_conclusion']] += 10 # weight bigrams in conclusion much higher
                vector[self.feature_dict['num_neg_bigrams']] += 5
            elif bigram in self.positive_bigrams:
                if conclusive_word_reached:
                    vector[self.feature_dict['num_pos_bigrams_conclusion']] += 10 # weight bigrams in conclusion much higher
                vector[self.feature_dict['num_pos_bigrams']] += 5

        # NOTE: I would like to normalize this vector for good practice but I will keep it as is for this assignment.
        # END STUDENT CODE
        
        vector[-1] = 1
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                # BEGIN STUDENT CODE
                # create and fill in matrix x and vector y
                # Initialize matrix X and vector Y for the minibatch
                X = np.zeros((len(minibatch), self.n_features + 1))
                Y = np.zeros(len(minibatch))
                
                # Fill in X and Y with each files vector and class info
                for i, name in enumerate(minibatch):
                    X[i] = documents[name]
                    Y[i] = classes[name]

                # compute y_hat

                y_hat = sigma(np.dot(X, self.theta)) # order of X and theta matters here
                # y_hat = sigma(X @ self.theta) # should be the same as above
                # NOTE: calling the built in sigma function sometimes gives a division by zero warning
                # print('y_hat calculated' + str(y_hat))

                # update cross entropy loss
                loss += -np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
                # hmmm... should I use np.sum for this?
                # print('loss calculated: ' + str(loss))

                # compute gradient
                gradient = np.dot(X.T, y_hat - Y) / len(minibatch)
                # gradient = (X.T @ (y_hat - Y)) / len(minibatch) # should be the same as above
                # print('gradient calculated' + str(gradient))

                # update weights (and bias)
                self.theta = self.theta - (eta * gradient)
                # print('weights updated')

                # END STUDENT CODE
            print(loss)
            print('len of filenames: ' + str(len(filenames)))
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            # BEGIN STUDENT CODE
            # get most likely class (recall that P(y=1|x) = y_hat)
            y_hat = sigma(np.dot(self.theta, documents[name]))
            
            # Determine the predicted class
            if y_hat > 0.5:
                predicted_class = 1    
            else:
                predicted_class = 0
            
            # Return a dictionary of filenames mapped to their correct and predicted
            results[name]['correct'] = classes[name]
            results[name]['predicted'] = predicted_class
            # END STUDENT CODE
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # BEGIN STUDENT CODE
        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # F1 = 2 * (precision * recall) / (precision + recall)
        

        TP = 0  # true positive
        FP = 0  # false positive
        TN = 0  # true negative
        FN = 0  # false negative

        for name in results:
            # true positive
            if results[name]['correct'] == 1 and results[name]['predicted'] == 1:
                TP += 1
            # false positive
            elif results[name]['correct'] == 0 and results[name]['predicted'] == 1:
                FP += 1
            # true negative
            elif results[name]['correct'] == 0 and results[name]['predicted'] == 0:
                TN += 1
            # false negative
            elif results[name]['correct'] == 1 and results[name]['predicted'] == 0:
                FN += 1

        # calculate precision, recall, F1, and accuracy
        precision = TP / (TP + FP)  # num predicted positive reviews were actually positive
        recall = TP / (TP + FN)  # num actual positive reviews were predicted positive
        F1 = 2 * (precision * recall) / (precision + recall)  # F1 thingy
        accuracy = (TP + TN) / (TP + TN + FP + FN)  #  how many reviews were predicted correctly

        # print results
        print('Precision: ' + str(round(precision,2)))
        print('Recall: ' + str(round(recall,2)))
        print('F1: ' + str(round(F1,2)))
        print('Accuracy: ' + str(round(accuracy,2)))

        return precision, recall, F1, accuracy
    # END STUDENT CODE


    '''---------------------------------------------------------------------------------'''


if __name__ == '__main__':
    lr = LogisticRegression(n_features=8)

    # print('common negative words: ' + str(lr.negatives))
    # print('common positive words: ' + str(lr.positives))

    # print('common negative bigrams: ' + str(lr.negative_bigrams))
    # print('common positive bigrams: ' + str(lr.positive_bigrams))

    # Painstakingly manually choose words from the top 500 that aren't proper nouns or super movie specific
    # I also added similiar wordss for example, 'lame' was in the list so I added 'lamest'
    # My positives list is longer...not sure if this matters yet
    lr.negatives = [
                    "atrocious", "atrociously",
                    "incoherent", "incoherently",
                    "dud",
                    "horrid", "horridly",
                    "shoddy",
                    "overwrought",
                    "feeble", "feebly",
                    "horrendous", "horrendously",
                    "ineffectual",
                    "pathetic", "pathetically",
                    "reject", "rejected",
                    "lame", "lamest",
                    "leaden",
                    "incompetence", "incompetent",
                    "abysmal", "abysmally",
                    "unamusing",
                    "travesty",
                    "putrid",
                    "absurd", "absurdly", "absurdity",
                    "muck",
                    "moron", "moronic",
                    "plod", "plodding",
                    "stupid", "stupidest", "stupidity",
                    "nonsensical", "nonsense",
                    "unimpressive",
                    "irritate", "irritating", "irritatingly",
                    "unentertaining",
                    "clunker",
                    "ill-advised",
                    "insipid", "insipidity",
                    "woeful", "woefully",
                    "unacceptable", "unacceptably",
                    "terrible", "terribly",
                    "vomit",
                    "rot", "rotting",
                    "inept", "ineptitude",
                    "uninterested", "disinterested",
                    "embarrass", "embarrassing", "embarrassingly",
                    "unwatchable",
                    "unbearable", "unbearably",
                    "unlikable",
                    "unsatisfying", "unsatisfied",
                    "unbelievable", "unbelievably",
                    "tedious", "tediously",
                    "sloppy", "sloppiness",
                    "sketch", "sketchy",
                    "repetitive", "repetition",
                    "regret", "regrettable", "regrettably",
                    "offensive", "offensively",
                    "ineffective", "ineffectively",
                    "dreadful", "dreadfully",
                    "disastrous", "disastrously",
                    "disappoint", "disappointing", "disappointingly",
                    "dismal", "dismally",
                    "clumsy", "clumsily",
                    "chaos", "chaotic",
                    "bore", "boring", "boringly",
                    "awful", "awfully",
                    "appall", "appalling", "appallingly",
                    "annoy", "annoying", "annoyingly",
                    "aggravate", "aggravating"
                ]
    lr.positives = [
                    "ideal", "ideals",
                    "love", "loving", "lovingly",
                    "masterful", "masterfully",
                    "exhilarate", "exhilarating",
                    "steady",
                    "must-see",
                    "symbol", "symbols",
                    "introspect", "introspective",
                    "divine",
                    "powerful", "powerfully",
                    "vivid", "vividly",
                    "audacious",
                    "harmonize", "harmony",
                    "foundation",
                    "uncompromising",
                    "deft", "deftly",
                    "affection", "affectionate",
                    "sensitive", "sensitivity",
                    "remark", "remarkable",
                    "admire", "admiration",
                    "comfort", "comforts",
                    "passion", "passionate",
                    "cherish", "cherished",
                    "work", "workings",
                    "meticulous", "meticulously",
                    "stand-out",
                    "honor", "honour",
                    "droll",
                    "brisk",
                    "notion", "notions",
                    "authentic", "authenticity",
                    "unwavering",
                    "respect", "respectful",
                    "elegant", "elegantly",
                    "purpose", "purposeful",
                    "resolve", "resolves",
                    "immerse", "immersive",
                    "embrace", "embraces",
                    "resilience", "resilient",
                    "enthusiasm", "enthusiastic",
                    "profound",
                    "captivate", "captivating",
                    "inspire", "inspiring",
                    "compassion", "compassionate",
                    "dedicate", "dedication",
                    "praise",
                    "commend", "commendable",
                    "endear", "endearing",
                    "integrity",
                    "impress", "impressive",
                    "enchant", "enchanting",
                    "revelation",
                    "satisfy", "satisfying", "satisfied", "satisfactory", "satisfyingly",
                    "tender", "tenderness",
                    "heartfelt",
                    "exquisite",
                    "joy", "joyful",
                    "nurture", "nurturing",
                    "refresh", "refreshing",
                    "invigorate", "invigorating",
                    "outstanding",
                    "exception", "exceptional",
                    "celebrate", "celebration",
                    "uplift", "uplifting",
                    "pleasant",
                    "grace", "graceful",
                    "heartwarming",
                    "charm", "charming",
                    "delight", "delightful",
                    "admirable",
                    "reassure", "reassuring",
                    "astound", "astounding",
                    "awe", "awe-inspiring",
                    "allure", "alluring",
                    "appreciate", "appreciation",
                    "breathtaking",
                    "vibrate", "vibrant",
                    "enrich", "enriching",
                    "encouraging",
                    "magnificence", "magnificent",
                    "radiance", "radiant",
                    "phenomenon", "phenomenal",
                    "stunning", "stunned", "stunningly"
                    "brilliance", "brilliant",
                    "value", "valuable",
                    "reward", "rewarding",
                    "treasure", "treasured",
                    "superb",
                    "splendid",
                    "superior",
                    "noteworthy",
                    "noble",
                    "nourish", "nourishing",
                    "positive",
                    "precious",
                    "prosper", "prosperous",
                    "rejuvenate", "rejuvenating",
                    "robust",
                    "sturdy",
                    "sunny",
                    "life", "lively",
                    "trust", "trustworthy",
                    "venerate", "venerable",
                    "victory", "victorious",
                    "wholesome",
                    "wonderful",
                    "worthy",
                    "zealous",
                    "zest", "zestful"
                ]


    # conclusive words that may be symbollic of the end of a review
    # I thought this can help identify sentiment if we weight the positive and negative words that appear after these words
    # I should have done bigrams but I did this prior to that
    lr.conclusive_words = [
                    "however",
                    "conclusion",
                    "opinion",
                    "final",
                    "synopsis",
                    "ultimately",
                    "overall",
                    "summary",
                    "end",
                    "conclusively",
                    "lastly",
                    "thus",
                    "therefore",
                    "hence",
                    "nutshell",
                    "essence",
                    "verdict",
                    "recap",
                    "retrospect",
                    "simply",
                    "brief",
                    "conclude",
                    "wrap",
                    "bottom",
                    "line",
                    "closing",
                    "parting",
                    "endnote",
                    "mark",
                    "give",
                    "rate",
                    "rating",
                    "score",
                    "star",
                    "stars",
                    "film",
                    "movie",
                    "review",
                    "reviews",
                    "critic",
                    "critics",
                    "critique",
                    "critiques",
                    "criticism",
                ]

    # conclusive_words = []
    '''
    Ok so here I am trying to remove the bigrams that have no words in common
    with the union of the negative top N and positive top N
    This should help remove bigrams that are not indicative of sentiment
    '''

    # union of negative and positive words for use in filtering bigrams
    pos_neg_union = set(lr.negatives).union(set(lr.positives))
            
    # remove negative bigrams that contain two words that are not in the union set
    for bigram in lr.negative_bigrams[:]:  # Iterating over a copy using slicing
        if bigram[0] not in pos_neg_union and bigram[1] not in pos_neg_union:
            lr.negative_bigrams.remove(bigram)
                
    # remove positive bigrams that contain two words that are not in the union set
    for bigram in lr.positive_bigrams[:]:  # Iterating over a copy using slicing
        if bigram[0] not in pos_neg_union and bigram[1] not in pos_neg_union:
            lr.positive_bigrams.remove(bigram)

    # print('revised negative bigrams: ' + str(lr.negative_bigrams))
    # print('revised positive bigrams: ' + str(lr.positive_bigrams))

    # make sure these point to the right directories
    lr.train('movie_reviews/train', batch_size=1, n_epochs=1, eta=0.01)
    # lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    results = lr.test('movie_reviews/dev')
    # results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)