# CS542 Fall 2021 Programming Assignment 2
# Logistic Regression Classifier

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random

'''
Computes the logistic function.
'''
def sigma(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, n_features=2):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        # self.class_dict = {'action': 0, 'comedy': 1}
        self.negatives, self.positives = self.load_sentiment_data('opinion-lexicon-English')
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {'num_pos_words': 0, 'num_neg_words': 1}
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

    def load_sentiment_data(self, path):
        negatives = []
        positives = []

        # Dataset from 
        ''' Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
            Proceedings of the ACM SIGKDD International Conference on Knowledge 
            Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
            Washington, USA, 
        '''
        # open text file and read in lines
        with open(os.path.join(path, 'negative-words.txt')) as f:
            for line in f:
                negatives.append(line[:-1]) # remove newline character
        with open(os.path.join(path, 'positive-words.txt')) as f:
            for line in f:
                positives.append(line[:-1]) # remove newline character

        return negatives, positives

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        # BEGIN STUDENT CODE
        # count all neg and pos words in document
        for word in document:
            if word in self.negatives:
                vector[self.feature_dict['num_neg_words']] += 1
            elif word in self.positives:
                vector[self.feature_dict['num_pos_words']] += 1
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
                # print('y_hat calculated')

                # update cross entropy loss
                loss += -np.sum(Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
                # # unsure if we need to do this but this would give the average loss for the batch
                # loss += batch_loss/len(minibatch)
                # print('loss calculated: ' + str(loss))

                # compute gradient
                gradient = np.dot(X.T, y_hat - Y) / len(minibatch)
                # gradient = (X.T @ (y_hat - Y)) / len(minibatch) # should be the same as above
                # print('gradient calculated')

                # update weights (and bias)
                self.theta = self.theta - (eta * gradient)
                # print('weights updated')

                # END STUDENT CODE
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
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (TP + TN) / (TP + TN + FP + FN)

        # print results
        print('Precision: ' + str(round(precision,2)))
        print('Recall: ' + str(round(recall,2)))
        print('F1: ' + str(round(F1,2)))
        print('Accuracy: ' + str(round(accuracy,2)))

        return precision, recall, F1, accuracy

if __name__ == '__main__':
    lr = LogisticRegression(n_features=4)
    # make sure these point to the right directories
    lr.train('movie_reviews/train', batch_size=3, n_epochs=1, eta=0.1)
    # lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    results = lr.test('movie_reviews/dev')
    # results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)