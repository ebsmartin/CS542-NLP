# CS542 Fall 2021 Homework 3
# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
import random
from random import Random


class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from worked example
        # self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        # self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
        #                   'dwarf': 4, 'cheered': 5}

        self.tag_dict = {}
        self.word_dict = {}


        # initial tag weights [shape = (len(tag_dict),)]
        initial_shape = (len(self.tag_dict),)
        self.initial = np.random.randn(initial_shape) # create a random normal distribution of weights
        # self.initial = np.array([-0.3, -0.7, 0.3])
        # tag-to-tag transition weights [shape = (len(tag_dict),len(tag_dict))]
        transition_shape = (len(self.tag_dict), len(self.tag_dict))
        self.transition = np.random.randn(transition_shape)
        # self.transition = np.array([[-0.7, 0.3, -0.3],
        #                             [-0.3, -0.7, 0.3],
        #                             [0.3, -0.3, -0.7]])
        # tag emission weights [shape = (len(word_dict),len(tag_dict))]
        emission_shape = (len(self.word_dict), len(self.tag_dict))
        self.emission = np.random.randn(emission_shape)
        # self.emission = np.array([[-0.3, -0.7, 0.3],
        #                           [0.3, -0.3, -0.7],
        #                           [-0.3, 0.3, -0.7],
        #                           [-0.7, -0.3, 0.3],
        #                           [0.3, -0.7, -0.3],
        #                           [-0.7, 0.3, -0.3]])
        self.unk_index = -1

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        tag_vocabulary = set()
        word_vocabulary = set()
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # BEGIN STUDENT CODE
                    # create vocabularies of every tag and word that exists in the training data
                    for line in f:
                        line = line.strip()
                        if line == '':
                            continue
                        word_tag_pairs = line.split()
                        for pair in word_tag_pairs:
                            word, tag = pair.rsplit('/', maxsplit=1)
                            tag_vocabulary.add(tag)
                            word_vocabulary.add(word)
                    # END STUDENT CODE
        # create tag_dict and word_dict
        # if you implemented the rest of this function correctly, these should be formatte as they are above in __init__
        self.tag_dict = {v: k for k, v in enumerate(tag_vocabulary)}
        self.word_dict = {v: k for k, v in enumerate(word_vocabulary)}

    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''
    def load_data(self, data_set):
        sentence_ids = [] # doc name + ordinal number of sentence (e.g., ca010)
        sentences = dict()
        tag_lists = dict()
        word_lists = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    # BEGIN STUDENT CODE
                    # for each sentence in the document
                    index = 0
                    for sentence in f: # each line is a sentence
                        #  1) create a list of tags and list of words that appear in this sentence
                        tag_list = []
                        word_list = []
                        sentence = sentence.strip()
                        if sentence == '':
                            continue
                        word_tag_pairs = sentence.split()
                        for pair in word_tag_pairs:
                            word, tag = pair.rsplit('/', maxsplit=1)
                            tag_list.append(tag)
                            word_list.append(word)

                        #  2) create the sentence ID, add it to sentence_ids
                        sentence_id = name + str(index)
                        sentence_ids.append(sentence_id)
                        
                        
                        #  3) add this sentence's tag list to tag_lists and word list to word_lists
                        tag_lists[sentence_id] = tag_list
                        word_lists[sentence_id] = word_list
                        sentences[sentence_id] = sentence
                        
                        index += 1
                    # END STUDENT CODE

        return sentence_ids, sentences, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        best_path = []
        # BEGIN STUDENT CODE
        
        # initialization step
        for s in range(N):
            word_index = self.word_dict.get(sentence[0], self.unk_index)
            v[s, 0] = self.initial[s] + self.emission[word_index, s]
            
        # recursion step
        for t in range(1, T):
            word_index = self.word_dict.get(sentence[t], self.unk_index)
            for s in range(N):
                # 1) fill out the t-th column of viterbi trellis with the max of the t-1-th column of trellis
                #      + transition weights to each state
                #      + emission weights of t-th observateion
                trans_probs = v[:, t-1] + self.transition[:, s]
                max_trans_prob = np.max(trans_probs)
                v[s, t] = max_trans_prob + self.emission[word_index, s]
                
                #  2) fill out the t-th column of the backpointer trellis with the associated argmax values
                backpointer[s, t] = np.argmax(trans_probs)
                
        # termination step
        #  1) get the most likely ending state, insert it into best_path
        best_path.append(np.argmax(v[:, T-1]))
        
        #  2) fill out best_path from backpointer trellis
        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])
        
        # END STUDENT CODE
        return best_path


    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set, dummy_data=None):
        self.make_dicts(train_set)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(train_set)
        if dummy_data is None: # for automated testing: DO NOT CHANGE!!
            Random(0).shuffle(sentence_ids)
            self.initial = np.zeros(len(self.tag_dict))
            self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
            self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        else:
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            # get the word sequence for this sentence and the correct tag sequence
            sentence = word_lists[sentence_id]
            correct_tags = tag_lists[sentence_id]
            # use viterbi to predict
            predicted_tags = self.viterbi(sentence)
            # if mistake
            if predicted_tags != correct_tags:
                for t, word in enumerate(sentence):
                    word_idx = self.word_dict[word]

                    # promote weights that appear in correct sequence
                    self.initial[correct_tags[t]] += 1  # just add one to the weight
                    self.emission[word_idx, correct_tags[t]] += 1 

                    if t > 0:  # if not the first word
                        self.transition[correct_tags[t-1], correct_tags[t]] += 1  # add one to the weight

                    # demote weights that appear in (incorrect) predicted sequence
                    self.initial[predicted_tags[t]] -= 1  # subtract one from the weight
                    self.emission[word_idx, predicted_tags[t]] -= 1

                    if t > 0:  # if not the first word
                        self.transition[predicted_tags[t-1], predicted_tags[t]] -= 1

            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')


# get the word sequence for this sentence and the correct tag sequence
            words = word_lists[sentence_id]
            correct_tags = [self.tag_dict[tag] for tag in tag_lists[sentence_id]]
            
            # use viterbi to predict
            predicted_tags = self.viterbi(words)
            
            # if mistake
            if predicted_tags != correct_tags:
                for t, word in enumerate(words):
                    word_idx = self.word_dict[word]
                    
                    # promote weights that appear in correct sequence
                    self.initial[correct_tags[t]] += 1
                    self.emission[word_idx, correct_tags[t]] += 1
                    if t > 0:
                        self.transition[correct_tags[t-1], correct_tags[t]] += 1
                        
                    # demote weights that appear in (incorrect) predicted sequence
                    self.initial[predicted_tags[t]] -= 1
                    self.emission[word_idx, predicted_tags[t]] -= 1
                    if t > 0:
                        self.transition[predicted_tags[t-1], predicted_tags[t]] -= 1


    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    def test(self, dev_set, dummy_data=None):
        results = defaultdict(dict)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(dev_set)
        if dummy_data is not None: # for automated testing: DO NOT CHANGE!!
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            # should be very similar to train function before mistake check
            # get the word sequence for this sentence and the correct tag sequence
            sentence = word_lists[sentence_id]
            correct_tags = tag_lists[sentence_id]
            # use viterbi to predict
            predicted_tags = self.viterbi(sentence)
            results[sentence_id]['correct'] = correct_tags  # makes a dictionary of dictionaries, for example: results['ca01']['correct'] = ['nn', 'vb', 'dt']
            results[sentence_id]['predicted'] = predicted_tags
            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return sentences, results

    '''
    Given results, calculates overall accuracy.
    This evaluate function calculates accuracy ONLY,
    no precision or recall calculations are required.
    '''
    def evaluate(self, sentences, results, dummy_data=False):
        if not dummy_data:
            self.sample_results(sentences, results)
        accuracy = 0.0
        # BEGIN STUDENT CODE
        # for each sentence, how many words were correctly tagged out of the total words in that sentence?'
        # number of words correctly tagged / total number of words)
        # sum up the number of words correctly tagged for all sentences
        # divide by the total number of words in all sentences
        total_words = 0
        total_correct = 0
        for sentence_id in results:
            correct_tags = results[sentence_id]['correct']
            predicted_tags = results[sentence_id]['predicted']
            total_words += len(correct_tags)
            for i in range(len(correct_tags)):
                if correct_tags[i] == predicted_tags[i]:
                    total_correct += 1
        accuracy = total_correct / total_words
        # END STUDENT CODE
        return accuracy
        
    '''
    Prints out some sample results, with original sentence,
    correct tag sequence, and predicted tag sequence.
    This is just to view some results in an interpretable format.
    You do not need to do anything in this function.
    '''
    def sample_results(self, sentences, results, size=2):
        print('\nSample results')
        results_sample = [random.choice(list(results)) for i in range(size)]
        inv_tag_dict = {v: k for k, v in self.tag_dict.items()}
        for sentence_id in results_sample:
            length = len(results[sentence_id]['correct'])
            correct_tags = [inv_tag_dict[results[sentence_id]['correct'][i]] for i in range(length)]
            predicted_tags = [inv_tag_dict[results[sentence_id]['predicted'][i]] for i in range(length)]
            print(sentence_id,\
                sentences[sentence_id],\
                'Correct:\t',correct_tags,\
                '\n Predicted:\t',predicted_tags,'\n')



if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    pos.train('brown/train') # train: toy data
    #pos.train('brown_news/train') # train: news data only
    #pos.train('brown/train') # train: full data
    sentences, results = pos.test('brown/dev') # test: toy data
    #sentences, results = pos.test('brown_news/dev') # test: news data only
    #sentences, results = pos.test('brown/dev') # test: full data
    print('\nAccuracy:', pos.evaluate(sentences, results))
