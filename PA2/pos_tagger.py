	# CS542 Fall 2021 Homework 3
	# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
import random
from random import Random


''' This assignment was too hard for me. I tried for absolutely too long to get this working to no avail.
	I should have asked to set up a meeting outside of office hours to get help but I have a hard time asking for help sometimes.
'''

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
		# looks like we can use the * to unpack the shape tuple since the randn() needs that
		self.initial = np.random.randn(*initial_shape) # create a random normal distribution of weights
		# self.initial = np.array([-0.3, -0.7, 0.3])
		# tag-to-tag transition weights [shape = (len(tag_dict),len(tag_dict))]
		transition_shape = (len(self.tag_dict), len(self.tag_dict))
		self.transition = np.random.randn(*transition_shape)
		# self.transition = np.array([[-0.7, 0.3, -0.3],
		#                             [-0.3, -0.7, 0.3],
		#                             [0.3, -0.3, -0.7]])
		# tag emission weights [shape = (len(word_dict),len(tag_dict))]
		emission_shape = (len(self.word_dict), len(self.tag_dict))
		self.emission = np.random.randn(*emission_shape)
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


	def load_data(self, data_set):
		sentence_ids = []  # doc name + ordinal number of sentence (e.g., ca010)
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
					for sentence in f:  # each line is a sentence
						# 1) create a list of tags and list of words that appear in this sentence
						tag_list = []
						word_list = []
						sentence = sentence.strip()
						if sentence == '':
							continue
						word_tag_pairs = sentence.split()
						for pair in word_tag_pairs:
							word, tag = pair.rsplit('/', maxsplit=1)
							tag_idx = self.tag_dict.get(tag, -1)
							word_idx = self.word_dict.get(word, -1)
							tag_list.append(tag_idx)
							word_list.append(word_idx)

						# 2) create the sentence ID, add it to sentence_ids
						sentence_id = name + str(index)
						sentence_ids.append(sentence_id)

						# 3) add this sentence's tag list to tag_lists and word list to word_lists
						tag_lists[sentence_id] = tag_list
						word_lists[sentence_id] = word_list
						sentences[sentence_id] = sentence

						index += 1
					# END STUDENT CODE

		return sentence_ids, sentences, tag_lists, word_lists

	# Non Broadcasting Version
	# def viterbi(self, sentence):
	# 	T = len(sentence)
	# 	N = len(self.tag_dict)
	# 	v = np.zeros((N, T))
	# 	backpointer = np.zeros((N, T), dtype=int)
	# 	best_path = []

	# 	# BEGIN STUDENT CODE

	# 	# initialization step
	# 	word_idx = self.word_dict.get(sentence[0], -1)  # returns -1 if the word is unknown
	# 	for s in range(N):
	# 		if word_idx != -1:  # if the word is known
	# 			v[s, 0] = self.initial[s] + self.emission[word_idx, s]
	# 		else:
	# 			v[s, 0] = self.initial[s]
	# 		backpointer[s, 0] = 0

	# 	# recursion step
	# 	for t in range(1, T):
	# 		word_idx = self.word_dict.get(sentence[t], -1)  # returns -1 if the word is unknown
	# 		for s in range(N):
	# 			trans_probs = [v[s_prime, t-1] + self.transition[s_prime, s] + self.emission[word_idx, s] for s_prime in range(N)]
	# 			max_trans_prob = max(trans_probs)
	# 			backpointer[s, t] = trans_probs.index(max_trans_prob)
	# 			if word_idx != -1:  # if the word is known
	# 				v[s, t] = max_trans_prob + self.emission[word_idx, s]
	# 			else:
	# 				v[s, t] = max_trans_prob

	# 	# termination step
	# 	# 1) get the most likely ending state, insert it into best_path
	# 	best_last_tag = np.argmax(v[:, T-1])
	# 	best_path.append(best_last_tag)

	# 	# 2) fill out best_path from backpointer trellis
	# 	for t in range(T-1, 0, -1):
	# 		best_path.insert(0, backpointer[best_path[0], t])

	# 	# END STUDENT CODE

	# 	return best_path

	def viterbi(self, sentence):
		T = len(sentence)
		N = len(self.tag_dict)
		v = np.zeros((N, T))
		backpointer = np.zeros((N, T), dtype=int)
		best_path = []

		# BEGIN STUDENT CODE

		# initialization step
		word_idx = self.word_dict.get(sentence[0], -1)  # returns -1 if the word is unknown
		if word_idx != -1:  # if the word is known
			v[:, 0] = self.initial + self.emission[word_idx]
		else:
			v[:, 0] = self.initial  # use emission of 0 if the word is unknown
		backpointer[:, 0] = 0
		# recursion step
		for t in range(1, T):
			word_idx = self.word_dict.get(sentence[t], -1)  # returns -1 if the word is unknown
			
			# Trying broadcasting to calculate the entire column for v at time t
			trans_probs = v[:, t-1, None] + self.transition
			max_trans_prob = np.max(trans_probs, axis=0)
			backpointer[:, t] = np.argmax(trans_probs, axis=0)
			
			if word_idx != -1:  # if the word is known
				v[:, t] = max_trans_prob + self.emission[word_idx]
			else:
				v[:, t] = max_trans_prob 

		# termination step

		best_last_tag = np.argmax(v[:, T-1])
		
		best_path.append(best_last_tag)

		# fill out best_path from backpointer trellis
		for t in range(T-1, 0, -1):
			best_last_tag = backpointer[best_last_tag, t]
			best_path.insert(0, best_last_tag)  # insert at the beginning of the list

		# END STUDENT CODE

		return best_path


	'''
	Trains a structured perceptron part-of-speech tagger on a training set.
	'''
	def train(self, train_set, dummy_data=None):
		self.make_dicts(train_set)
		sentence_ids, sentences, tag_lists, word_lists = self.load_data(train_set)
		# print self.tag_dict and self.word_dict to see what they look like
		# print('tag_dict:', self.tag_dict)
		# print('word_dict:', self.word_dict)
		# print('tag_lists', tag_lists)
		# print('word_lists', word_lists)

		if dummy_data is None:  # for automated testing: DO NOT CHANGE!!
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
			words = word_lists[sentence_id]
			correct_tags = tag_lists[sentence_id]
			# print(f"Processing sentence: {sentences[sentence_id]}")
			# print(f"Correct tags: {correct_tags}")
			# use viterbi to predict
			predicted_tags = self.viterbi(words)
			# print(f"Predicted tags: {predicted_tags} Correct tags: {correct_tags}")

			# if mistake
			if predicted_tags != correct_tags:
				# print("Weights before update:")
				# print(f"Initial: {self.initial}")
				# print(f"Transition: {self.transition}")
				# print(f"Emission: {self.emission}")
				# Update initial weights based on the first tag of the sentence
				# print("increment", correct_tags[0], "decrement", predicted_tags[0])
				self.initial[correct_tags[0]] += 1
				self.initial[predicted_tags[0]] -= 1

				# Unsure about everything below this point...
				# Attempting to update transition and emission weights
				for t, word in enumerate(words):
    				# If the word is not in the dictionary, skip updating the emission weights
					if word not in self.word_dict:
						continue

					word_idx = self.word_dict[word]
					correct_tag_idx = self.tag_dict[correct_tags[t]]
					predicted_tag_idx = self.tag_dict[predicted_tags[t]]

					# Update emission weights
					self.emission[word_idx, correct_tag_idx] += 1
					self.emission[word_idx, predicted_tag_idx] -= 1
					if t > 0:  # there is a previous tag
						prev_correct_tag_idx = self.tag_dict.get(correct_tags[t-1], -1)
						prev_predicted_tag_idx = self.tag_dict.get(predicted_tags[t-1], -1)

						# Update transition weights only if the tag indices exist
						if prev_correct_tag_idx != -1 and correct_tag_idx != -1:
							self.transition[prev_correct_tag_idx, correct_tag_idx] += 1

						if prev_predicted_tag_idx != -1 and predicted_tag_idx != -1:
							self.transition[prev_predicted_tag_idx, predicted_tag_idx] -= 1
				# print("Weights after update:")
				# print(f"Initial: {self.initial}")
				# print(f"Transition: {self.transition}")
				# print(f"Emission: {self.emission}")

			# END STUDENT CODE
			if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
				print(i + 1, 'training sentences tagged')



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
	sentence_ids = ['ca010', 'ca030', 'ca040']
	sentences = {'ca010': 'Alice/nn admired/vb Dorothy/nn',
		'ca030': 'every/dt dwarf/nn cheered/vb',
		'ca040': 'Dorothy/nn admired/vb every/dt dwarf/nn'}
	tag_lists = {'ca010': [0, 1, 0], 'ca030': [2, 0, 1], 'ca040': [0, 1, 2, 0]}
	word_lists = {'ca010': [0, 1, 2], 'ca030': [3, 4, 5], 'ca040': [0, 1, 3, 4]}

	pos.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
	pos.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
						'dwarf': 4, 'cheered': 5}
	pos.initial = np.array([-0.3, -0.7, 0.3])
	pos.transition = np.array([[-0.7, 0.3, -0.3],
								[-0.3, -0.7, 0.3],
								[0.3, -0.3, -0.7]])
	pos.emission = np.array([[-0.3, -0.7, 0.3],
								[0.3, -0.3, -0.7],
								[-0.3, 0.3, -0.7],
								[-0.7, -0.3, 0.3],
								[0.3, -0.7, -0.3],
								[-0.7, 0.3, -0.3]])
	


	pos.train('data_small/train',dummy_data=(sentence_ids,sentences,tag_lists,word_lists))

	expected_best_path = [0, 1, 2, 0]
	best_path = pos.viterbi(word_lists['ca040'])
	print('Expected best path:', expected_best_path)
	print('Output best path:', best_path)

	# pos.train('brown_news/train') # train: news data only
	# pos.train('brown/train') # train: full data
	sentences, results = pos.test('data_small/test') # test: toy data
	# sentences, results = pos.test('brown_news/dev') # test: news data only
	#sentences, results = pos.test('brown/dev') # test: full data
	print('\nAccuracy:', pos.evaluate(sentences, results))



