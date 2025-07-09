#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
from collections import defaultdict
import codecs

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """

        # Tokens from the test corpus.
        self.tokens = None

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = None

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))

                # V lines
                for _ in range(self.unique_words):
                    identifier, token, num_ocurrences = f.readline().strip().split(' ')
                    self.index[token] = identifier
                    self.word[identifier] = token
                    self.unigram_count[token] = int(num_ocurrences)

                # Read rest of lines
                lines = f.readlines()
                for i in range(len(lines) - 1):
                    id1, id2, logProb = lines[i].strip().split(' ')
                    self.bigram_prob[id1][id2] = float(logProb)

                return True

        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def P(self, word):
        """
        Returns probability of word.
        """
        return int(self.unigram_count[word]) / self.total_words

    def compute_entropy_cumulatively(self, word):

        # Formula for probability: P(wi-1 wi) = λ1 * P(wi | wi-1) + λ2 * P(wi) + λ3
        # Some of its terms might be zero, depending on if we know wi-1 or wi

        # Identifier of the current word (might or not exist in index).
        identifier = self.index[word] if word in self.index else None

        # Second token (wi) is unknown ==> P(wi-1 wi) = λ3
        if identifier is None:
            prob = self.lambda3

        # First token in bigram is unknown but second is known ==> P(wi-1 wi) = λ2 * P(wi) + λ3
        elif self.last_index is None:
            prob = self.lambda2 * self.P(word) + self.lambda3

        # Both tokens are known
        else:
            # And they exist as a biagram ==> P(wi-1 wi) = λ1 * P(wi | wi-1) + λ2 * P(wi) + λ3
            if self.last_index in self.bigram_prob and identifier in self.bigram_prob[self.last_index]:
                prob = (self.lambda1 * math.exp(self.bigram_prob[self.last_index][identifier]) +
                        self.lambda2 * self.P(word) + self.lambda3)
            else:  # Not a biagram in the model ==> P(wi-1 wi) = λ2 * P(wi) + λ3
                prob = self.lambda2 * self.P(word) + self.lambda3

        # len(self.tokens) corresponds to the number of tokens in the test corpus.
        self.logProb += (-1 / len(self.tokens)) * math.log(prob)

        self.test_words_processed += 1
        self.last_index = identifier

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, encoding='utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower())
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed,
                                                                bigram_tester.logProb))


if __name__ == "__main__":
    main()
