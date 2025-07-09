import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""


class Generator(object):
    """
    This class generates words from a language model.
    """

    def __init__(self):

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
        self.last_index = -1

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

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """

        print(w, end=" ")
        identifier = self.index[w]
        for i in range(n - 1):

            candidates = []
            for id1 in self.bigram_prob.keys():  # {id1 -> {...}, anotherId1 -> {...}, ...}
                if id1 == identifier:
                    for id2 in self.bigram_prob[id1]:
                        candidates.append((id1, id2))

            probabilities = list(map(lambda bigram: math.exp(self.bigram_prob[bigram[0]][bigram[1]]), candidates))

            if len(candidates) != 0:
                chosenBigram = random.choices(population=candidates, weights=probabilities)
                nextIdentifier = chosenBigram[0][1]
            else:
                nextIdentifier = str(random.randint(0, self.unique_words - 1))

            nextToken = self.word[nextIdentifier]
            print(nextToken, end=" ")

            identifier = nextIdentifier


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str, required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start.lower(), arguments.number_of_words)


if __name__ == "__main__":
    main()
