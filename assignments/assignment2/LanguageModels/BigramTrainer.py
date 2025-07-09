#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import faulthandler
import math
import argparse
import nltk
from collections import defaultdict
import codecs

faulthandler.enable()

"""
This file is part of the computer assignments for the course DD2417 Language Engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file f.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = text_file.read().encode('utf-8').decode().lower()
        try:
            self.tokens = nltk.word_tokenize(text)  # Also lowers words.
        except LookupError:
            nltk.download('punkt')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # If it's a new word.
        if token not in self.index:
            self.index[token] = self.num_unique_words
            self.word[self.num_unique_words] = token
            self.num_unique_words += 1

        self.num_total_words += 1
        self.unigram_count[token] += 1

        # If there are at least two words.
        if self.last_index != -1:
            self.bigram_count[self.last_index][self.index[token]] += 1

        self.last_index = self.index[token]

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
        # Initial row
        rows = [str(self.num_unique_words) + " " + str(self.num_total_words)]

        # V lines
        for i in range(self.num_unique_words):
            rows.append(str(i) + " " + str(self.word[i]) + " " + str(self.unigram_count[self.word[i]]))

        for id1 in self.bigram_count.keys():
            for id2 in self.bigram_count[id1].keys():
                bi_count = self.bigram_count[id1][id2]
                uni_count = self.unigram_count[self.word[id1]]
                log_prob = format(math.log(bi_count) - math.log(uni_count), '.15f')
                rows.append(str(id1) + " " + str(id2) + " " + str(log_prob))

        # Last row marking eof
        rows.append(str(-1))

        return rows

    def __init__(self):
        """
        Constructor.
        """

        self.tokens = None

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.num_unique_words = 0

        # The total number of words in the training corpus.
        self.num_total_words = 0


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str, required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8') as f:
            for row in stats:
                f.write(row + '\n')
    else:
        for row in stats:
            print(row)


if __name__ == "__main__":
    main()
