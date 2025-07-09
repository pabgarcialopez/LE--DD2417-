import os
import math
import random
import nltk
import numpy as np
import numpy.random as rand
import os.path
import argparse
from collections import defaultdict

"""
Python implementation of the Glove training algorithm from the article by Pennington, Socher and Manning (2014).

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2021,2022 by Johan Boye.
"""


# noinspection PyTypeChecker
class Glove:
    def __init__(self, continue_training, left_window_size, right_window_size, output_file):

        self.output_file = output_file

        self.left_window_size = left_window_size
        self.right_window_size = right_window_size

        # Mapping from words to IDs.
        self.word2id = defaultdict(lambda: None)

        # Mapping from IDs to words.
        self.id2word = defaultdict(lambda: None)

        # Mapping from focus words to neighbours to counts (called X 
        # to be consistent with the notation in the Glove paper).
        self.X = defaultdict(lambda: defaultdict(int))

        # Mapping from word IDs to (focus) word vectors. (called w_vector 
        # to be consistent with the notation in the Glove paper).
        self.w_vector = defaultdict(lambda: None)

        # Mapping from word IDs to (context) word vectors (called w_tilde_vector
        # to be consistent with the notation in the Glove paper)
        self.w_tilde_vector = defaultdict(lambda: None)

        # The ID of the latest encountered new word.
        self.latest_new_word = -1

        # Total number of tokens processed
        self.tokens_processed = 0

        # Dimension of word vectors.
        self.dimension = 50

        # Cutoff for gradient descent.
        self.epsilon = 0.01

        # Initial learning rate.
        self.learning_rate = 0.05

        # The number of times we can tolerate that loss increases
        self.patience = 5

        # Padding at the beginning and end of the token stream
        self.pad_word = '<pad>'

        # Temporary file used for storing the model
        self.temp_file = "temp__.txt"

        # Possibly continue training from pretrained vectors 
        if continue_training and os.path.exists(self.temp_file):
            self.read_temp_file(self.temp_file)

    # ------------------------------------------------------------
    #
    #  Methods for processing all files and computing all counts
    #

    def get_word_id(self, word):
        """ 
        Returns the word ID for a given word. If the word has not
        been encountered before, the necessary data structures for
        that word are initialized.
        """
        word = word.lower()
        if word in self.word2id:
            return self.word2id[word]

        else:
            # This word has never been encountered before. Init all necessary
            # data structures.
            self.latest_new_word += 1
            self.id2word[self.latest_new_word] = word
            self.word2id[word] = self.latest_new_word

            # Initialize arrays with random numbers in [-0.5,0.5].
            w = rand.rand(self.dimension) - 0.5
            self.w_vector[self.latest_new_word] = w
            w_tilde = rand.rand(self.dimension) - 0.5
            self.w_tilde_vector[self.latest_new_word] = w_tilde
            return self.latest_new_word

    def update_counts(self, focus_word, context):
        """
        Updates counts based on the local context window.
        """
        focus_word_id = self.get_word_id(focus_word)
        all_context_words = self.X[focus_word_id]
        if all_context_words is None:
            all_context_words = defaultdict(int)
            self.X[focus_word_id] = all_context_words
        for idx in context:
            count = all_context_words[idx]
            if count is None:
                count = 0
            all_context_words[idx] = count + 1

    def get_context(self, i):
        """
        Returns the context of token no i as a list of word indices.
        
        :param      i:     Index of the focus word in the list of tokens
        :type       i:     int
        """

        lws = self.left_window_size
        rws = self.right_window_size

        context_indices = list(range(max(0, i - lws), i)) + list(range(i + 1, min(i + rws + 1, len(self.tokens))))

        # Getting the respective context indices in the entire corpus
        return [self.get_word_id(self.tokens[j]) for j in context_indices]

    def process_files(self, file_or_dir):
        """
        This function recursively processes all files in a directory.
        
        Each file is tokenized and the tokens are put in the list
        self.tokens. Then each token is processed through the methods
        'get_context' and 'update_counts' above.
        """
        if os.path.isdir(file_or_dir):
            for root, dirs, files in os.walk(file_or_dir):
                for file in files:
                    self.process_files(os.path.join(root, file))
        else:
            print(file_or_dir)
            stream = open(file_or_dir, mode='r', encoding='utf-8', errors='ignore')
            text = stream.read()
            try:
                self.tokens = nltk.word_tokenize(text)
            except LookupError:
                nltk.download('punkt')
                self.tokens = nltk.word_tokenize(text)
            for i, token in enumerate(self.tokens):
                self.tokens_processed += 1
                context = self.get_context(i)
                self.update_counts(token, context)
                if self.tokens_processed % 10000 == 0:
                    print('Processed', "{:,}".format(self.tokens_processed), 'tokens')

    #
    #  End of methods for processing all files and computing all counts
    #
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    #
    #   Loss function, gradient descent, etc.
    #

    def f(self, count):
        """
        The "f" function from the Glove article
        """
        return math.pow(count / 100.0, 0.75) if count < 100 else 1.0

    # noinspection PyUnresolvedReferences
    def loss(self):
        """
        Returns the total loss, computed from all the vectors.
        """
        loss = 0
        V = len(self.word2id)
        for i in range(V):
            wi = self.w_vector[i]
            for j in range(V):
                wj = self.w_tilde_vector[j]
                loss += self.f(self.X[i][j]) * ((wi.dot(wj) - (math.log(self.X[i][j]) if self.X[i][j] > 0 else 0)) ** 2)

        return loss / 2

    # noinspection PyUnresolvedReferences
    def compute_gradient(self, i, j):
        """
        Computes the gradient of the loss function w.r.t. w_vector[i] and
        w.r.t w_tilde_vector[j]
        
        Returns wi_vector_grad, wj_tilde_vector_grad
        """
        Xij = self.X[i][j]
        wi = self.w_vector[i]
        wj = self.w_tilde_vector[j]
        aux = self.f(Xij) * (wi.dot(wj) - math.log(Xij))

        wi_vector_grad = wj * aux
        wj_tilde_vector_grad = wi * aux
        return wi_vector_grad, wj_tilde_vector_grad

    def train(self):
        """
        Trains the vectors using stochastic gradient descent
        """

        def select_words_i():
            co_ocurrences_i = []  # For each word i, how many co ocurrences with the rest of the words.
            for i in range(len(self.X)):
                co_ocurrences_i.append(sum(self.X[i].values()))

            probabilities = []
            X_sum = sum(co_ocurrences_i)
            for j in range(len(self.X)):
                probabilities.append(co_ocurrences_i[j] / X_sum)

            return random.choices(population=range(len(self.X)), weights=probabilities, k=100000)

        def select_word_j(i):
            j = random.randrange(len(self.X[i]))
            while self.X[i][j] <= 0:
                j = random.randrange(len(self.X[i]))
            return j

        iterations = 0
        prev_loss = 0

        while self.patience > 0:

            if iterations % 100000 == 0:
                list_of_word_indeces = select_words_i()

            i = list_of_word_indeces[iterations % 100000]
            j = select_word_j(i)

            # Compute gradient and update word vectors.
            wi_vector_grad, wj_tilde_vector_grad = self.compute_gradient(i, j)
            self.w_vector[i] -= self.learning_rate * wi_vector_grad
            self.w_tilde_vector[j] -= self.learning_rate * wj_tilde_vector_grad

            if iterations % 1000 == 0:
                print("Iter {}".format(iterations))

            # Compute loss every 100000 iterations
            if iterations % 100000 == 0:
                current_loss = self.loss()
                if current_loss > prev_loss:
                    self.patience -= 1
                prev_loss = current_loss

            if iterations % 1000000 == 0:
                self.write_word_vectors_to_file(self.output_file)
                self.write_temp_file(self.temp_file)
                self.learning_rate *= 0.99

            iterations += 1

    #
    #  End of loss function, gradient descent, etc.
    #
    # -------------------------------------------------------

    # -------------------------------------------------------
    #
    #  I/O
    #

    def write_word_vectors_to_file(self, filename):
        """
        Writes the vectors to file. These are the vectors you would
        export and use in another application.
        """
        with open(filename, 'w') as f:
            for idx in self.id2word.keys():
                f.write('{} '.format(self.id2word[idx]))
                for i in self.w_vector[idx]:
                    f.write('{} '.format(i))
                f.write('\n')
        f.close()

    def write_temp_file(self, filename):
        """
        Saves the state of the computation to file, so that
        training can be resumed later.
        """
        with open(filename, 'w') as f:
            f.write('{} '.format(self.learning_rate))
            f.write('\n')
            for idx in self.id2word.keys():
                f.write('{} '.format(self.id2word[idx]))
                for i in list(self.w_vector[idx]):
                    f.write('{} '.format(i))
                for i in list(self.w_tilde_vector[idx]):
                    f.write('{} '.format(i))
                f.write('\n')
        f.close()

    def read_temp_file(self, fname):
        """
        Reads the partially trained model from file, so
        that training can be resumed.
        """
        i = 0
        with open(fname) as f:
            self.learning_rate = float(f.readline())
            for line in f:
                data = line.split()
                w = data[0]
                vec = np.array([float(x) for x in data[1:self.dimension + 1]])
                self.id2word[i] = w
                self.word2id[w] = i
                self.w_vector[i] = vec
                vec = np.array([float(x) for x in data[self.dimension + 1:]])
                self.w_tilde_vector[i] = vec
                i += 1
        f.close()
        self.dimension = len(self.w_vector[0])

    #
    #  End of I/O
    #
    # -------------------------------------------------------


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Glove trainer')
    parser.add_argument('--file', '-f', type=str, default='./data',
                        help='The files used in the training.')
    parser.add_argument('--output', '-o', type=str, default='vectors.txt',
                        help='The file where the vectors are stored.')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')
    parser.add_argument('--continue_training', '-c', action='store_true', default=False,
                        help='Continues training from where it was left off.')

    arguments = parser.parse_args()

    glove = Glove(arguments.continue_training,
                  arguments.left_window_size,
                  arguments.right_window_size,
                  arguments.output)

    glove.process_files(arguments.file)
    print('Processed', "{:,}".format(glove.tokens_processed), 'tokens')
    print('Found', len(glove.word2id), 'unique words')
    glove.train()


if __name__ == '__main__':
    main()
