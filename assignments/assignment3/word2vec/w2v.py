import os
import random
import re
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""


class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__U = None  # context vector matrix
        self.__V = None
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = int(dimension)
        self.__lws = int(window_size)
        self.__rws = int(window_size)
        self.__C = self.__lws + self.__rws
        self.__init_lr = float(learning_rate)
        self.__lr = float(learning_rate)
        self.__nsample = int(nsample)
        self.__epochs = int(epochs)
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling

    def init_params(self, W, w2i, i2w):
        self.__W = W
        self.__w2i = w2i
        self.__i2w = i2w

    @property
    def vocab_size(self):
        return self.__V

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        return re.sub(r'[^A-Za-z\s\r]', '', line).split()

    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)

    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        lws, rws = self.__lws, self.__rws

        # Retrieiving the context indices in the sentence
        sl = len(sent)  # Sentence length
        context_indices_in_sent = (list(range(max(0, i - lws), i)) +
                                   list(range(i + 1, min(i + rws + 1, sl))))

        # Getting the respective context indices in the entire corpus
        return [self.__w2i[sent[j]] for j in context_indices_in_sent]

    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """

        # Step 1 (preparing the maps) -----------------------------------------------

        index, total_words = 0, 0
        self.__w2i, self.__i2w, self.unigram_count = {}, {}, {}
        for clean_line in self.text_gen():
            for word in clean_line:
                if word not in self.__w2i:
                    self.__w2i[word] = index
                    self.__i2w[index] = word
                    self.unigram_count[word] = 1
                    index += 1
                else:
                    self.unigram_count[word] += 1
                total_words += 1

        # Step 2 (build unigram and corrected distributions) ------------------------

        # Build unigram distribution:
        self.unigram_dist = {}
        for word, num_times in self.unigram_count.items():
            self.unigram_dist[word] = num_times / total_words

        # Build corrected unigram distribution
        self.corrected_unigram_dist = {}
        denominator_sum = sum(list(map(lambda x: x ** 0.75, self.unigram_dist.values())))
        for word, dist in self.unigram_dist.items():
            self.corrected_unigram_dist[word] = (dist ** 0.75) / denominator_sum

        # Step 3 (build focus and context) ------------------------------------

        focus_words = []
        context_indices = []  # Will turn out to be a matrix
        for clean_line in self.text_gen():
            for i, word in enumerate(clean_line):
                if word not in focus_words:
                    focus_words.append(word)
                    context_indices.append(self.get_context(clean_line, i))
                else:
                    focus_index = focus_words.index(word)
                    context_indices[focus_index].extend(self.get_context(clean_line, i))

        return focus_words, context_indices

    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """

        # Note: For every positive word, need to sample n negative words

        # Use corrected distribution?
        words = self.corrected_unigram_dist.keys() if self.__use_corrected else self.unigram_dist.keys()
        probabilities = self.corrected_unigram_dist.values() if self.__use_corrected else self.unigram_dist.values()

        num_sampled = 0
        negative_samples_indices = set()
        while num_sampled < number:
            sample_word = random.choices(population=list(words), weights=list(probabilities))[0]
            sample_index = self.__w2i[sample_word]
            if sample_index != xb and sample_index != pos:
                negative_samples_indices.add(sample_index)
                num_sampled += 1

        return list(negative_samples_indices)

    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """

        def update_lr(ep, N, i, starting_lr, lr):
            if lr < starting_lr * 0.0001:
                return starting_lr * 0.0001
            else:
                return starting_lr * (1 - (ep * N + i) / (self.__epochs * N + 1))

        # x is the list of focused words.
        # t is the matix of context indeces.
        # N is the number of words.
        x, t = self.skipgram_data()
        N = self.__V = len(x)
        print("Dataset contains {} datapoints".format(N))

        # The parameters we need to learn are two matrices W (focus words) and U (context words),
        # each containing an embedding per word in the vocab V.
        # Each of these embeddings have dimension self.__H

        # Initialize parameters with uniform distribution. We will obtain a matrix of samples of size N x H
        self.__W = np.random.uniform(size=(N, self.__H))
        self.__U = np.random.uniform(size=(N, self.__H))

        # Defining learning rate
        lr = starting_lr = self.__lr

        # To be able to use sigmoid on vectors:
        sigmoid_v = np.vectorize(self.sigmoid)

        for epoch in range(self.__epochs):
            for i in tqdm(range(N)):  # i is the index of the current focus word

                if self.__use_lr_scheduling:
                    lr = update_lr(epoch, N, i, starting_lr, lr)

                # For every positive index
                for pos_id in t[i]:

                    # Get a list of negative samples
                    negative_samples = self.negative_sampling(self.__nsample, i, pos_id)

                    # Accumulate gradient of loss function w.r.t focus word v with positive word.
                    self.__W[i] -= lr * self.__U[pos_id].dot(sigmoid_v(self.__U[pos_id].T.dot(self.__W[i])) - 1)
                    for neg_id in negative_samples:
                        # Accumulate gradient of loss function w.r.t focus word v with negative words.
                        self.__W[i] -= lr * self.__U[neg_id].dot(sigmoid_v(self.__U[neg_id].T.dot(self.__W[i])))

                    # Perform gradient descent of loss function w.r.t the positive word
                    self.__U[pos_id] -= lr * self.__W[i].dot(sigmoid_v(self.__U[pos_id].T.dot(self.__W[i])) - 1)
                    for neg_id in negative_samples:
                        # Perform gradient descent of loss function w.r.t the negative word
                        self.__U[neg_id] -= lr * self.__W[i].dot(sigmoid_v(self.__U[neg_id].T.dot(self.__W[i])))

    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """

        result = []
        estimator = NearestNeighbors(metric=metric).fit(self.__W)

        for word in words:
            if word in self.__w2i:
                context_vector = self.__W[self.__w2i[word]]
                distance, indices_of_closest_words = estimator.kneighbors([context_vector])
                distance, indices_of_closest_words = distance[0], indices_of_closest_words[0]
                # Now we have indices of closest words
                closest_words = []
                for i in range(len(indices_of_closest_words)):
                    index = indices_of_closest_words[i]
                    closest_words.append((self.__i2w[index], distance[i]))
                result.append(closest_words)

        return result

    def write_to_file(self):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("w2v.txt", 'w') as f:
                W = self.__W
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in self.__i2w.items():
                    f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i, :])) + "\n")
        except IOError:
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:]))
                    i2w.append(word)

                w2v.init_params(W, w2i, i2w)
        except IOError:
            print("Error: failing to load the model to the file")
        return w2v

    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file()
        self.interact()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='h1.txt',
                        help='Comma-separated source text files to be trained on')
    parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-d', '--dimension', default=50, help='Dimensionality of word vectors')
    parser.add_argument('-ws', '--window-size', default=2, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save):
        w2v = Word2Vec.load(args.save)
        if w2v:
            w2v.interact()
    else:
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
