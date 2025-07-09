import math
import os
import pickle

import numpy as np

from dep_parser import Parser
from logreg import LogisticRegression


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """

    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Builds the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`
        
        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        :param      ds:     Training dataset instance having the feature maps
        """

        # Initialize parsing tree parameters.
        i = 0
        stack = []
        predicted_tree = [0] * len(words)

        # Just for cleannes.
        parser = self.__parser

        # Iterate until
        valid_moves = parser.valid_moves(i, stack, predicted_tree)
        while len(valid_moves) > 0:
            # Get the datapoint associated with the current configuration (attach dummy 1 at the beggining)
            datapoint = np.insert(ds.dp2array(words, tags, i, stack), 0, 1)
            # Let the model predict a move
            predicted_move, prob, bestProb = None, 0, -math.inf
            for move in valid_moves:
                prob = model.conditional_log_prob(move, datapoint)
                if prob > bestProb:
                    bestProb = prob
                    predicted_move = move

            # Once we have predicted a move, update configuration with that move.
            i, stack, predicted_tree = self.__parser.move(i, stack, predicted_tree, predicted_move)

            # Get new valid moves
            valid_moves = self.__parser.valid_moves(i, stack, predicted_tree)

        return predicted_tree

    def evaluate(self, model, test_file, training_ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`

        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      training_ds:         Training dataset instance having the feature maps
        """

        wellParsedSentences, totalSentences, UAScount, totalUAS = 0, 0, 0, 0

        with open(test_file) as source:
            for words, tags, heads, relations in self.__parser.trees(source):
                # Example:
                # words:  ['<ROOT>', 'From', 'the', 'AP', 'comes', 'this', 'story', ':']
                # tags:  ['<ROOT>', 'ADP', 'DET', 'PROPN', 'VERB', 'DET', 'NOUN', 'PUNCT']
                # heads:  [0, 3, 3, 4, 0, 6, 4, 4]
                # relations:  ['root', 'case', 'det', 'nmod', 'root', 'det', 'nsubj', 'punct']

                predicted_heads = self.build(model, words, tags, training_ds)

                if heads == predicted_heads:
                    wellParsedSentences += 1
                    UAScount += len(heads)
                else:
                    for i in range(len(heads)):
                        if heads[i] == predicted_heads[i]:
                            UAScount += 1

                totalSentences += 1
                totalUAS += len(heads) - 1

        print("Sentence-level accuracy: {:.2f}%".format((wellParsedSentences / totalSentences) * 100))
        print("UAS accuracy: {:.2f}%".format((UAScount / totalUAS) * 100))


if __name__ == '__main__':

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)

    # Train LR model
    if os.path.exists('model.pkl'):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open('model.pkl', 'rb'))
        ds.to_arrays()
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        print("Beggining training...", end="\n\n")
        lr.fit(*ds.to_arrays())
        print("Ended training", end="\n\n")
        pickle.dump(lr, open('model.pkl', 'wb'))

    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev-projective.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())

    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev-projective.conllu', ds)
