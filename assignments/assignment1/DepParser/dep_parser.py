from pathlib import Path
from parse_dataset import Dataset
import argparse
import numpy


def checkLA(stack, pred_tree, correct_tree, numMatching):
    topmostWord = stack[-1]
    secondTopmostWord = stack[-2]
    predTreeCopy = pred_tree.copy()
    predTreeCopy[secondTopmostWord] = topmostWord
    new_numMatching = sum(a == b for a, b in zip(predTreeCopy, correct_tree))
    return new_numMatching == numMatching + 1


def checkRA(stack, pred_tree, correct_tree, numMatching):
    topmostWord = stack[-1]
    secondTopmostWord = stack[-2]
    predTreeCopy = pred_tree.copy()
    predTreeCopy[topmostWord] = secondTopmostWord
    new_numMatchingPairs = sum(elem1 == elem2 for elem1, elem2 in zip(predTreeCopy, correct_tree))
    # Important to note that in order to return RA as the correct decision, two conditions must be met:
    # 1. By choosing RA, there is one more match than before.
    # 2. The dependants of the topmost word have all been asigned (and this translates to having
    #    the same number of references to the topmost word in both trees). If this wasn't checked,
    #    we might be removing a word from the stack that has unassigned dependants.
    return (new_numMatchingPairs == numMatching + 1 and
            predTreeCopy.count(topmostWord) == correct_tree.count(topmostWord))


class Parser:
    SH, LA, RA = 0, 1, 2

    def conllu(self, source):
        buffer = []
        for line in source:
            line = line.rstrip()  # strip off the trailing newline
            if not line.startswith("#"):
                if not line:
                    yield buffer
                    buffer = []
                else:
                    columns = line.split("\t")
                    if columns[0].isdigit():  # skip range tokens
                        buffer.append(columns)

    def trees(self, source):
        """
        Reads trees from an input source.

        Args: source: An iterable, such as a file pointer.

        Yields: Triples of the form `words`, `tags`, heads where: `words`
        is the list of words of the tree (including the pseudo-word
        <ROOT> at position 0), `tags` is the list of corresponding
        part-of-speech tags, and `heads` is the list of head indices
        (one head index per word in the tree).
        """
        for rows in self.conllu(source):
            words = ["<ROOT>"] + [row[1] for row in rows]
            tags = ["<ROOT>"] + [row[3] for row in rows]
            tree = [0] + [int(row[6]) for row in rows]
            relations = ["root"] + [row[7] for row in rows]
            yield words, tags, tree, relations

    def step_by_step(self, string):
        """
        Parses a string and builds a dependency tree. In each step,
        the user needs to input the move to be made.
        """
        w = ("<ROOT> " + string).split()
        i, stack, pred_tree = 0, [], [0] * len(w)  # Input configuration
        while True:
            print("----------------")
            print("Buffer: ", w[i:])
            print("Stack: ", [w[s] for s in stack])
            print("Predicted tree: ", pred_tree)
            try:
                ms = input("Move: (Shift,Left,Right): ").lower()[0]
                m = Parser.SH if ms == 's' else Parser.LA if ms == 'l' else Parser.RA if ms == 'r' else -1
                if m not in self.valid_moves(i, stack, pred_tree):
                    print("Illegal move")
                    continue
            except:
                print("Illegal move")
                continue
            i, stack, pred_tree = self.move(i, stack, pred_tree, m)
            if i == len(w) and stack == [0]:
                # Terminal configuration
                print("----------------")
                print("Final predicted tree: ", pred_tree)
                return

    def create_dataset(self, source, train=False):
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(source) as f:
            for w, tags, tree, relations in self.trees(f):
                i, stack, pred_tree = 0, [], [0] * len(tree)  # Input configuration
                m = self.compute_correct_move(i, stack, pred_tree, tree)
                while m is not None:
                    ds.add_datapoint(w, tags, i, stack, m, train)
                    i, stack, pred_tree = self.move(i, stack, pred_tree, m)
                    m = self.compute_correct_move(i, stack, pred_tree, tree)
        return ds

    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser
        configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        moves = []

        # As long as there's still a word in the buffer, we can shift it.
        if i < len(pred_tree):
            moves.append(self.SH)

        # Since RA will delete the topmost word on the stack,
        # it is an allowed operation as long as there are at
        # least 2 elements in the stack.
        if i >= 2 and len(stack) >= 2:
            moves.append(self.RA)

        # Since LA will delete the second topmost word on the stack,
        # it is an allowed operation as long as there are at
        # least 3 elements in the stack (root must be kept).
        if i >= 3 and len(stack) >= 3:
            moves.append(self.LA)

        return moves

    def move(self, i, stack, pred_tree, move):
        """
        Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """

        # Add unprocessed word to the stack and increment index.
        if move == self.SH:
            stack.append(i)  # We insert the index rather than the word itself because it's more convenient.
            i += 1

        # Create an arc from the topmost word to the second topmost word on the stack
        # then remove the second topmost word from stack
        elif move == self.LA:
            topWord = stack[-1]
            # Remove the 2nd topmost word from stack which also serves as the position in which we update the tree
            index = stack.pop(-2)
            pred_tree[index] = topWord

        # Create an arc from the second topmost word to the topmost word on the stack
        # then remove the topmost word from stack
        elif move == self.RA:
            secondTopWord = stack[-2]
            index = stack.pop(-1)
            pred_tree[index] = secondTopWord

        return i, stack, pred_tree

    def compute_correct_moves(self, tree):
        """
        Computes the sequence of moves (transformations) the parser 
        must perform in order to produce the input tree.
        """
        i, stack, pred_tree = 0, [], [0] * len(tree)  # Input configuration
        moves = []
        m = self.compute_correct_move(i, stack, pred_tree, tree)
        while m is not None:
            moves.append(m)
            i, stack, pred_tree = self.move(i, stack, pred_tree, m)
            m = self.compute_correct_move(i, stack, pred_tree, tree)
        return moves

    def compute_correct_move(self, i, stack, pred_tree, correct_tree):
        """
        Given a parser configuration (i,stack,pred_tree), and 
        the correct final tree, this method computes the  correct 
        move to do in that configuration.
    
        See the textbook, chapter 18.2.1.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            correct_tree: The correct dependency tree.
        
        Returns:
            The correct move for the specified parser
            configuration, or `None` if no move is possible.
        """
        assert len(pred_tree) == len(correct_tree)

        # Parse if there are words left
        if i <= len(pred_tree):

            if i <= 1 and len(stack) <= 1:
                return self.SH  # Can only do SH in this case

            else:
                # Compute the number of matching pairs between the two trees.
                numMatchingPairs = sum(elem1 == elem2 for elem1, elem2 in zip(pred_tree, correct_tree))

                # Precondition to obtain second top most word in stack.
                if len(stack) >= 2:

                    # Check if by choosing LA we increase the number of matches.
                    if checkLA(stack, pred_tree, correct_tree, numMatchingPairs):
                        return self.LA

                    # If LA didn't work, check if by choosing RA we increase the number of matches.
                    if checkRA(stack, pred_tree, correct_tree, numMatchingPairs):
                        return self.RA

                    # pred_tree == correct_tree means to draw a right arc from root to the only word left in the stack.
                    if pred_tree == correct_tree:
                        return self.RA

                    # If nothing worked
                    return self.SH

        return None


filename = Path("en-ud-dev-projective.conllu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transition-based dependency parser')
    parser.add_argument('-s', '--step_by_step', type=str, help='step-by-step parsing of a string')
    parser.add_argument('-m', '--compute_correct_moves', type=str, default=filename,
                        help='compute the correct moves given a correct tree')
    args = parser.parse_args()

    p = Parser()
    if args.step_by_step:
        p.step_by_step(args.step_by_step)

    elif args.compute_correct_moves:
        with open(args.compute_correct_moves, encoding='utf-8') as source:
            for w, tags, tree, relations in p.trees(source):
                print(p.compute_correct_moves(tree))
