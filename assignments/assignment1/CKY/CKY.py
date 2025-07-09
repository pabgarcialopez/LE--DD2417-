from terminaltables import AsciiTable
import argparse

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""


class CKY:
    # The unary rules as a dictionary from words to non-terminals,
    # e.g. { cuts : [Noun, Verb] }
    unary_rules = {}

    # The binary rules as a dictionary of dictionaries. A rule
    # S->NP,VP would result in the structure:
    # { NP : {VP : [S]}} 
    binary_rules = {}

    # The parsing table
    table = []

    # The backpointers in the parsing table
    backptr = []

    # The words of the input sentence
    words = []

    # Reads the grammar file and initializes the 'unary_rules' and
    # 'binary_rules' dictionaries
    def __init__(self, grammar_file):
        stream = open(grammar_file, mode='r', encoding='utf8')
        for line in stream:
            rule = line.split("->")
            left = rule[0].strip()
            right = rule[1].split(',')
            if len(right) == 2:
                # A binary rule
                first = right[0].strip()
                second = right[1].strip()
                if first in self.binary_rules:
                    first_rules = self.binary_rules[first]
                else:
                    first_rules = {}
                    self.binary_rules[first] = first_rules
                if second in first_rules:
                    second_rules = first_rules[second]
                    if left not in second_rules:
                        second_rules.append(left)
                else:
                    second_rules = [left]
                    first_rules[second] = second_rules
            if len(right) == 1:
                # A unary rule
                word = right[0].strip()
                if word in self.unary_rules:
                    word_rules = self.unary_rules[word]
                    if left not in word_rules:
                        word_rules.append(left)
                else:
                    word_rules = [left]
                    self.unary_rules[word] = word_rules

    def areInGrammar(self, d1, d2):
        return d1 in self.binary_rules and d2 in self.binary_rules[d1]

    # Computes symbol dependencies for cell in position (i, j)
    def computeDependencies(self, i, j):

        if i == j:  # Main diagonal
            for symbol in self.unary_rules[self.words[j]]:
                ptr = (symbol, i, j, self.words[j], None, None, self.words[j], None, None)
                self.backptr.append(ptr)
            return self.unary_rules[self.words[j]]

        else:
            dependencies = []
            row = i + 1
            col = i

            # Number of cell comparisons is j - i.
            for _ in range(j - i):
                dependencies1 = self.table[i][col]
                dependencies2 = self.table[row][j]

                if dependencies1 != "" and dependencies2 != "":
                    for d1 in dependencies1:
                        for d2 in dependencies2:
                            if self.areInGrammar(d1, d2):
                                dependencies += self.binary_rules[d1][d2]  # List joining
                                for symbol in self.binary_rules[d1][d2]:
                                    ptr = (symbol, i, j, d1, i, col, d2, row, j)
                                    self.backptr.append(ptr)
                row += 1
                col += 1

            return dependencies

    # Parses the sentence a and computes all the cells in the
    # parse table, and all the backpointers in the table
    def parse(self, s):
        self.words = s.split()
        numWords = len(self.words)
        self.table = [["" for _ in range(numWords)] for _ in range(numWords)]

        # Run the matrix on its diagonals.
        for d in range(numWords):
            for j in range(d, numWords):
                i = j - d  # row
                self.table[i][j] = self.computeDependencies(i, j)

    # Prints the parse table
    def print_table(self):
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print(t.table)

    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    def print_trees(self, row, column, symbol):
        for tup in self.backptr:
            if tup[0] == symbol and tup[1] == row and tup[2] == column:

                # Base case
                if tup[4] is None:
                    print(symbol + "[" + tup[3] + "]", end=""),
                    return

                # Recursive case
                else:
                    print(symbol + "(", end="")
                    self.print_trees(tup[4], tup[5], tup[3])
                    print(" ", end="")
                    self.print_trees(tup[7], tup[8], tup[6])
                    print(")", end="")

                # To separate two trees.
                if row == 0 and column == len(self.table) - 1:
                    print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CKY parser')
    parser.add_argument('--grammar', '-g', type=str, required=True, help='The grammar describing legal sentences.')
    parser.add_argument('--input_sentence', '-i', type=str, required=True, help='The sentence to be parsed.')
    parser.add_argument('--print_parsetable', '-pp', action='store_true', help='Print parsetable')
    parser.add_argument('--print_trees', '-pt', action='store_true', help='Print trees')
    parser.add_argument('--symbol', '-s', type=str, default='S', help='Root symbol')

    arguments = parser.parse_args()

    cky = CKY(arguments.grammar)
    cky.parse(arguments.input_sentence)
    if arguments.print_parsetable:
        cky.print_table()
    if arguments.print_trees:
        cky.print_trees(0, len(cky.table) - 1, arguments.symbol)


if __name__ == '__main__':
    main()
