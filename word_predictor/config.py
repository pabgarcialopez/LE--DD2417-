# Universal config
RANDOM_SEED = 0                 # Random seed used for train-val-test split
EXTRINSIC_EVAL_SIZE = 1000      # Number of sentences for keystroke test
NUM_PREDICTIONS = [1,2,3,4]     # Num predictions to be given by model

# Word-id mapping
START_SYMBOL = '<s>'            # Start symbol
PADDING_SYMBOL = ' '            # Padding symbol
UNK_SYMBOL = '<unk>'            # Unknown symbol

# N-gram specific config
TRAINED_PATH = "ngram/"  # Trained models' path
MIN_N = 2  # Smallest N for N-gram model
MAX_N = 5  # Greatest N for N-gram model