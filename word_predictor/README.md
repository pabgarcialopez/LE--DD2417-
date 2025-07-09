# Word Predictor (DD2417 Language Engineering mini-project)

This repository contains our implementation for the mini-project of the subject DD2417.

## File structure

Note: unless the files are in the same directory, imports will not work.

* `data` directory: training material
  - `HP_all.txt`: the full corpus of the Harry Potter series used for training, validation and testing (sentences are randomly splitted based on the given ratios for these tasks)
  - `guardian_test.txt`: some text from Guardian used for testing how well the models generalize
  - [missing] `glove.6B.50d`: GloVe embeddings (provided for previous assignment), not uploaded due to file size limits, but should be place in the `data` directory
  `ngram` directory: n-gram models are saved here
  `nn` directory: neural network models are saved here
* `config.py`: some configuration options
* `prepare.py`: methods related to text tokenization and preparation, train-val-test splitting, building word-id mappings
* `evaluate.py`: extrinsic evaluation method (number of saved keystrokes test)
* `gui.py`: GUI definition using [Gradio](https://gradio.app)
* `n_gram_trainer.ipynb`: n-gram model trainer jupyter notebook
* `n_gram_tester.ipynb`: n-gram model tester jupyter notebook
* `nn.ipynb`: recurrent neural network-based solution jupyter notebook (contains training and evaluation as well)

## How to run

To run the solutions, open the jupyter notebooks and run the cells. To skip the training of the neural network approach, run each cell except for the ones performing the training, and uncomment the cell calling `load_model` and provide a model path.

To try the GUI, run the (last) cell of `nn.ipynb` or `n_gram_tester.ipynb` with the call to the `get_gui` method.
