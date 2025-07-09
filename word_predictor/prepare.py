import io
import nltk
import random
from config import START_SYMBOL, PADDING_SYMBOL, UNK_SYMBOL, RANDOM_SEED


def tokenize_sentences(file, start_symbol=None):
    print("Tokenizing input corpus...")
    with io.open(file, mode='r', encoding='utf-8') as f:
        text = f.read().encode('utf-8').decode('utf-8-sig').lower()
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        sentences = nltk.sent_tokenize(text)

    for (i, s) in enumerate(sentences):
        sentences[i] = [w.lower() for w in nltk.word_tokenize(s)]

    if start_symbol is not None:
        sentences = prepend_start_symbol(sentences, start_symbol)

    print("Tokenization ready.")
    return sentences


def prepend_start_symbol(sentences, start_symbol):
    new_sentences = []
    for (i, s) in enumerate(sentences):
        new_sentences.append(start_symbol + s)
    return new_sentences


def train_val_test_split(sentences, train_split=0.6, validation_split=0.2, test_split=0.2):
    assert train_split + validation_split + test_split <= 1.0
    s = sentences.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(s)
    train_sentences = s[:int(train_split * len(s))]
    validation_sentences = s[int(train_split * len(s)):int((train_split + validation_split) * len(s))]
    test_sentences = s[int((train_split + validation_split) * len(s)):int((train_split + validation_split + test_split) * len(s))]
    print(f'Train dataset size: {len(train_sentences)}')
    print(f'Validation dataset size: {len(validation_sentences)}')
    print(f'Test dataset size: {len(test_sentences)}')
    return train_sentences, validation_sentences, test_sentences


def buildWordIdMappings(sentences):
    w2i = {}
    i2w = []
    w2i[PADDING_SYMBOL] = 0
    i2w.append(PADDING_SYMBOL)
    w2i[START_SYMBOL] = 1
    i2w.append(START_SYMBOL)
    w2i[UNK_SYMBOL] = 2
    i2w.append(UNK_SYMBOL)
    for s in sentences:
        for (i, word) in enumerate(s):
            if word not in w2i:
                w2i[word] = len(w2i)
                i2w.append(word)
    return w2i, i2w