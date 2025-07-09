from tqdm import tqdm
from config import START_SYMBOL


def evaluate_extrinsic(ds, model, num_preds, n=0):
    """
    Evaluate the model based on the extrinsic metric of keystroke savings.
    :param ds: list of sentences
    :param model: word predictor model
    :param num_preds: list of numbers of predictions
    """

    for k in num_preds:
        print(f'k={k} suggestion(s)')
        keystrokes = 0
        all_characters = 0
        sum_possible_words = 0
        number_of_words = 0
        for sentence in tqdm(ds, desc="Keystrokes evaluation"):
            sentence = sentence.copy()
            while (n == 0 and sentence[0] != START_SYMBOL) or (n > 1 and (len(sentence) <= n-2 or sentence[n-2] != START_SYMBOL)):
                sentence.insert(0, START_SYMBOL)  # insert the necessary number of start symbols if needed
            start_pred = 1 if n == 0 else n - 1
            for i in range(start_pred, len(sentence)):
                word = sentence[i]
                all_characters += len(word)
                previous_words = sentence[max(0, i-n+1):i] if n > 0 else sentence[:i]
                for j in range(len(word) + 1):
                    pred_words, n_possible_words = model.predict(previous_words, word[:j], k)
                    if word in pred_words[:k]:  # word was predicted
                        keystrokes += j
                        sum_possible_words += n_possible_words
                        break
                    if j == len(word):  # model couldn't predict word.
                        keystrokes += len(word)
                number_of_words += 1

        print(f' Keystrokes: {keystrokes}')
        print(f' All characters: {all_characters}')
        print(f' Keystroke savings: {(1 - keystrokes / all_characters) * 100:.2f}%')
        print(f' Average number of possible words when correctly guessed: {int(sum_possible_words / number_of_words)}')