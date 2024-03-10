import os
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, data_path: str = "./data.txt"):
        if not os.path.exists(data_path.strip()):
            raise FileNotFoundError("Path to data doesn't exist.")

        if not data_path.endswith(".txt"):
            raise ValueError("The file is not a text file. Only text files are supported.")

        with open(data_path, 'r') as file:
            self.data = file.read().splitlines()

    def split_data(self, train_split: float, dev_test_split: float):
        train_words, temp_words = train_test_split(self.data, train_size=train_split, random_state=42)
        dev_words, test_words = train_test_split(temp_words, test_size=dev_test_split, random_state=42)

        return train_words, dev_words, test_words

    def prepare_xs_ys(self):
        chars = sorted(list(set(''.join(self.train_words))))

        char_to_int = {s: i + 1 for i, s in enumerate(chars)}
        char_to_int['.'] = 0
        # int_to_char seems to be redundant. See if you can remove it safely
        int_to_char = {i: s for s, i in char_to_int.items()}

        two_chars = set()
        for c1 in chars + ["."]:
            for c2 in chars + ["."]:
                two_chars.add(c1 + c2)

        two_chars = sorted(list(two_chars))

        char2_to_int = {s: i for i, s in enumerate(two_chars)}
        int_to_char2 = {i: s for i, s in enumerate(two_chars)}

    